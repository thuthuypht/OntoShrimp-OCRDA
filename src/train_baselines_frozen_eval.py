#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair multi-seed visual/DA baselines for frozen OntoShrimp OCRDA-v7.4.4 evaluation.

Methods:
  source_only       supervised source model only
  coral             Deep CORAL feature covariance alignment
  mmd               multi-kernel RBF MMD feature alignment
  ccda_no_ontology  class-conditional DA with source/EMA consensus pseudo labels,
                    no ontology graph, no ontology topology

Protocol safeguards:
  * target_adapt labels are never exposed.
  * target_dev labels are diagnostic only and never used to select checkpoints.
  * Dataset 3 final external test is intentionally absent from this script.
  * DA baselines use a FIXED final adaptation epoch, pre-declared before target-dev
    inspection, rather than selecting a target-aware checkpoint.
"""
import argparse, copy, json, math, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_ocrda_v7_4_4 import (
    CLASS_NAMES, CLASS_TO_ID, set_seed, build_transforms,
    ShrimpCSVDataset, ShrimpDualViewCSVDataset, class_balancing,
    VisualModel, evaluate, ema_update_module, js_divergence,
)


def coral_loss(zs, zt):
    """Deep CORAL covariance discrepancy (Sun & Saenko style)."""
    if zs.size(0) < 2 or zt.size(0) < 2:
        return zs.sum() * 0.0
    xs = zs - zs.mean(0, keepdim=True)
    xt = zt - zt.mean(0, keepdim=True)
    cs = xs.t().mm(xs) / max(zs.size(0) - 1, 1)
    ct = xt.t().mm(xt) / max(zt.size(0) - 1, 1)
    d = zs.size(1)
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


def _pairwise_sqdist(x, y):
    return (x.pow(2).sum(1, keepdim=True) + y.pow(2).sum(1, keepdim=True).t() - 2*x@y.t()).clamp_min(0)


def mmd_rbf_loss(zs, zt, multipliers=(0.5, 1.0, 2.0, 4.0)):
    """Unbiased-ish multi-kernel RBF MMD with median heuristic."""
    if zs.size(0) < 2 or zt.size(0) < 2:
        return zs.sum() * 0.0
    z = torch.cat([zs.detach(), zt.detach()], 0)
    with torch.no_grad():
        d = _pairwise_sqdist(z, z)
        vals = d[d > 0]
        med = vals.median() if vals.numel() else torch.tensor(1.0, device=z.device)
        base = med.clamp_min(1e-4)
    dss = _pairwise_sqdist(zs, zs)
    dtt = _pairwise_sqdist(zt, zt)
    dst = _pairwise_sqdist(zs, zt)
    loss = 0.0
    for m in multipliers:
        sigma2 = base * float(m)
        kss = torch.exp(-dss / (2*sigma2))
        ktt = torch.exp(-dtt / (2*sigma2))
        kst = torch.exp(-dst / (2*sigma2))
        loss = loss + kss.mean() + ktt.mean() - 2*kst.mean()
    return loss / len(multipliers)


def ccda_loss(zs, ys, zt, logits_t, p_src_t, p_ema_t,
              conf=0.60, js_max=0.12, pseudo_ce_weight=0.50):
    """Class-conditional DA without ontology, using source/EMA agreement."""
    y_s = p_src_t.argmax(1)
    y_e = p_ema_t.argmax(1)
    cs = p_src_t.max(1).values
    ce = p_ema_t.max(1).values
    js = js_divergence(p_src_t, p_ema_t)
    sel = (y_s == y_e) & (cs >= conf) & (ce >= conf) & (js <= js_max)
    zero = zt.sum() * 0.0
    if not sel.any():
        return zero, {"selected_rate":0.0, "selected_classes":0}

    pseudo = y_e
    l_pseudo = F.cross_entropy(logits_t[sel], pseudo[sel])
    terms=[]
    used=0
    for c in range(len(CLASS_NAMES)):
        ms = ys == c
        mt = sel & (pseudo == c)
        if ms.any() and mt.any():
            ps = F.normalize(zs[ms].mean(0, keepdim=True), dim=1)
            pt = F.normalize(zt[mt].mean(0, keepdim=True), dim=1)
            terms.append(1.0 - (ps*pt).sum())
            used += 1
    l_align = torch.stack(terms).mean() if terms else zero
    return l_align + pseudo_ce_weight*l_pseudo, {
        "selected_rate": float(sel.float().mean().item()),
        "selected_classes": int(used),
        "mean_js": float(js.mean().item()),
    }


def resolve_split(split_dir, name):
    p=Path(split_dir)/name
    if not p.exists(): raise FileNotFoundError(p)
    return p


def build_loaders(args):
    st, tw, ts, test = build_transforms(args.image_size)
    sp=Path(args.split_dir)
    src_train=ShrimpCSVDataset(resolve_split(sp,args.source_train_file),args.sdbd_root,args.tsbd_root,st,True)
    src_val=ShrimpCSVDataset(resolve_split(sp,args.source_val_file),args.sdbd_root,args.tsbd_root,test,True)
    src_test=ShrimpCSVDataset(resolve_split(sp,args.source_test_file),args.sdbd_root,args.tsbd_root,test,True)
    tgt_train=ShrimpDualViewCSVDataset(resolve_split(sp,args.target_adapt_file),args.sdbd_root,args.tsbd_root,tw,ts)
    tgt_dev=None
    if args.target_dev_file:
        p=Path(sp)/args.target_dev_file
        if p.exists(): tgt_dev=ShrimpCSVDataset(p,args.sdbd_root,args.tsbd_root,test,True)
    sampler, ce_w=class_balancing(src_train)
    pin=torch.cuda.is_available()
    loaders={
        'src_train':DataLoader(src_train,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=pin,drop_last=True),
        'src_val':DataLoader(src_val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers),
        'src_test':DataLoader(src_test,batch_size=args.batch_size,shuffle=False,num_workers=args.workers),
        'tgt_train':DataLoader(tgt_train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=pin,drop_last=True),
        'tgt_dev':DataLoader(tgt_dev,batch_size=args.batch_size,shuffle=False,num_workers=args.workers) if tgt_dev else None,
    }
    return loaders, ce_w


def warm_cache_path(args, seed):
    d=Path(args.source_cache_dir); d.mkdir(parents=True,exist_ok=True)
    return d/f"visual_sourcewarm_seed_{seed}.pt"


def make_visual(args, device):
    return VisualModel(args.emb_dim,pretrained=not args.no_pretrained,dropout=args.visual_dropout).to(device)


def optimizer_for(model,args,total_epochs):
    opt=torch.optim.AdamW([
        {'params':model.backbone.parameters(),'lr':args.backbone_lr},
        {'params':list(model.proj.parameters())+list(model.source_head.parameters())+list(model.target_head.parameters()),'lr':args.head_lr},
    ],weight_decay=args.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(total_epochs,1),eta_min=args.min_lr)
    return opt,sch


def ensure_source_cache(args, seed, loaders, ce_w, device):
    cp=warm_cache_path(args,seed)
    if cp.exists(): return cp
    print(f"[baseline seed={seed}] building PURE source-warm cache")
    set_seed(seed)
    model=make_visual(args,device)
    opt,sch=optimizer_for(model,args,args.source_epochs)
    crit=nn.CrossEntropyLoss(weight=ce_w.to(device),label_smoothing=args.label_smoothing)
    for ep in range(args.source_epochs):
        model.train(); loss_sum=0; n=0
        for x,y,_ in loaders['src_train']:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            _,logits=model(x,head='source')
            loss=crit(logits,y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            opt.step(); loss_sum += float(loss.detach().cpu()); n+=1
        sch.step()
        vm,_=evaluate(model,loaders['src_val'],device,head='source')
        print(f"  source warm ep={ep+1:02d} loss={loss_sum/max(n,1):.4f} valF1={vm['macro_f1']:.4f}")
    model.sync_target_from_source()
    torch.save({'seed':seed,'model_state_dict':model.state_dict(),'source_epochs':args.source_epochs},cp)
    return cp


def run_method(args, seed):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    loaders,ce_w=build_loaders(args)
    cp=ensure_source_cache(args,seed,loaders,ce_w,device)
    model=make_visual(args,device)
    model.load_state_dict(torch.load(cp,map_location=device)['model_state_dict'])
    model.sync_target_from_source()
    source_teacher=copy.deepcopy(model).eval()
    for p in source_teacher.parameters(): p.requires_grad_(False)

    seed_dir=Path(args.output_dir)/f"seed_{seed}"
    seed_dir.mkdir(parents=True,exist_ok=True)

    warm_val,_=evaluate(model,loaders['src_val'],device,head='source')
    history=[]

    if args.method != 'source_only':
        ema=copy.deepcopy(model).eval()
        for p in ema.parameters(): p.requires_grad_(False)
        opt,sch=optimizer_for(model,args,args.adapt_epochs)
        crit=nn.CrossEntropyLoss(weight=ce_w.to(device),label_smoothing=args.label_smoothing)
        for ep in range(args.adapt_epochs):
            model.train(); ema.eval(); source_teacher.eval()
            tgt_it=iter(loaders['tgt_train']); sums=defaultdict(float); nb=0
            for step,(xs,ys,_) in enumerate(loaders['src_train']):
                if args.max_steps_per_epoch>0 and step>=args.max_steps_per_epoch: break
                try: xw,xt,_,_=next(tgt_it)
                except StopIteration:
                    tgt_it=iter(loaders['tgt_train']); xw,xt,_,_=next(tgt_it)
                xs=xs.to(device); ys=ys.to(device); xw=xw.to(device); xt=xt.to(device)
                zs,ls,lt_src=model(xs,return_both=True)
                zt,_,lt=model(xt,return_both=True)
                lsrc=crit(ls,ys) + args.lambda_target_source_anchor*crit(lt_src,ys)
                if args.method=='coral':
                    lda=coral_loss(zs,zt); diag={}
                elif args.method=='mmd':
                    lda=mmd_rbf_loss(zs,zt); diag={}
                elif args.method=='ccda_no_ontology':
                    with torch.no_grad():
                        _,log_s=source_teacher(xw,head='source')
                        _,log_e=ema(xw,head='target')
                        ps=F.softmax(log_s,1); pe=F.softmax(log_e,1)
                    lda,diag=ccda_loss(zs,ys,zt,lt,ps,pe,args.ccda_conf,args.ccda_js_max,args.ccda_pseudo_ce_weight)
                else: raise ValueError(args.method)
                loss=lsrc + args.lambda_da*lda
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); opt.step()
                ema_update_module(ema,model,args.teacher_ema)
                sums['loss']+=float(loss.detach().cpu()); sums['da']+=float(lda.detach().cpu()); nb+=1
                for k,v in diag.items(): sums[k]+=float(v)
            sch.step()
            vm,_=evaluate(ema,loaders['src_val'],device,head='source')
            rec={'epoch':ep+1,'loss':sums['loss']/max(nb,1),'da_loss':sums['da']/max(nb,1),'source_val_macro_f1':vm['macro_f1']}
            for k in ['selected_rate','selected_classes','mean_js']:
                if k in sums: rec[k]=sums[k]/max(nb,1)
            history.append(rec)
            print(f"[{args.method} seed={seed}] adapt ep={ep+1:02d} loss={rec['loss']:.4f} da={rec['da_loss']:.4f} srcValF1={vm['macro_f1']:.4f}")
        final_model=ema
    else:
        final_model=model

    sm,_=evaluate(final_model,loaders['src_test'],device,head='source')
    td=None
    if loaders['tgt_dev'] is not None:
        td,_=evaluate(final_model,loaders['tgt_dev'],device,head='target')
    result={
        'seed':seed,'method':args.method,'method_version':'OCRDA-v7.4.4-frozen-eval-baseline-suite-v1',
        'source_warm_val':warm_val,'source_test':sm,'target_dev':td,
        'checkpoint_policy':'fixed final adaptation epoch; target-dev labels diagnostic only',
        'safeguards':{'target_adapt_labels_used':False,'target_dev_used_for_checkpoint_selection':False,'dataset3_final_used':False},
        'hyperparameters':vars(args),
    }
    (seed_dir/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if history: pd.DataFrame(history).to_csv(seed_dir/'history.csv',index=False)
    torch.save({'state_dict':final_model.state_dict(),'result':result},seed_dir/'best_model.pt')
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',choices=['source_only','coral','mmd','ccda_no_ontology'],required=True)
    ap.add_argument('--split-dir',required=True); ap.add_argument('--sdbd-root',required=True); ap.add_argument('--tsbd-root',required=True)
    ap.add_argument('--output-dir',required=True); ap.add_argument('--source-cache-dir',required=True)
    ap.add_argument('--source-train-file',default='source_train_final.csv'); ap.add_argument('--source-val-file',default='source_val_final.csv'); ap.add_argument('--source-test-file',default='source_test_final.csv')
    ap.add_argument('--target-adapt-file',default='target_adapt_final.csv'); ap.add_argument('--target-dev-file',default='target_dev_exposed.csv')
    ap.add_argument('--seeds',nargs='+',type=int,default=[2026])
    ap.add_argument('--source-epochs',type=int,default=8); ap.add_argument('--adapt-epochs',type=int,default=22); ap.add_argument('--max-steps-per-epoch',type=int,default=72)
    ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--image-size',type=int,default=224); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--emb-dim',type=int,default=256); ap.add_argument('--visual-dropout',type=float,default=0.20); ap.add_argument('--no-pretrained',action='store_true')
    ap.add_argument('--backbone-lr',type=float,default=5e-5); ap.add_argument('--head-lr',type=float,default=2e-4); ap.add_argument('--min-lr',type=float,default=1e-6); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--grad-clip',type=float,default=5.0); ap.add_argument('--label-smoothing',type=float,default=0.05)
    ap.add_argument('--teacher-ema',type=float,default=0.996); ap.add_argument('--lambda-target-source-anchor',type=float,default=0.20); ap.add_argument('--lambda-da',type=float,default=0.50)
    ap.add_argument('--ccda-conf',type=float,default=0.60); ap.add_argument('--ccda-js-max',type=float,default=0.12); ap.add_argument('--ccda-pseudo-ce-weight',type=float,default=0.50)
    args=ap.parse_args()
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    results=[run_method(args,s) for s in args.seeds]
    print(json.dumps({'method':args.method,'n_seeds':len(results)},indent=2))

if __name__=='__main__': main()
