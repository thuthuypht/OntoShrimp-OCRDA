from __future__ import annotations
import argparse, csv, hashlib, json, math, os, re, shutil, sys, tempfile, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from inference_model import VisualModel, test_transform, CLASS_NAMES

METHODS=['source_only','coral','mmd','ccda_no_ontology','relation_blind','shuffled_ontology','ocrda_no_reliability','ocrda_no_topology','full_ocrda']
SEEDS=[42,123,2026,3407,7777]
CLASS_TO_ID={c:i for i,c in enumerate(CLASS_NAMES)}
RUN_NAME='OCRDA_v7_4_4_FROZEN_DATASET3_EXTERNAL_EVAL_RUN'
PROTOCOL_ID='OCRDA-v7.4.4-frozen-dataset3-external-eval-v1'


def now(): return datetime.now(timezone.utc).isoformat()
def sha256_file(p,chunk=1<<20):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def find_dataset3(input_root:Path, work:Path):
    # Preferred: already unpacked by Kaggle.
    hits=list(input_root.rglob('02_recommended_final_external_test_balanced_150'))
    hits=[p for p in hits if p.is_dir()]
    if hits:
        return sorted(hits,key=lambda p:len(str(p)))[0], 'unpacked-input'
    # Also allow attached ZIP.
    zips=[]
    for p in input_root.rglob('*.zip'):
        if 'dataset3' in p.name.lower() and 'vietnam' in p.name.lower(): zips.append(p)
    if zips:
        z=sorted(zips,key=lambda p:len(str(p)))[0]
        dest=work/'_dataset3_extract'
        if dest.exists(): shutil.rmtree(dest)
        dest.mkdir(parents=True)
        with zipfile.ZipFile(z) as zz: zz.extractall(dest)
        hits=list(dest.rglob('02_recommended_final_external_test_balanced_150'))
        if hits: return hits[0], str(z)
    raise FileNotFoundError('Dataset3 balanced external-test folder not found. Attach Dataset3_Vietnam_v1 to Kaggle Input.')

def verify_dataset3(root:Path):
    lock=pd.read_csv(HERE/'DATASET3_LOCK_MANIFEST.csv')
    errors=[]; found=[]
    for _,r in lock.iterrows():
        p=root/str(r['relative_path'])
        if not p.exists(): errors.append(f'MISSING {r["relative_path"]}'); continue
        hh=sha256_file(p)
        if hh!=r['sha256']: errors.append(f'HASH_MISMATCH {r["relative_path"]}')
        found.append((str(r['class']),str(r['filename']),p,hh))
    extra=[]
    for cls in CLASS_NAMES:
        d=root/cls
        if not d.is_dir(): errors.append(f'MISSING_CLASS_DIR {cls}'); continue
        expected=set(lock.loc[lock['class']==cls,'filename'].astype(str))
        actual={p.name for p in d.iterdir() if p.is_file()}
        extra += [f'{cls}/{x}' for x in sorted(actual-expected)]
    counts=Counter(x[0] for x in found)
    if counts != Counter({'BG':50,'Healthy':50,'WSSV':50}): errors.append(f'BAD_COUNTS {dict(counts)}')
    if extra: errors.append('EXTRA_FILES '+','.join(extra[:20]))
    if errors:
        raise RuntimeError('Dataset3 lock verification FAILED:\n'+'\n'.join(errors[:100]))
    return found

class DS3Dataset(Dataset):
    def __init__(self,records):
        self.records=records; self.tf=test_transform(224)
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        cls,fn,p,_=self.records[i]
        with Image.open(p) as im: x=self.tf(im.convert('RGB'))
        return x,CLASS_TO_ID[cls],f'{cls}/{fn}'

# -------- checkpoint discovery --------
MODEL_ZIP_RE=re.compile(r'OCRDA_v7_4_4_FROZEN_MODEL_([A-Z0-9_]+)_SEED_(42|123|2026|3407|7777)\\.zip$',re.I)
METHOD_UP={m.upper():m for m in METHODS}

def infer_method_seed_from_path(p:Path):
    s=str(p).replace('\\\\','/').lower()
    seed=None
    mm=re.search(r'seed[_-](42|123|2026|3407|7777)',s)
    if mm: seed=int(mm.group(1))
    method=None
    for m in METHODS:
        if f'/{m}/' in s or m in p.name.lower():
            method=m; break
    return method,seed

def discover_models(input_root:Path):
    idx={}; evidence={}; dup=[]
    # Unpacked .pt checkpoints.
    for p in input_root.rglob('*.pt'):
        if p.name not in {'compact_model_fp16.pt','best_model.pt'} and 'compact' not in p.name.lower(): continue
        m,s=infer_method_seed_from_path(p)
        if m in METHODS and s in SEEDS:
            k=(m,s)
            if k in idx: dup.append((k,str(idx[k]),str(p)))
            else: idx[k]=('pt',p); evidence[k]=str(p)
    # Model ZIPs preserved as zip files.
    for z in input_root.rglob('*.zip'):
        mo=MODEL_ZIP_RE.search(z.name)
        if not mo: continue
        up=mo.group(1).upper(); seed=int(mo.group(2)); method=METHOD_UP.get(up.lower().upper())
        # More robust mapping by exact known uppercase name.
        method=None
        for m in METHODS:
            if up==m.upper(): method=m; break
        if method is None: continue
        k=(method,seed)
        if k in idx: dup.append((k,str(idx[k]),str(z)))
        else: idx[k]=('zip',z); evidence[k]=str(z)
    return idx,evidence,dup

def materialize_checkpoint(spec,work:Path,method,seed):
    typ,p=spec
    if typ=='pt': return p
    dest=work/'_model_extract'/method/f'seed_{seed}'
    dest.mkdir(parents=True,exist_ok=True)
    out=dest/'compact_model_fp16.pt'
    if out.exists(): return out
    with zipfile.ZipFile(p) as z:
        candidates=[n for n in z.namelist() if n.endswith('compact_model_fp16.pt') or n.endswith('best_model.pt')]
        if not candidates: raise RuntimeError(f'No model checkpoint inside {p}')
        data=z.read(candidates[0]); out.write_bytes(data)
    return out

def load_visual_checkpoint(p:Path,method:str,device):
    ck=torch.load(p,map_location='cpu')
    emb_dim=256; dropout=0.20
    args=ck.get('args') if isinstance(ck,dict) else None
    if isinstance(args,dict):
        emb_dim=int(args.get('emb_dim',256)); dropout=float(args.get('visual_dropout',0.20))
    model=VisualModel(emb_dim=emb_dim,dropout=dropout)
    if isinstance(ck,dict) and 'teacher_visual_state_dict' in ck:
        sd=ck['teacher_visual_state_dict']
    elif isinstance(ck,dict) and 'state_dict' in ck:
        sd=ck['state_dict']
    elif isinstance(ck,dict) and 'model_state_dict' in ck:
        sd=ck['model_state_dict']
    else:
        raise RuntimeError(f'Unsupported checkpoint format: {p}')
    # fp16 compact state is safely converted to module dtype during load.
    model.load_state_dict(sd,strict=True)
    model.to(device).eval()
    return model,ck

@torch.no_grad()
def evaluate_one(model,loader,device):
    ys=[]; ps=[]; probs=[]; ids=[]
    for x,y,sid in loader:
        x=x.to(device,non_blocking=True)
        _,logits=model(x,head='target')
        pr=F.softmax(logits,dim=1).cpu().numpy(); pred=pr.argmax(1)
        ys.extend(y.numpy().tolist()); ps.extend(pred.tolist()); probs.extend(pr.tolist()); ids.extend(list(sid))
    precision,recall,f1,support=precision_recall_fscore_support(ys,ps,labels=[0,1,2],zero_division=0)
    met={
      'accuracy':float(accuracy_score(ys,ps)),
      'macro_f1':float(f1_score(ys,ps,average='macro',zero_division=0)),
      'balanced_accuracy':float(balanced_accuracy_score(ys,ps)),
      'mcc':float(matthews_corrcoef(ys,ps)),
      'confusion_matrix':confusion_matrix(ys,ps,labels=[0,1,2]).tolist(),
      'per_class':{CLASS_NAMES[i]:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i in range(3)}
    }
    predrows=[]
    for sid,y,p,pr in zip(ids,ys,ps,probs):
        predrows.append({'sample_id':sid,'y_true':CLASS_NAMES[y],'y_pred':CLASS_NAMES[p],'prob_BG':pr[0],'prob_Healthy':pr[1],'prob_WSSV':pr[2]})
    return met,predrows

def result_path(results,m,s): return results/m/f'seed_{s}'/'dataset3_external_metrics.json'
def completed(results,m,s): return result_path(results,m,s).exists()

def write_state(run_root,current_seed=None,current_method=None):
    obj={'protocol_id':PROTOCOL_ID,'updated_utc':now(),'current_seed':current_seed,'current_method':current_method,
         'completed':[{'method':m,'seed':s} for s in SEEDS for m in METHODS if completed(run_root/'results',m,s)],
         'dataset3_opened':True,'development_locked':True,'training_performed':False,'tuning_performed':False}
    st=run_root/'state';st.mkdir(parents=True,exist_ok=True);(st/'external_eval_manifest.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    return obj

def aggregate(run_root):
    results=run_root/'results'; out=run_root/'summary'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for m in METHODS:
      for s in SEEDS:
        p=result_path(results,m,s)
        if not p.exists(): continue
        x=json.loads(p.read_text())['dataset3_external']
        rows.append({'method':m,'seed':s,'macro_f1':x['macro_f1'],'accuracy':x['accuracy'],'balanced_accuracy':x['balanced_accuracy'],'mcc':x['mcc'],
          'BG_f1':x['per_class']['BG']['f1'],'Healthy_f1':x['per_class']['Healthy']['f1'],'WSSV_f1':x['per_class']['WSSV']['f1'],
          'BG_precision':x['per_class']['BG']['precision'],'BG_recall':x['per_class']['BG']['recall']})
    df=pd.DataFrame(rows); df.to_csv(out/'per_seed_external_metrics.csv',index=False)
    ag=[]
    if len(df):
      for m,g in df.groupby('method',sort=False):
        r={'method':m,'n':len(g)}
        for c in ['macro_f1','accuracy','balanced_accuracy','mcc','BG_f1','Healthy_f1','WSSV_f1','BG_precision','BG_recall']:
          vals=g[c].to_numpy(float); mean=float(vals.mean()); sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
          # two-sided 95% t CI for n=5; scipy is available on Kaggle, otherwise normal approximation.
          try:
            from scipy.stats import t
            crit=float(t.ppf(0.975,len(vals)-1)) if len(vals)>1 else 0.0
          except Exception: crit=1.96
          half=crit*sd/math.sqrt(len(vals)) if len(vals)>1 else 0.0
          r[c+'_mean']=mean;r[c+'_sd']=sd;r[c+'_ci95_low']=mean-half;r[c+'_ci95_high']=mean+half
        ag.append(r)
    pd.DataFrame(ag).to_csv(out/'aggregate_external_mean_std_ci.csv',index=False)
    # Paired seed differences: Full - comparator.
    diffs=[]
    if len(df):
      full=df[df.method=='full_ocrda'].set_index('seed')
      for m in METHODS:
        if m=='full_ocrda': continue
        gg=df[df.method==m].set_index('seed'); common=sorted(set(full.index)&set(gg.index))
        if not common: continue
        for metric in ['macro_f1','BG_f1','Healthy_f1','WSSV_f1','balanced_accuracy','mcc']:
          d=(full.loc[common,metric]-gg.loc[common,metric]).to_numpy(float)
          rec={'comparator':m,'metric':metric,'n':len(d),'full_minus_comparator_mean':float(d.mean()),'sd':float(d.std(ddof=1)) if len(d)>1 else 0.0}
          if len(d)>=2:
            try:
              from scipy.stats import wilcoxon
              rec['wilcoxon_p_two_sided']=float(wilcoxon(d,alternative='two-sided',zero_method='wilcox').pvalue) if np.any(d!=0) else 1.0
            except Exception: rec['wilcoxon_p_two_sided']=None
          diffs.append(rec)
    pd.DataFrame(diffs).to_csv(out/'paired_external_differences_vs_full.csv',index=False)
    # Markdown summary
    lines=['# Dataset 3 Final External Evaluation','',f'Protocol: `{PROTOCOL_ID}`','',
           '> Dataset 3 is a locked final external test. These results must not be used for further tuning or model selection.','']
    if ag:
      lines += ['| Method | Macro-F1 mean±SD | BG F1 | Healthy F1 | WSSV F1 |','|---|---:|---:|---:|---:|']
      amap={r['method']:r for r in ag}
      for m in METHODS:
        if m not in amap: continue
        r=amap[m]; lines.append(f"| {m} | {r['macro_f1_mean']:.3f} ± {r['macro_f1_sd']:.3f} | {r['BG_f1_mean']:.3f} | {r['Healthy_f1_mean']:.3f} | {r['WSSV_f1_mean']:.3f} |")
    (out/'SUMMARY_EXTERNAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def make_zip(work,run_root,final=False):
    name='OCRDA_v7_4_4_DATASET3_EXTERNAL_EVAL_REPORTS_FINAL.zip' if final else 'OCRDA_v7_4_4_DATASET3_EXTERNAL_EVAL_REPORTS_LATEST.zip'
    zp=work/name; zp.unlink(missing_ok=True)
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
      for sub in ['results','summary','state']:
        d=run_root/sub
        if d.exists():
          for p in d.rglob('*'):
            if p.is_file(): z.write(p,Path(RUN_NAME)/sub/p.relative_to(d))
      for p in ['PRIMARY_ENDPOINT_LOCK.json','DATASET3_LOCK_MANIFEST.csv','README_KAGGLE.md']:
        z.write(HERE/p,Path(RUN_NAME)/p)
    return zp

def restore_resume(input_root,run_root):
    restored=[]
    # zipped external-eval resumes/reports
    zips=[p for p in input_root.rglob('*.zip') if 'DATASET3_EXTERNAL_EVAL' in p.name.upper() and ('REPORTS' in p.name.upper() or 'RESUME' in p.name.upper())]
    for z in zips:
      try:
        with zipfile.ZipFile(z) as zz:
          names=zz.namelist()
          if not any('dataset3_external_metrics.json' in n for n in names): continue
          tmp=run_root.parent/'_resume_extract'; tmp.mkdir(exist_ok=True)
          zz.extractall(tmp)
          srcs=list(tmp.rglob(RUN_NAME))
          if srcs:
            shutil.copytree(srcs[0],run_root,dirs_exist_ok=True); restored.append(str(z))
      except Exception: pass
    # unpacked resume dataset
    for mf in input_root.rglob('external_eval_manifest.json'):
      base=mf.parent.parent
      if base.name==RUN_NAME:
        shutil.copytree(base,run_root,dirs_exist_ok=True); restored.append(str(base))
    return restored

def main():
    ap=argparse.ArgumentParser(description='Inference-only locked Dataset3 external evaluator. NO TRAINING.')
    ap.add_argument('--input-root',default='/kaggle/input'); ap.add_argument('--work-root',default='/kaggle/working')
    ap.add_argument('--max-new-seeds',type=int,default=1); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--inventory-only',action='store_true')
    a=ap.parse_args()
    input_root=Path(a.input_root); work=Path(a.work_root); work.mkdir(parents=True,exist_ok=True)
    run_root=work/RUN_NAME; run_root.mkdir(parents=True,exist_ok=True)
    restored=restore_resume(input_root,run_root)
    dsroot,ds_source=find_dataset3(input_root,work); records=verify_dataset3(dsroot)
    idx,evidence,dups=discover_models(input_root)
    required=[(m,s) for s in SEEDS for m in METHODS]; missing=[k for k in required if k not in idx]
    print('='*88); print('OCRDA-v7.4.4 FROZEN DATASET3 EXTERNAL EVALUATION — INFERENCE ONLY')
    print('PROTOCOL:',PROTOCOL_ID); print('Dataset3:',dsroot,'source=',ds_source); print('Dataset3 lock: PASS (150 images; 50/class; SHA256 verified)')
    print('Frozen checkpoints found:',len(idx),'/ 45'); print('Restored external state:',restored or 'none')
    if dups: print('WARNING duplicate checkpoint candidates:',len(dups))
    if missing:
      print('\nMISSING FROZEN CHECKPOINTS:')
      for m,s in missing: print(f'  - {m} seed={s}')
      print('\nNO RETRAINING FALLBACK IS PERMITTED.')
      print('Attach the missing OCRDA_v7_4_4_FROZEN_MODEL_<METHOD>_SEED_<seed>.zip files (or their unpacked compact_model_fp16.pt files).')
      if not a.inventory_only: raise SystemExit(3)
    if a.inventory_only:
      print('\nINVENTORY ONLY: no Dataset3 inference executed.'); return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    loader=DataLoader(DS3Dataset(records),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=(device.type=='cuda'))
    new_seeds=0
    for seed in SEEDS:
      if all(completed(run_root/'results',m,seed) for m in METHODS):
        print(f'SKIP seed {seed}: all 9 methods already externally evaluated.'); continue
      if new_seeds>=a.max_new_seeds: break
      print(f'\n=== EXTERNAL EVAL SEED {seed} ===')
      for method in METHODS:
        if completed(run_root/'results',method,seed): print('SKIP',method,seed); continue
        spec=idx[(method,seed)]; cp=materialize_checkpoint(spec,work,method,seed); cp_hash=sha256_file(cp)
        model,ck=load_visual_checkpoint(cp,method,device)
        met,preds=evaluate_one(model,loader,device)
        sd=run_root/'results'/method/f'seed_{seed}'; sd.mkdir(parents=True,exist_ok=True)
        payload={'protocol_id':PROTOCOL_ID,'method':method,'seed':seed,'checkpoint_sha256':cp_hash,'checkpoint_source':evidence[(method,seed)],
                 'dataset3_manifest_sha256':json.loads((HERE/'PRIMARY_ENDPOINT_LOCK.json').read_text())['dataset3_lock_manifest_sha256'],
                 'dataset3_external':met,'safeguards':{'training_performed':False,'adaptation_performed':False,'checkpoint_selection_performed':False,'tuning_performed':False,'test_time_adaptation':False}}
        (sd/'dataset3_external_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        pd.DataFrame(preds).to_csv(sd/'dataset3_external_predictions.csv',index=False)
        print(f"  {method:22s} Macro-F1={met['macro_f1']:.4f} BG={met['per_class']['BG']['f1']:.4f}")
        del model,ck; 
        if device.type=='cuda': torch.cuda.empty_cache()
        write_state(run_root,seed,method); aggregate(run_root); make_zip(work,run_root,False)
      new_seeds+=1
      print('Seed complete. Download REPORTS_LATEST before next execution.')
    state=write_state(run_root); aggregate(run_root)
    all_done=len(state['completed'])==45
    latest=make_zip(work,run_root,False)
    print('\nREPORT:',latest)
    if all_done:
      final=make_zip(work,run_root,True); print('FINAL REPORT:',final)
      print('\nFINAL EXTERNAL EVALUATION COMPLETE. Protocol is now permanently closed to tuning.')

if __name__=='__main__': main()
