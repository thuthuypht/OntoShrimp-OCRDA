#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batched/resumable frozen evaluation runner for OCRDA-v7.4.4."""
import argparse, gc, hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile
from datetime import datetime, timezone
from pathlib import Path
PKG=Path(__file__).resolve().parent
DEFAULT_INPUT=Path('/kaggle/input'); DEFAULT_WORK=Path('/kaggle/working')
RUN_NAME='OCRDA_v7_4_4_FROZEN_EVAL_RUN'
VERSION='OCRDA-v7.4.4-frozen-baselines-ablations-v1'
SEEDS=[42,123,2026,3407,7777]
BATCHES={
 'A':['source_only','coral','mmd','ccda_no_ontology'],
 'B':['full_ocrda','relation_blind','shuffled_ontology','ocrda_no_reliability','ocrda_no_topology'],
}
EPOCHS=30; WARMUP=8; ADAPT_EPOCHS=22; BATCH_SIZE=8; MAX_STEPS=72; WORKERS=2

def banner(x): print('\n'+'='*104+'\n'+x+'\n'+'='*104,flush=True)
def utc_now(): return datetime.now(timezone.utc).isoformat()
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def verify_frozen_package():
    mf=json.loads((PKG/'FROZEN_MANIFEST.json').read_text(encoding='utf-8'))
    if mf.get('package_id')!=VERSION: raise RuntimeError('Wrong package version: '+str(mf.get('package_id')))
    checks=[(PKG/'core'/'train_ocrda_v7_4_4.py',mf['full_trainer_sha256']),(PKG/'core'/'train_ocrda_v7_4_4_ablation.py',mf['ablation_trainer_sha256']),(PKG/'core'/'train_baselines_frozen_eval.py',mf['baseline_trainer_sha256']),(PKG/'core'/'ShrimpOntology.owl',mf['ontology_sha256'])]
    for p,exp in checks:
        got=sha256(p)
        if got!=exp: raise RuntimeError(f'FROZEN HASH MISMATCH: {p.name}\nexpected={exp}\ngot={got}')
    for n,exp in mf['protocol_sha256'].items():
        p=PKG/'protocol_dev'/n
        if sha256(p)!=exp: raise RuntimeError('Protocol hash mismatch: '+n)
    print('FROZEN VERSION CHECK: PASS — exact Full OCRDA-v7.4.4 trainer/config locked.',flush=True)
    return mf

def assert_dataset3_not_attached(input_root):
    bad=[]
    tokens=['dataset3','dataset_3','dataset-3','vietnam_v1','vietnam-v1']
    for p in input_root.iterdir() if input_root.exists() else []:
        n=p.name.lower()
        if any(t in n for t in tokens): bad.append(str(p))
    # also detect characteristic locked-test metadata names without scanning every image
    for pat in ['metadata_balanced_external_test.csv','dataset3_clean_keep_list.csv']:
        bad += [str(p) for p in input_root.rglob(pat)] if input_root.exists() else []
    if bad:
        raise RuntimeError('Dataset 3 appears attached. REMOVE it before formal baseline/ablation runs:\n'+'\n'.join(sorted(set(bad))[:20]))
    print('Dataset-3 guard: PASS (not attached/detected).',flush=True)

def ensure_deps():
    req={'torch':'torch','torchvision':'torchvision','pandas':'pandas','numpy':'numpy','PIL':'Pillow','rdflib':'rdflib','sklearn':'scikit-learn'}; miss=[]
    for mod,pkg in req.items():
        try: __import__(mod)
        except Exception: miss.append(pkg)
    if miss: subprocess.check_call([sys.executable,'-m','pip','install','-q',*miss])

def pretrained_ok(require_gpu=True):
    import torch
    from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
    if require_gpu and not torch.cuda.is_available(): raise RuntimeError('GPU not available. Kaggle -> Settings -> Accelerator -> GPU.')
    if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0),flush=True)
    try:
        m=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1); del m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print('ImageNet ConvNeXt-Tiny pretrained weights: OK',flush=True); return True
    except Exception as e:
        print('WARNING pretrained unavailable:',repr(e),flush=True); return False

def merge_tree(src,dst):
    if not src.exists(): return
    for p in src.rglob('*'):
        rel=p.relative_to(src); q=dst/rel
        if p.is_dir(): q.mkdir(parents=True,exist_ok=True)
        elif p.is_file(): q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)

def compatible_resume(path):
    n=path.name.lower(); return 'ocrda_v7_4_4_frozen' in n and 'resume' in n and path.suffix.lower()=='.zip'

def restore_resumes(input_root,run_root,explicit=None):
    cand=[]
    for x in explicit or []:
        p=Path(x)
        if p.exists(): cand.append(p)
    if input_root.exists(): cand += [p for p in input_root.rglob('*.zip') if compatible_resume(p)]
    uniq=[]; seen=set()
    for p in cand:
        s=str(p.resolve())
        if s not in seen: seen.add(s); uniq.append(p)
    uniq.sort(key=lambda p:(p.stat().st_size,str(p)))
    if not uniq: print('Resume restore: none found.',flush=True); return []
    restored=[]; tmp=Path(tempfile.mkdtemp(prefix='ocrda744frozen_restore_'))
    try:
        for i,zp in enumerate(uniq):
            td=tmp/f'z{i}'; td.mkdir(); print('Resume restore:',zp,flush=True)
            try:
                with zipfile.ZipFile(zp) as z: z.extractall(td)
            except Exception as e: print('  skip invalid ZIP',e,flush=True); continue
            roots=[p for p in td.rglob(RUN_NAME) if p.is_dir()]; src=sorted(roots,key=lambda p:len(p.parts))[0] if roots else td
            merge_tree(src,run_root); restored.append(str(zp))
    finally: shutil.rmtree(tmp,ignore_errors=True)
    return restored

def copy_protocol(run_root):
    proto=run_root/'protocol_dev'; proto.mkdir(parents=True,exist_ok=True)
    for p in (PKG/'protocol_dev').iterdir():
        if p.is_file() and not (proto/p.name).exists(): shutil.copy2(p,proto/p.name)
    return proto

def validate_root(root,rels):
    checks=[Path(str(x).replace('\\','/')) for x in rels[:min(10,len(rels))]]
    return sum((root/r).exists() for r in checks)

def infer_root(csv_path,dataset_id,input_root):
    import pandas as pd
    df=pd.read_csv(csv_path)
    if 'dataset_id' in df:
        sub=df[df.dataset_id.astype(str).str.upper()==dataset_id.upper()]
        if len(sub): df=sub
    rels=df.original_relpath.dropna().astype(str).tolist(); first=Path(rels[0].replace('\\','/')); filename=first.name; cand=[]
    for hit in input_root.rglob(filename):
        if PKG in hit.parents: continue
        root=hit
        for _ in first.parts: root=root.parent
        sc=validate_root(root,rels)
        if sc: cand.append((sc,root))
    if not cand:
        for d in input_root.rglob('*'):
            if d.is_dir():
                sc=validate_root(d,rels)
                if sc: cand.append((sc,d))
    if not cand: raise FileNotFoundError(f'Cannot infer {dataset_id} root. Attach OntoShrimp image dataset as Kaggle Input.')
    cand.sort(key=lambda x:(-x[0],len(str(x[1])))); print(dataset_id,'root =',cand[0][1],'validation =',cand[0][0],flush=True); return cand[0][1]

def metric_path(results,method,seed):
    sd=results/method/f'seed_{seed}'
    for n in ['metrics_v7_4_4.json','metrics.json']:
        p=sd/n
        if p.exists(): return p
    return None

def done(results,method,seed): return metric_path(results,method,seed) is not None

def run(cmd): print('\nCMD:\n',' '.join(map(str,cmd)),flush=True); subprocess.check_call([str(x) for x in cmd])

def halfify(obj):
    import torch
    if torch.is_tensor(obj):
        x=obj.detach().cpu(); return x.half() if torch.is_floating_point(x) else x
    if isinstance(obj,dict): return {k:halfify(v) for k,v in obj.items()}
    if isinstance(obj,list): return [halfify(v) for v in obj]
    if isinstance(obj,tuple): return tuple(halfify(v) for v in obj)
    return obj

def compact_model(results,method,seed):
    import torch
    sd=results/method/f'seed_{seed}'; src=sd/'best_model.pt'; dst=sd/'compact_model_fp16.pt'
    if dst.exists(): return dst
    if not src.exists(): print('WARNING model missing:',src,flush=True); return None
    ck=torch.load(src,map_location='cpu')
    if method in ['source_only','coral','mmd','ccda_no_ontology']:
        compact={'format':'visual-baseline-fp16-v1','method':method,'seed':seed}
        if isinstance(ck,dict) and 'state_dict' in ck: compact['state_dict']=halfify(ck['state_dict'])
        if isinstance(ck,dict) and 'result' in ck: compact['result']=ck['result']
    else:
        keep=['teacher_visual_state_dict','teacher_rgcn_state_dict','source_prototypes','prototype_memory_values','prototype_memory_initialized','prototype_memory_update_counts','ontology_nodes','ontology_relations','recognition_anchor_ids','args','result']
        compact={'format':'ocrda-v744-frozen-eval-teacher-fp16-v1','method':method,'seed':seed}
        if isinstance(ck,dict):
            for k in keep:
                if k in ck: compact[k]=halfify(ck[k])
    torch.save(compact,dst); del ck,compact; gc.collect(); src.unlink(missing_ok=True)
    for p in sd.glob('*.npz'): p.unlink(missing_ok=True)
    print('Compacted model:',dst,f'{dst.stat().st_size/1024/1024:.1f} MB',flush=True); return dst

def summarize(run_root):
    out=run_root/'summary'; out.mkdir(parents=True,exist_ok=True)
    run([sys.executable,PKG/'summarize_suite.py','--results-dir',run_root/'results','--out-dir',out,'--manifest',PKG/'FROZEN_MANIFEST.json'])

def completed(results):
    return [{'method':m,'seed':s} for m in BATCHES['A']+BATCHES['B'] for s in SEEDS if done(results,m,s)]

def write_state(run_root,batch,current_seed=None,current_method=None,restored=None):
    st=run_root/'state'; st.mkdir(parents=True,exist_ok=True)
    obj={'version':VERSION,'updated_utc':utc_now(),'batch':batch,'current_seed':current_seed,'current_method':current_method,'completed':completed(run_root/'results'),'restored_from':restored or [],'dataset3_used':False,'full_ocrda_frozen':True}
    (st/'resume_manifest.json').write_text(json.dumps(obj,indent=2),encoding='utf-8'); return obj

def add_tree(z,base,arcroot,pred=lambda p:True):
    if not base.exists(): return
    for p in base.rglob('*'):
        if p.is_file() and pred(p): z.write(p,arcroot/p.relative_to(base))

def make_report_zip(work,run_root,batch):
    zp=work/f'OCRDA_v7_4_4_FROZEN_BATCH_{batch}_REPORTS_LATEST.zip'; zp.unlink(missing_ok=True)
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for sub in ['results','summary','state']:
            add_tree(z,run_root/sub,Path(RUN_NAME)/sub,lambda p:p.suffix.lower() not in {'.pt','.pth','.npz'})
        z.write(PKG/'FROZEN_MANIFEST.json',Path(RUN_NAME)/'FROZEN_MANIFEST.json')
    return zp

def make_resume_zip(work,run_root,batch):
    zp=work/f'OCRDA_v7_4_4_FROZEN_BATCH_{batch}_RESUME_LATEST.zip'; zp.unlink(missing_ok=True)
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for sub in ['results','summary','state']:
            add_tree(z,run_root/sub,Path(RUN_NAME)/sub,lambda p:p.suffix.lower() not in {'.pt','.pth','.npz'})
    return zp

def make_model_zip(work,results,method,seed):
    sd=results/method/f'seed_{seed}'; model=sd/'compact_model_fp16.pt'; metric=metric_path(results,method,seed)
    if not model.exists(): return None
    safe=method.upper(); zp=work/f'OCRDA_v7_4_4_FROZEN_MODEL_{safe}_SEED_{seed}.zip'; zp.unlink(missing_ok=True)
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        z.write(model,Path(method)/f'seed_{seed}'/model.name)
        if metric: z.write(metric,Path(method)/f'seed_{seed}'/metric.name)
    model.unlink(missing_ok=True); print('MODEL ZIP:',zp,f'{zp.stat().st_size/1024/1024:.1f} MB',flush=True); return zp

def run_baseline(method,seed,proto,results,caches,sdbd,tsbd,no_pre):
    cmd=[sys.executable,PKG/'core'/'train_baselines_frozen_eval.py','--method',method,'--split-dir',proto,'--sdbd-root',sdbd,'--tsbd-root',tsbd,'--output-dir',results/method,'--source-cache-dir',caches/'visual','--seeds',seed,'--source-epochs',WARMUP,'--adapt-epochs',ADAPT_EPOCHS,'--max-steps-per-epoch',MAX_STEPS,'--batch-size',BATCH_SIZE,'--workers',WORKERS,'--target-dev-file','target_dev_exposed.csv',*no_pre]
    run(cmd)

def run_ocrda(method,seed,proto,results,caches,sdbd,tsbd,no_pre,use_cache=False):
    trainer=PKG/'core'/('train_ocrda_v7_4_4.py' if method=='full_ocrda' else 'train_ocrda_v7_4_4_ablation.py')
    cmd=[sys.executable,trainer,'--ontology',PKG/'core'/'ShrimpOntology.owl','--split-dir',proto,'--sdbd-root',sdbd,'--tsbd-root',tsbd,'--output-dir',results/method,'--method',method,'--source-cache-dir',caches/'ocrda','--seeds',seed,'--epochs',EPOCHS,'--warmup-epochs',WARMUP,'--max-adapt-steps-per-epoch',MAX_STEPS,'--batch-size',BATCH_SIZE,'--workers',WORKERS,'--target-dev-file','target_dev_exposed.csv','--report-target-dev',*no_pre]
    if use_cache: cmd.append('--use-source-cache')
    run(cmd)

def clean_seed_caches(caches,seed):
    pats=[caches/'ocrda'/f'ocrda_sourcewarm_seed_{seed}.pt']
    # visual cache name is discovered by seed to avoid depending on old filename convention
    if (caches/'visual').exists(): pats += list((caches/'visual').glob(f'*{seed}*.pt'))
    for p in pats:
        if p.exists():
            sz=p.stat().st_size/1024/1024; p.unlink(); print(f'Deleted transient cache {p.name} ({sz:.1f} MB)',flush=True)

def batch_complete(results,batch): return all(done(results,m,s) for m in BATCHES[batch] for s in SEEDS)
def suite_complete(results): return all(done(results,m,s) for m in BATCHES['A']+BATCHES['B'] for s in SEEDS)

def finalize(work,latest,name):
    if latest and latest.exists():
        dst=work/name; shutil.copy2(latest,dst); print('FINAL:',dst,flush=True); return dst

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--batch',choices=['A','B'],required=True); ap.add_argument('--input-root',default=str(DEFAULT_INPUT)); ap.add_argument('--work-root',default=str(DEFAULT_WORK)); ap.add_argument('--resume-zip',action='append',default=[]); ap.add_argument('--no-auto-restore',action='store_true'); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--max-new-seeds',type=int,default=1)
    a=ap.parse_args(); input_root=Path(a.input_root); work=Path(a.work_root); run_root=work/RUN_NAME; results=run_root/'results'; caches=run_root/'source_caches'; results.mkdir(parents=True,exist_ok=True); caches.mkdir(parents=True,exist_ok=True)
    mf=verify_frozen_package(); banner(f'{VERSION} — BATCH {a.batch}: '+('BASELINES' if a.batch=='A' else 'FROZEN FULL + OCRDA ABLATIONS'))
    assert_dataset3_not_attached(input_root)
    restored=[] if a.no_auto_restore else restore_resumes(input_root,run_root,a.resume_zip)
    proto=copy_protocol(run_root); write_state(run_root,a.batch,restored=restored)
    if a.dry_run:
        print('DRY RUN PASS. Completed:',completed(results)); return
    ensure_deps(); use_pre=pretrained_ok(); no_pre=[] if use_pre else ['--no-pretrained']
    sdbd=infer_root(proto/'source_train_final.csv','SDBD',input_root); tsbd=infer_root(proto/'target_adapt_final.csv','TSBD',input_root)
    processed=0; report_zip=resume_zip=None
    methods=BATCHES[a.batch]
    for seed in SEEDS:
        pending=[m for m in methods if not done(results,m,seed)]
        if not pending: print('SKIP completed seed',seed,'for batch',a.batch,flush=True); continue
        if processed>=max(1,a.max_new_seeds): print('SAFETY STOP: one new seed-bundle completed. Download artifacts then rerun.',flush=True); break
        banner(f'BATCH {a.batch} — SEED {seed} — pending={pending}')
        # Batch B fairness rule: all ablations MUST reuse the exact source-warm cache produced by frozen Full OCRDA.
        if a.batch=='B':
            cache=caches/'ocrda'/f'ocrda_sourcewarm_seed_{seed}.pt'
            ab_pending=[m for m in methods if m!='full_ocrda' and not done(results,m,seed)]
            if ab_pending and not cache.exists() and done(results,'full_ocrda',seed):
                print('Exact source cache missing after resume. Deterministically re-running frozen Full OCRDA for this seed to rebuild it before ablations.',flush=True)
                run_ocrda('full_ocrda',seed,proto,results,caches,sdbd,tsbd,no_pre,use_cache=False)
                compact_model(results,'full_ocrda',seed); summarize(run_root); write_state(run_root,a.batch,seed,'full_ocrda_cache_rebuilt',restored)
                make_report_zip(work,run_root,a.batch); make_resume_zip(work,run_root,a.batch); make_model_zip(work,results,'full_ocrda',seed)
        for method in methods:
            if done(results,method,seed): print('SKIP completed',method,seed,flush=True); continue
            write_state(run_root,a.batch,seed,method,restored)
            if a.batch=='A': run_baseline(method,seed,proto,results,caches,sdbd,tsbd,no_pre)
            else:
                use_cache=(method!='full_ocrda')
                if use_cache and not (caches/'ocrda'/f'ocrda_sourcewarm_seed_{seed}.pt').exists():
                    raise RuntimeError('Fairness guard: ablation cannot run without frozen Full OCRDA source-warm cache for same seed.')
                run_ocrda(method,seed,proto,results,caches,sdbd,tsbd,no_pre,use_cache=use_cache)
            compact_model(results,method,seed); summarize(run_root); write_state(run_root,a.batch,seed,method+'_COMPLETE',restored)
            report_zip=make_report_zip(work,run_root,a.batch); resume_zip=make_resume_zip(work,run_root,a.batch); model_zip=make_model_zip(work,results,method,seed)
            print('METHOD COMPLETE:',method,seed,flush=True); print('REPORT:',report_zip,flush=True); print('RESUME:',resume_zip,flush=True); print('MODEL:',model_zip,flush=True)
        clean_seed_caches(caches,seed); processed+=1
        print('\nSEED-BUNDLE COMPLETE:',seed,'BATCH',a.batch,flush=True)
        print('DOWNLOAD REPORT + RESUME and all MODEL ZIPs for this seed before rerunning.',flush=True)
    summarize(run_root); write_state(run_root,a.batch,None,'BATCH_COMPLETE' if batch_complete(results,a.batch) else 'PAUSED',restored)
    report_zip=make_report_zip(work,run_root,a.batch); resume_zip=make_resume_zip(work,run_root,a.batch)
    if batch_complete(results,a.batch):
        finalize(work,report_zip,f'OCRDA_v7_4_4_FROZEN_BATCH_{a.batch}_REPORTS_FINAL.zip')
        finalize(work,resume_zip,f'OCRDA_v7_4_4_FROZEN_BATCH_{a.batch}_RESUME_FINAL.zip')
    if suite_complete(results):
        finalize(work,report_zip,'OCRDA_v7_4_4_FROZEN_EVAL_REPORTS_FINAL.zip')
    print('\nDataset 3 was NOT used. Full OCRDA-v7.4.4 remains frozen.',flush=True)
if __name__=='__main__': main()
