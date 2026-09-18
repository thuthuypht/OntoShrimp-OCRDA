#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the 45 frozen OCRDA-v7.4.4 model checkpoints.

Purpose: recover lost checkpoint weights only. The model/protocol is already frozen.
Dataset 3 is explicitly forbidden here.

One execution regenerates at most one seed bundle (9 methods), then stops.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile
from datetime import datetime, timezone
from pathlib import Path

PKG=Path(__file__).resolve().parent
DEFAULT_INPUT=Path('/kaggle/input')
DEFAULT_WORK=Path('/kaggle/working')
RUN_NAME='OCRDA_v7_4_4_REGENERATE_FROZEN_45_MODELS_RUN'
PACKAGE_ID='OCRDA-v7.4.4-regenerate-frozen-45-models-v1'
FROZEN_ID='OCRDA-v7.4.4-frozen-baselines-ablations-v1'
SEEDS=[42,123,2026,3407,7777]
METHODS_EXTERNAL=['source_only','coral','mmd','ccda_no_ontology','relation_blind','shuffled_ontology','ocrda_no_reliability','ocrda_no_topology','full_ocrda']
# Full must precede the four OCRDA ablations so they reuse its exact source-warm cache.
METHODS_RUN=['source_only','coral','mmd','ccda_no_ontology','full_ocrda','relation_blind','shuffled_ontology','ocrda_no_reliability','ocrda_no_topology']
BASELINES={'source_only','coral','mmd','ccda_no_ontology'}
ABLATIONS={'relation_blind','shuffled_ontology','ocrda_no_reliability','ocrda_no_topology'}
EPOCHS=30; WARMUP=8; ADAPT_EPOCHS=22; BATCH_SIZE=8; MAX_STEPS=72; WORKERS=2
MODEL_RE=re.compile(r'^OCRDA_v7_4_4_FROZEN_MODEL_([A-Z0-9_]+)_SEED_(42|123|2026|3407|7777)\.zip$',re.I)
SEED_BUNDLE_RE=re.compile(r'^OCRDA_v7_4_4_REGENERATED_MODELS_SEED_(42|123|2026|3407|7777)\.zip$',re.I)


def now(): return datetime.now(timezone.utc).isoformat()
def banner(x): print('\n'+'='*108+'\n'+x+'\n'+'='*108,flush=True)
def sha256_file(p:Path,chunk=1<<20):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def verify_package():
    info=json.loads((PKG/'PACKAGE_INFO.json').read_text(encoding='utf-8'))
    if info.get('package_id')!=PACKAGE_ID: raise RuntimeError('Wrong regeneration package ID')
    mf=json.loads((PKG/'FROZEN_MANIFEST.json').read_text(encoding='utf-8'))
    if mf.get('package_id')!=FROZEN_ID: raise RuntimeError('Wrong frozen manifest ID')
    checks=[
      (PKG/'core'/'train_ocrda_v7_4_4.py',mf['full_trainer_sha256']),
      (PKG/'core'/'train_ocrda_v7_4_4_ablation.py',mf['ablation_trainer_sha256']),
      (PKG/'core'/'train_baselines_frozen_eval.py',mf['baseline_trainer_sha256']),
      (PKG/'core'/'ShrimpOntology.owl',mf['ontology_sha256']),
    ]
    for p,exp in checks:
        got=sha256_file(p)
        if got!=exp: raise RuntimeError(f'FROZEN HASH MISMATCH {p.name}: expected={exp} got={got}')
    for n,exp in mf['protocol_sha256'].items():
        p=PKG/'protocol_dev'/n
        got=sha256_file(p)
        if got!=exp: raise RuntimeError(f'PROTOCOL HASH MISMATCH {n}: expected={exp} got={got}')
    print('FROZEN INTEGRITY CHECK: PASS — exact v7.4.4 trainers/ontology/protocol.',flush=True)
    return mf

def assert_dataset3_not_attached(input_root:Path):
    bad=[]; tokens=['dataset3','dataset_3','dataset-3','vietnam_v1','vietnam-v1']
    for p in input_root.iterdir() if input_root.exists() else []:
        n=p.name.lower()
        if any(t in n for t in tokens): bad.append(str(p))
    for pat in ['metadata_balanced_external_test.csv','dataset3_clean_keep_list.csv','DATASET3_LOCK_MANIFEST.csv']:
        bad += [str(p) for p in input_root.rglob(pat)] if input_root.exists() else []
    if bad:
        raise RuntimeError('Dataset 3 is attached/detected. REMOVE it before checkpoint regeneration:\n'+'\n'.join(sorted(set(bad))[:30]))
    print('Dataset-3 guard: PASS — final external test is NOT attached.',flush=True)

def ensure_deps():
    req={'torch':'torch','torchvision':'torchvision','pandas':'pandas','numpy':'numpy','PIL':'Pillow','rdflib':'rdflib','sklearn':'scikit-learn'}
    miss=[]
    for mod,pkg in req.items():
        try: __import__(mod)
        except Exception: miss.append(pkg)
    if miss: subprocess.check_call([sys.executable,'-m','pip','install','-q',*miss])

def require_pretrained():
    import torch
    from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
    if not torch.cuda.is_available():
        raise RuntimeError('GPU not available. Kaggle -> Settings -> Accelerator -> GPU.')
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    try:
        m=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError('Frozen formal run used ImageNet-pretrained ConvNeXt-Tiny. Pretrained weights are unavailable. Turn Kaggle Internet ON or make the torchvision weights cache available; regeneration is aborted rather than switching to --no-pretrained. Original error: '+repr(e))
    print('ImageNet ConvNeXt-Tiny pretrained weights: PASS (required; no fallback).',flush=True)

def copy_protocol(run_root:Path):
    dst=run_root/'protocol_dev'; dst.mkdir(parents=True,exist_ok=True)
    for p in (PKG/'protocol_dev').iterdir():
        if p.is_file(): shutil.copy2(p,dst/p.name)
    return dst

def validate_root(root,rels):
    checks=[Path(str(x).replace('\\','/')) for x in rels[:min(12,len(rels))]]
    return sum((root/r).exists() for r in checks)

def infer_root(csv_path,dataset_id,input_root):
    import pandas as pd
    df=pd.read_csv(csv_path)
    if 'dataset_id' in df:
        sub=df[df.dataset_id.astype(str).str.upper()==dataset_id.upper()]
        if len(sub): df=sub
    rels=df.original_relpath.dropna().astype(str).tolist()
    first=Path(rels[0].replace('\\','/')); filename=first.name; cand=[]
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
    if not cand: raise FileNotFoundError(f'Cannot infer {dataset_id} root. Attach the same OntoShrimp dataset used in the frozen formal evaluation.')
    cand.sort(key=lambda x:(-x[0],len(str(x[1]))))
    print(dataset_id,'root =',cand[0][1],'validation =',cand[0][0],flush=True)
    return cand[0][1]

def metric_path(results,method,seed):
    sd=results/method/f'seed_{seed}'
    for n in ['metrics_v7_4_4.json','metrics.json']:
        p=sd/n
        if p.exists(): return p
    return None

def run(cmd):
    print('\nCMD:\n',' '.join(map(str,cmd)),flush=True)
    subprocess.check_call([str(x) for x in cmd])

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
    if not src.exists(): raise FileNotFoundError(f'Expected trainer checkpoint missing: {src}')
    ck=torch.load(src,map_location='cpu')
    if method in BASELINES:
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
    print('Compacted checkpoint:',dst,f'{dst.stat().st_size/1024/1024:.1f} MB',flush=True)
    return dst

def individual_model_zip_name(method,seed): return f'OCRDA_v7_4_4_FROZEN_MODEL_{method.upper()}_SEED_{seed}.zip'

def make_model_zip(work,results,method,seed):
    sd=results/method/f'seed_{seed}'; model=sd/'compact_model_fp16.pt'; metric=metric_path(results,method,seed)
    if not model.exists(): raise FileNotFoundError(model)
    zp=work/individual_model_zip_name(method,seed); zp.unlink(missing_ok=True)
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        z.write(model,Path(method)/f'seed_{seed}'/model.name)
        if metric: z.write(metric,Path(method)/f'seed_{seed}'/metric.name)
    cp_hash=sha256_file(model); zip_hash=sha256_file(zp); model.unlink(missing_ok=True)
    print('MODEL ZIP:',zp.name,f'{zp.stat().st_size/1024/1024:.1f} MB','checkpoint_sha256='+cp_hash,flush=True)
    return zp,cp_hash,zip_hash

def infer_method_seed_from_path(p:Path):
    x=str(p).replace('\\','/').lower(); seed=None; method=None
    mm=re.search(r'seed[_-](42|123|2026|3407|7777)',x)
    if mm: seed=int(mm.group(1))
    for m in METHODS_EXTERNAL:
        if f'/{m}/' in x or m in p.name.lower(): method=m; break
    return method,seed

def discover_artifacts(input_root:Path,work:Path):
    """Return {(method,seed): ('pt'|'zip', Path)}. Direct compact .pt is preferred."""
    idx={}
    for root in [input_root,work]:
        if not root.exists(): continue
        for p in root.rglob('*.pt'):
            if p.name!='compact_model_fp16.pt' and 'compact' not in p.name.lower(): continue
            m,sd=infer_method_seed_from_path(p)
            if m in METHODS_EXTERNAL and sd in SEEDS: idx[(m,sd)]=('pt',p)
    for root in [input_root,work]:
        if not root.exists(): continue
        for p in root.rglob('OCRDA_v7_4_4_FROZEN_MODEL_*_SEED_*.zip'):
            mo=MODEL_RE.match(p.name)
            if not mo: continue
            up=mo.group(1).upper(); sd=int(mo.group(2)); method=None
            for m in METHODS_EXTERNAL:
                if m.upper()==up: method=m; break
            if method is not None and (method,sd) not in idx: idx[(method,sd)]=('zip',p)
    return idx

def artifact_checkpoint_bytes(spec):
    typ,p=spec
    if typ=='pt': return p.read_bytes()
    with zipfile.ZipFile(p) as z:
        names=[n for n in z.namelist() if n.endswith('compact_model_fp16.pt') or n.endswith('best_model.pt')]
        if not names: raise RuntimeError(f'No compact checkpoint inside {p}')
        return z.read(names[0])

def artifact_checkpoint_sha256(spec):
    typ,p=spec
    if typ=='pt': return sha256_file(p)
    return hashlib.sha256(artifact_checkpoint_bytes(spec)).hexdigest()

def restore_metric_from_artifact(spec,results:Path,method:str,seed:int):
    sd=results/method/f'seed_{seed}'; sd.mkdir(parents=True,exist_ok=True)
    if metric_path(results,method,seed): return
    typ,p=spec
    if typ=='pt':
        for n in ['metrics_v7_4_4.json','metrics.json']:
            q=p.parent/n
            if q.exists(): shutil.copy2(q,sd/n); return
        return
    try:
        with zipfile.ZipFile(p) as z:
            names=[n for n in z.namelist() if n.endswith('/metrics_v7_4_4.json') or n.endswith('/metrics.json')]
            if names:
                n=names[0]; (sd/Path(n).name).write_bytes(z.read(n))
    except Exception as e: print('WARNING could not restore metric from',p,e,flush=True)

def restore_from_seed_bundles(input_root:Path,work:Path,results:Path):
    # Kaggle may auto-unpack the outer seed bundle; direct .pt files are then discovered.
    # Also support outer seed bundles that remain as ZIP files.
    restored=[]; exroot=work/'_seed_bundle_extract'; exroot.mkdir(parents=True,exist_ok=True)
    for root in [input_root,work]:
        if not root.exists(): continue
        for b in root.rglob('OCRDA_v7_4_4_REGENERATED_MODELS_SEED_*.zip'):
            if not SEED_BUNDLE_RE.match(b.name): continue
            td=exroot/b.stem
            if not td.exists() or not any(td.rglob('compact_model_fp16.pt')):
                td.mkdir(parents=True,exist_ok=True)
                try:
                    with zipfile.ZipFile(b) as z: z.extractall(td)
                    restored.append(str(b))
                except Exception as e: print('WARNING invalid seed bundle',b,e,flush=True)
    idx=discover_artifacts(input_root,work)
    for (m,sd),spec in idx.items(): restore_metric_from_artifact(spec,results,m,sd)
    return idx,restored

def merge_tree(src:Path,dst:Path):
    if not src.exists(): return
    for p in src.rglob('*'):
        q=dst/p.relative_to(src)
        if p.is_dir(): q.mkdir(parents=True,exist_ok=True)
        elif p.is_file(): q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)

def restore_light_resumes(input_root:Path,run_root:Path,work:Path):
    restored=[]; tmp=work/'_resume_extract'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True,exist_ok=True)
    # zipped resumes
    zs=[]
    for root in [input_root,work]:
        if root.exists():
            zs += [p for p in root.rglob('*.zip') if 'v7_4_4_regen_resume' in p.name.lower()]
    for i,zp in enumerate(sorted(set(zs),key=lambda p:(p.stat().st_size,str(p)))):
        td=tmp/f'z{i}'; td.mkdir(parents=True,exist_ok=True)
        try:
            with zipfile.ZipFile(zp) as z: z.extractall(td)
            roots=[p for p in td.rglob(RUN_NAME) if p.is_dir()]
            if roots: merge_tree(sorted(roots,key=lambda p:len(p.parts))[0],run_root); restored.append(str(zp))
        except Exception as e: print('WARNING resume restore failed',zp,e,flush=True)
    # Kaggle-unpacked resume datasets
    for mf in input_root.rglob('regen_resume_manifest.json') if input_root.exists() else []:
        base=mf.parent.parent
        if base.name==RUN_NAME:
            merge_tree(base,run_root); restored.append(str(base))
    return restored

def model_complete(idx,method,seed): return (method,seed) in idx

def seed_complete(idx,seed): return all((m,seed) in idx for m in METHODS_EXTERNAL)

def write_state(run_root:Path,idx,current_seed=None,current_method=None,restored=None):
    st=run_root/'state'; st.mkdir(parents=True,exist_ok=True)
    obj={
      'package_id':PACKAGE_ID,'frozen_model_version':'OCRDA-v7.4.4','updated_utc':now(),
      'current_seed':current_seed,'current_method':current_method,
      'completed_models':[{'method':m,'seed':s,'model_artifact':idx[(m,s)][1].name if (m,s) in idx else None} for s in SEEDS for m in METHODS_EXTERNAL if (m,s) in idx],
      'completed_seeds':[s for s in SEEDS if seed_complete(idx,s)],
      'restored_from':restored or [],'dataset3_used':False,'tuning_performed':False,'checkpoint_selection_changed':False,
      'regeneration_rule':'one regeneration attempt per missing method/seed; no metric-based rerun or cherry-pick'
    }
    (st/'regen_resume_manifest.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    return obj

def summarize(run_root):
    out=run_root/'summary'; out.mkdir(parents=True,exist_ok=True)
    run([sys.executable,PKG/'summarize_suite.py','--results-dir',run_root/'results','--out-dir',out,'--manifest',PKG/'FROZEN_MANIFEST.json'])

def add_tree(z,base,arcroot,pred=lambda p:True):
    if not base.exists(): return
    for p in base.rglob('*'):
        if p.is_file() and pred(p): z.write(p,arcroot/p.relative_to(base))

def make_light_resume(work,run_root):
    zp=work/'OCRDA_v7_4_4_REGEN_RESUME_LATEST.zip'; zp.unlink(missing_ok=True)
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for sub in ['results','summary','state']:
            add_tree(z,run_root/sub,Path(RUN_NAME)/sub,lambda p:p.suffix.lower() not in {'.pt','.pth','.npz','.zip'})
        z.write(PKG/'REGENERATION_MANIFEST.json',Path(RUN_NAME)/'REGENERATION_MANIFEST.json')
    return zp

def make_report_zip(work,run_root,final=False):
    name='OCRDA_v7_4_4_REGEN_REPORTS_FINAL.zip' if final else 'OCRDA_v7_4_4_REGEN_REPORTS_LATEST.zip'
    zp=work/name; zp.unlink(missing_ok=True)
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for sub in ['results','summary','state']:
            add_tree(z,run_root/sub,Path(RUN_NAME)/sub,lambda p:p.suffix.lower() not in {'.pt','.pth','.npz','.zip'})
        for n in ['REGENERATION_MANIFEST.json','FROZEN_MANIFEST.json','REQUIRED_FROZEN_MODELS.csv']:
            z.write(PKG/n,Path(RUN_NAME)/n)
    return zp

def collect_model_zips(input_root:Path,work:Path):
    # Extract any outer seed bundles first.
    dummy=work/RUN_NAME/'results'; dummy.mkdir(parents=True,exist_ok=True)
    restore_from_seed_bundles(input_root,work,dummy)
    return discover_artifacts(input_root,work)

def make_seed_bundle(work:Path,results:Path,seed:int,idx):
    missing=[m for m in METHODS_EXTERNAL if (m,seed) not in idx]
    if missing: raise RuntimeError(f'Cannot bundle seed {seed}; missing {missing}')
    zp=work/f'OCRDA_v7_4_4_REGENERATED_MODELS_SEED_{seed}.zip'; zp.unlink(missing_ok=True)
    manifest={'package_id':PACKAGE_ID,'seed':seed,'created_utc':now(),'dataset3_used':False,'format':'direct-compact-pt-v1','methods':[]}
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for m in METHODS_EXTERNAL:
            spec=idx[(m,seed)]; data=artifact_checkpoint_bytes(spec)
            arc=Path('models')/m/f'seed_{seed}'/'compact_model_fp16.pt'
            z.writestr(str(arc),data)
            met=metric_path(results,m,seed)
            cp_hash=hashlib.sha256(data).hexdigest()
            manifest['methods'].append({'method':m,'checkpoint_path':str(arc),'checkpoint_sha256':cp_hash,'metric_file':met.name if met else None})
            if met: z.write(met,arcname=str(Path('models')/m/f'seed_{seed}'/met.name))
        z.writestr('seed_manifest.json',json.dumps(manifest,indent=2))
    print('SEED MODEL BUNDLE:',zp,f'{zp.stat().st_size/1024/1024:.1f} MB',flush=True)
    print('When uploaded to Kaggle, this bundle exposes 9 direct compact_model_fp16.pt files, compatible with the external evaluator.',flush=True)
    return zp

def make_final_bundle(work:Path,input_root:Path):
    idx=collect_model_zips(input_root,work)
    missing=[(m,sd) for sd in SEEDS for m in METHODS_EXTERNAL if (m,sd) not in idx]
    if missing:
        print('Final 45-model bundle NOT created yet. Missing protected checkpoints:',len(missing),flush=True)
        for m,sd in missing[:50]: print(f'  - {m} seed={sd}',flush=True)
        print('Attach/downloaded prior OCRDA_v7_4_4_REGENERATED_MODELS_SEED_<seed>.zip bundles as Kaggle Inputs, then run --bundle-only.',flush=True)
        return None
    out=work/'OCRDA_v744_FROZEN_45_MODELS_BUNDLE.zip'; out.unlink(missing_ok=True)
    mani={'bundle_id':'OCRDA-v7.4.4-regenerated-45-frozen-models-v1','created_utc':now(),'dataset3_used':False,'frozen_protocol_id':FROZEN_ID,'format':'45-direct-compact-pt-v1','models':[]}
    with zipfile.ZipFile(out,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for sd in SEEDS:
            for m in METHODS_EXTERNAL:
                spec=idx[(m,sd)]; data=artifact_checkpoint_bytes(spec)
                arc=Path('models')/m/f'seed_{sd}'/'compact_model_fp16.pt'
                z.writestr(str(arc),data)
                mani['models'].append({'method':m,'seed':sd,'checkpoint_path':str(arc),'checkpoint_sha256':hashlib.sha256(data).hexdigest()})
        z.writestr('REGENERATION_45_MODEL_MANIFEST.json',json.dumps(mani,indent=2))
        z.write(PKG/'REQUIRED_FROZEN_MODELS.csv','REQUIRED_FROZEN_MODELS.csv')
        z.write(PKG/'FROZEN_MANIFEST.json','FROZEN_MANIFEST.json')
        z.write(PKG/'REGENERATION_MANIFEST.json','REGENERATION_MANIFEST.json')
    print('\nFINAL 45-MODEL BUNDLE CREATED:',out,flush=True)
    print(f'Size: {out.stat().st_size/1024/1024:.1f} MB',flush=True)
    print('Compatibility: upload this ZIP as one Kaggle Dataset. Kaggle unpacks it to models/<method>/seed_<seed>/compact_model_fp16.pt; the locked Dataset3 external evaluator discovers these direct .pt checkpoints without relying on model-ZIP filename parsing.',flush=True)
    return out

def run_baseline(method,seed,proto,results,caches,sdbd,tsbd):
    cmd=[sys.executable,PKG/'core'/'train_baselines_frozen_eval.py','--method',method,'--split-dir',proto,'--sdbd-root',sdbd,'--tsbd-root',tsbd,'--output-dir',results/method,'--source-cache-dir',caches/'visual','--seeds',seed,'--source-epochs',WARMUP,'--adapt-epochs',ADAPT_EPOCHS,'--max-steps-per-epoch',MAX_STEPS,'--batch-size',BATCH_SIZE,'--workers',WORKERS,'--target-dev-file','target_dev_exposed.csv']
    run(cmd)

def run_ocrda(method,seed,proto,results,caches,sdbd,tsbd,use_cache=False,cache_rebuild_only=False):
    trainer=PKG/'core'/('train_ocrda_v7_4_4.py' if method=='full_ocrda' else 'train_ocrda_v7_4_4_ablation.py')
    outdir=results/method
    if cache_rebuild_only: outdir=results/'_cache_rebuild_full_ocrda'
    cmd=[sys.executable,trainer,'--ontology',PKG/'core'/'ShrimpOntology.owl','--split-dir',proto,'--sdbd-root',sdbd,'--tsbd-root',tsbd,'--output-dir',outdir,'--method',method,'--source-cache-dir',caches/'ocrda','--seeds',seed,'--epochs',EPOCHS,'--warmup-epochs',WARMUP,'--max-adapt-steps-per-epoch',MAX_STEPS,'--batch-size',BATCH_SIZE,'--workers',WORKERS,'--target-dev-file','target_dev_exposed.csv','--report-target-dev']
    if use_cache: cmd.append('--use-source-cache')
    run(cmd)
    if cache_rebuild_only:
        shutil.rmtree(outdir,ignore_errors=True)

def clean_seed_caches(caches:Path,seed:int):
    pats=[caches/'ocrda'/f'ocrda_sourcewarm_seed_{seed}.pt']
    if (caches/'visual').exists(): pats += list((caches/'visual').glob(f'*{seed}*.pt'))
    for p in pats:
        if p.exists():
            print('Deleting transient source-warm cache:',p.name,f'{p.stat().st_size/1024/1024:.1f} MB',flush=True); p.unlink()

def regenerate_one_seed(seed:int,proto,results,caches,sdbd,tsbd,input_root,work,restored):
    idx=collect_model_zips(input_root,work)
    pending=[m for m in METHODS_RUN if (m,seed) not in idx]
    banner(f'REGENERATE FROZEN MODELS — SEED {seed} — pending={pending}')
    if not pending:
        return idx
    # If full checkpoint is already protected in a model ZIP but ablations remain and the transient cache vanished,
    # rebuild only the source-warm cache in a throwaway output. Never replace/select the already-regenerated full checkpoint.
    ocrda_cache=caches/'ocrda'/f'ocrda_sourcewarm_seed_{seed}.pt'
    if any(m in ABLATIONS for m in pending) and ('full_ocrda',seed) in idx and not ocrda_cache.exists():
        print('Full model already protected but common OCRDA source cache is absent. Rebuilding the deterministic source-warm cache in throwaway output; existing full checkpoint is NOT replaced.',flush=True)
        run_ocrda('full_ocrda',seed,proto,results,caches,sdbd,tsbd,use_cache=False,cache_rebuild_only=True)
        if not ocrda_cache.exists(): raise RuntimeError('Failed to rebuild common OCRDA source-warm cache')
    for method in METHODS_RUN:
        idx=collect_model_zips(input_root,work)
        if (method,seed) in idx:
            print('SKIP protected model:',method,seed,flush=True); restore_metric_from_artifact(idx[(method,seed)],results,method,seed); continue
        write_state(work/RUN_NAME,idx,seed,method,restored)
        if method in BASELINES:
            run_baseline(method,seed,proto,results,caches,sdbd,tsbd)
        else:
            use_cache=(method!='full_ocrda')
            if use_cache and not ocrda_cache.exists(): raise RuntimeError('Fairness guard: OCRDA ablation cannot run without same-seed Full OCRDA source-warm cache.')
            run_ocrda(method,seed,proto,results,caches,sdbd,tsbd,use_cache=use_cache)
        compact_model(results,method,seed)
        zp,cp_hash,zh=make_model_zip(work,results,method,seed)
        idx=collect_model_zips(input_root,work)
        restore_metric_from_artifact(('zip',zp),results,method,seed)
        summarize(work/RUN_NAME)
        write_state(work/RUN_NAME,idx,seed,method+'_COMPLETE',restored)
        rz=make_light_resume(work,work/RUN_NAME); rp=make_report_zip(work,work/RUN_NAME,False)
        print('METHOD COMPLETE:',method,seed,flush=True); print('LIGHT RESUME:',rz,flush=True); print('REPORT:',rp,flush=True)
    idx=collect_model_zips(input_root,work)
    if not seed_complete(idx,seed): raise RuntimeError('Seed finished training loop but 9 protected checkpoints are not all present.')
    seed_bundle=make_seed_bundle(work,results,seed,idx)
    # Keep only the compact 9-model outer seed bundle as the durable artifact.
    # Extract a working copy so subsequent executions in the same session can see direct .pt checkpoints.
    ex=work/'_seed_bundle_extract'/seed_bundle.stem; ex.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(seed_bundle) as zz: zz.extractall(ex)
    for m in METHODS_EXTERNAL:
        q=work/individual_model_zip_name(m,seed)
        if q.exists(): q.unlink()
    clean_seed_caches(caches,seed)
    summarize(work/RUN_NAME); idx=collect_model_zips(input_root,work); write_state(work/RUN_NAME,idx,seed,'SEED_COMPLETE',restored)
    rz=make_light_resume(work,work/RUN_NAME); rp=make_report_zip(work,work/RUN_NAME,False)
    print('\nSEED COMPLETE:',seed,flush=True)
    print('DOWNLOAD NOW:',seed_bundle,flush=True); print('DOWNLOAD LIGHT RESUME:',rz,flush=True); print('DOWNLOAD REPORT:',rp,flush=True)
    return idx

def inventory(input_root,work,run_root):
    idx,_=restore_from_seed_bundles(input_root,work,run_root/'results')
    idx=discover_artifacts(input_root,work)
    print('\nMODEL INVENTORY')
    print('-'*70)
    for s in SEEDS:
        n=sum((m,s) in idx for m in METHODS_EXTERNAL)
        print(f'seed {s}: {n}/9 protected checkpoints', 'COMPLETE' if n==9 else 'PENDING')
    print(f'Total protected checkpoints visible: {len(idx)}/45')
    nxt=next((s for s in SEEDS if not seed_complete(idx,s)),None)
    print('NEXT SEED:',nxt if nxt is not None else 'ALL COMPLETE')
    return idx,nxt

def main():
    ap=argparse.ArgumentParser(description='Regenerate lost frozen OCRDA-v7.4.4 checkpoint weights; NO Dataset3.')
    ap.add_argument('--input-root',default=str(DEFAULT_INPUT)); ap.add_argument('--work-root',default=str(DEFAULT_WORK))
    ap.add_argument('--max-new-seeds',type=int,default=1); ap.add_argument('--inventory-only',action='store_true'); ap.add_argument('--bundle-only',action='store_true'); ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(); input_root=Path(a.input_root); work=Path(a.work_root); work.mkdir(parents=True,exist_ok=True)
    run_root=work/RUN_NAME; results=run_root/'results'; caches=run_root/'source_caches'; results.mkdir(parents=True,exist_ok=True); caches.mkdir(parents=True,exist_ok=True)
    mf=verify_package(); assert_dataset3_not_attached(input_root)
    restored=restore_light_resumes(input_root,run_root,work)
    idx,seed_bundle_restored=restore_from_seed_bundles(input_root,work,results); restored += seed_bundle_restored
    copy_protocol(run_root); idx,nxt=inventory(input_root,work,run_root); write_state(run_root,idx,restored=restored)
    if a.dry_run:
        print('DRY RUN PASS. No training performed.'); return
    if a.bundle_only:
        make_final_bundle(work,input_root); return
    if a.inventory_only:
        print('INVENTORY ONLY: no GPU training performed.'); return
    if nxt is None:
        print('All 45 protected model ZIPs are visible; no regeneration needed.'); make_final_bundle(work,input_root); return
    ensure_deps(); require_pretrained()
    proto=copy_protocol(run_root)
    sdbd=infer_root(proto/'source_train_final.csv','SDBD',input_root)
    tsbd=infer_root(proto/'target_adapt_final.csv','TSBD',input_root)
    processed=0
    for seed in SEEDS:
        idx=collect_model_zips(input_root,work)
        if seed_complete(idx,seed): print('SKIP complete seed',seed,flush=True); continue
        if processed>=max(1,a.max_new_seeds): break
        regenerate_one_seed(seed,proto,results,caches,sdbd,tsbd,input_root,work,restored); processed+=1
        break
    idx=collect_model_zips(input_root,work); summarize(run_root); state=write_state(run_root,idx,restored=restored)
    all_done=len(idx)>=45 and all(seed_complete(idx,s) for s in SEEDS)
    make_light_resume(work,run_root); make_report_zip(work,run_root,all_done)
    if all_done:
        make_final_bundle(work,input_root)
        print('\nALL 45 FROZEN CHECKPOINTS REGENERATED. DO NOT RETRAIN OR SELECT BY METRICS.',flush=True)
        print('Next step: upload OCRDA_v744_FROZEN_45_MODELS_BUNDLE.zip as a Kaggle Dataset, then run the locked Dataset3 external evaluator.',flush=True)
    else:
        print('\nSAFETY STOP: exactly one seed bundle was regenerated in this execution.',flush=True)
        print('Download the SEED MODELS bundle + LIGHT RESUME + REPORT before closing Kaggle.',flush=True)

if __name__=='__main__': main()
