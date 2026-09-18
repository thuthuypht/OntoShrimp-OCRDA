#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
SEEDS=[42,123,2026,3407,7777]
METHOD_ORDER=['source_only','coral','mmd','ccda_no_ontology','relation_blind','shuffled_ontology','ocrda_no_reliability','ocrda_no_topology','full_ocrda']

def load_metric(seed_dir):
    for name in ['metrics_v7_4_4.json','metrics.json']:
        p=seed_dir/name
        if p.exists(): return json.loads(p.read_text(encoding='utf-8'))
    return None

def tcrit95(n):
    table={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}
    return table.get(n,1.96)

def summ(vals):
    x=np.asarray([v for v in vals if v is not None and np.isfinite(v)],float)
    if len(x)==0:return {'n':0,'mean':None,'std':None,'ci95_t':None}
    m=float(x.mean()); sd=float(x.std(ddof=1)) if len(x)>1 else 0.0
    ci=None if len(x)<2 else [m-tcrit95(len(x))*sd/math.sqrt(len(x)),m+tcrit95(len(x))*sd/math.sqrt(len(x))]
    return {'n':len(x),'mean':m,'std':sd,'ci95_t':ci}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-dir',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--manifest',default=None)
    a=ap.parse_args(); root=Path(a.results_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; raw={}
    for method in METHOD_ORDER:
        raw[method]={}
        for seed in SEEDS:
            obj=load_metric(root/method/f'seed_{seed}')
            if not obj: continue
            raw[method][str(seed)]=obj
            sd=obj.get('source_test') or {}; td=obj.get('target_dev') or {}
            row={'method':method,'seed':seed,'selected_epoch':obj.get('selected_epoch'),
                 'source_macro_f1':sd.get('macro_f1'),'source_accuracy':sd.get('accuracy'),
                 'target_dev_macro_f1':td.get('macro_f1'),'target_dev_accuracy':td.get('accuracy'),
                 'target_dev_balanced_accuracy':td.get('balanced_accuracy'),'target_dev_mcc':td.get('mcc')}
            for c in ['BG','Healthy','WSSV']:
                pc=(td.get('per_class') or {}).get(c,{})
                row[f'target_dev_precision_{c}']=pc.get('precision'); row[f'target_dev_recall_{c}']=pc.get('recall'); row[f'target_dev_f1_{c}']=pc.get('f1')
            rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(out/'per_seed_metrics.csv',index=False,encoding='utf-8-sig')
    agg=[]
    cols=['source_macro_f1','target_dev_macro_f1','target_dev_accuracy','target_dev_balanced_accuracy','target_dev_mcc','target_dev_f1_BG','target_dev_f1_Healthy','target_dev_f1_WSSV']
    for method in METHOD_ORDER:
        x=df[df.method==method] if len(df) else pd.DataFrame()
        if len(x)==0: continue
        rec={'method':method,'n_seeds':len(x)}
        for col in cols:
            z=summ(x[col].tolist()); rec[col+'_mean']=z['mean']; rec[col+'_std']=z['std']; rec[col+'_ci95_low']=z['ci95_t'][0] if z['ci95_t'] else None; rec[col+'_ci95_high']=z['ci95_t'][1] if z['ci95_t'] else None
        agg.append(rec)
    adf=pd.DataFrame(agg); adf.to_csv(out/'aggregate_mean_std_ci.csv',index=False,encoding='utf-8-sig')
    pair=[]
    if len(df) and 'full_ocrda' in set(df.method):
        full=df[df.method=='full_ocrda'][['seed','target_dev_macro_f1']].rename(columns={'target_dev_macro_f1':'full'})
        for method in METHOD_ORDER:
            if method=='full_ocrda': continue
            m=df[df.method==method][['seed','target_dev_macro_f1']].rename(columns={'target_dev_macro_f1':'other'})
            j=full.merge(m,on='seed').dropna()
            if len(j):
                delta=j['full']-j['other']; z=summ(delta.tolist())
                pair.append({'comparison':f'full_ocrda - {method}','n':len(j),'mean_delta_f1':z['mean'],'std_delta':z['std'],'wins_full':int((delta>0).sum()),'ties':int((delta==0).sum()),'losses_full':int((delta<0).sum())})
    pd.DataFrame(pair).to_csv(out/'paired_differences_vs_full.csv',index=False,encoding='utf-8-sig')
    lines=['# OCRDA-v7.4.4 FROZEN Formal Baseline/Ablation Summary','',
           '> Full OCRDA-v7.4.4 is frozen. target-dev is diagnostic/development only. Dataset 3 is not used here.','',
           '| Method | n | Target-dev Macro-F1 mean±SD | BG F1 | Healthy F1 | WSSV F1 | Source F1 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        f=lambda k: 'NA' if r.get(k) is None else f"{r[k]:.3f}"
        lines.append(f"| {r['method']} | {r['n_seeds']} | {f('target_dev_macro_f1_mean')} ± {f('target_dev_macro_f1_std')} | {f('target_dev_f1_BG_mean')} | {f('target_dev_f1_Healthy_mean')} | {f('target_dev_f1_WSSV_mean')} | {f('source_macro_f1_mean')} |")
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'all_metrics.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
    status={'dataset3_used':False,'full_model_frozen':True,'complete_methods':{m:sorted([int(s) for s in raw[m]]) for m in METHOD_ORDER}}
    if a.manifest and Path(a.manifest).exists(): status['frozen_manifest']=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    (out/'FORMAL_EVAL_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()
