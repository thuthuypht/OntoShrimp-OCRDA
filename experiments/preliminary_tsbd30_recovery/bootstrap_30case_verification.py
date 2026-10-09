#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

REQ=['image_id','true_label','yolo_only_pred','yolo_ontology_pred']
HIST={'onto_mean':0.556,'yolo_mean':0.564,'diff':-0.008,'p':0.53}

def validate(df):
    miss=[c for c in REQ if c not in df.columns]
    if miss: raise ValueError(f'Missing required columns: {miss}')
    if len(df)!=30: raise ValueError(f'Expected exactly 30 cases, found {len(df)}')
    if df[REQ].isna().any().any(): raise ValueError('Blank/NA values found in required columns')
    for c in REQ:
        if df[c].astype(str).str.strip().eq('').any(): raise ValueError(f'Blank values in {c}')
    if df['image_id'].duplicated().any(): raise ValueError('Duplicate image_id values found')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--outdir',default=None)
    a=ap.parse_args()
    inp=Path(a.input); out=Path(a.outdir) if a.outdir else inp.parent; out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(inp); validate(df)
    for c in ['true_label','yolo_only_pred','yolo_ontology_pred']:
        if not pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].astype(str).str.strip()
    labels=sorted(set(df.true_label)|set(df.yolo_only_pred)|set(df.yolo_ontology_pred),key=str)
    y=df.true_label.to_numpy(); yy=df.yolo_only_pred.to_numpy(); yo=df.yolo_ontology_pred.to_numpy()
    mf=lambda t,p:f1_score(t,p,labels=labels,average='macro',zero_division=0)
    point_y,point_o=mf(y,yy),mf(y,yo); acc_y,acc_o=accuracy_score(y,yy),accuracy_score(y,yo)
    rng=np.random.default_rng(a.seed);B=a.bootstrap;n=len(df)
    by=np.empty(B);bo=np.empty(B);d=np.empty(B)
    for i in range(B):
        idx=rng.integers(0,n,size=n); by[i]=mf(y[idx],yy[idx]); bo[i]=mf(y[idx],yo[idx]); d[i]=bo[i]-by[i]
    lo,hi=np.percentile(d,[2.5,97.5]); p=min(1.0,2*min(np.mean(d<=0),np.mean(d>=0)))
    s={'n_cases':n,'labels':'|'.join(map(str,labels)),'bootstrap_resamples':B,'random_seed':a.seed,
       'yolo_only_point_macro_f1':point_y,'yolo_ontology_point_macro_f1':point_o,
       'point_difference_ontology_minus_yolo':point_o-point_y,'yolo_only_accuracy':acc_y,
       'yolo_ontology_accuracy':acc_o,'bootstrap_mean_macro_f1_yolo_only':by.mean(),
       'bootstrap_mean_macro_f1_yolo_ontology':bo.mean(),'bootstrap_mean_difference_ontology_minus_yolo':d.mean(),
       'bootstrap_diff_ci95_low':lo,'bootstrap_diff_ci95_high':hi,'bootstrap_empirical_two_sided_p':p,
       'historical_reported_p_value':HIST['p'],'historical_p_value_definition_preserved':False}
    pd.DataFrame([s]).to_csv(out/'bootstrap_30case_results.csv',index=False)
    pd.DataFrame({'resample':np.arange(1,B+1),'macro_f1_yolo_only':by,'macro_f1_yolo_ontology':bo,'diff_ontology_minus_yolo':d}).to_csv(out/'bootstrap_30case_resamples.csv',index=False)
    with (out/'bootstrap_30case_results.txt').open('w',encoding='utf-8') as f:
        f.write('TSBD 30-CASE PAIRED BOOTSTRAP VERIFICATION\n'+'='*48+'\n\n')
        f.write(f'Input: {inp}\nN: {n}\nLabels: {labels}\nBootstrap: {B}\nSeed: {a.seed}\n\n')
        f.write(f'YOLO-only point Macro-F1: {point_y:.6f}\nYOLO+Ontology point Macro-F1: {point_o:.6f}\n')
        f.write(f'YOLO-only accuracy: {acc_y:.6f}\nYOLO+Ontology accuracy: {acc_o:.6f}\n\n')
        f.write(f'Bootstrap mean YOLO-only: {by.mean():.6f}\nBootstrap mean YOLO+Ontology: {bo.mean():.6f}\n')
        f.write(f'Mean paired difference: {d.mean():.6f}\n95% percentile CI: [{lo:.6f}, {hi:.6f}]\n')
        f.write(f'Two-sided empirical sign p: {p:.6f}\n\n')
        f.write('Historical manuscript values are UNVERIFIED: Ontology=0.556, YOLO=0.564, diff=-0.008, p=0.53.\n')
        f.write('The exact original p-value algorithm was not preserved. Do not force recomputed values to match history.\n')
    print(pd.DataFrame([s]).T.to_string(header=False))
    print('\nWrote outputs to',out)
if __name__=='__main__': main()
