from pathlib import Path
import zipfile, sys
METHODS=['source_only','coral','mmd','ccda_no_ontology','relation_blind','shuffled_ontology','ocrda_no_reliability','ocrda_no_topology','full_ocrda']
SEEDS=[42,123,2026,3407,7777]
root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
expected=[f'OCRDA_v7_4_4_FROZEN_MODEL_{m.upper()}_SEED_{s}.zip' for s in SEEDS for m in METHODS]
found={p.name:p for p in root.rglob('OCRDA_v7_4_4_FROZEN_MODEL_*_SEED_*.zip')}
missing=[n for n in expected if n not in found]
print(f'Found {len(expected)-len(missing)}/45 expected model ZIPs under {root}')
if missing:
    print('\nMISSING:')
    for n in missing: print(' -',n)
    raise SystemExit(2)
out=root/'OCRDA_v744_FROZEN_45_MODELS_BUNDLE.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
    for n in expected: z.write(found[n],arcname=n)
print('\nCreated:',out)
print('Upload this outer ZIP as one Kaggle Dataset. The external evaluator will read the 45 inner model ZIPs.')
