from pathlib import Path
import re,sys,zipfile
METHODS=['source_only','coral','mmd','ccda_no_ontology','relation_blind','shuffled_ontology','ocrda_no_reliability','ocrda_no_topology','full_ocrda']
SEEDS=[42,123,2026,3407,7777]
root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
expected={f'OCRDA_v7_4_4_FROZEN_MODEL_{m.upper()}_SEED_{s}.zip' for s in SEEDS for m in METHODS}
found={p.name for p in root.rglob('OCRDA_v7_4_4_FROZEN_MODEL_*_SEED_*.zip')}
print(f'Visible individual model ZIPs: {len(expected & found)}/45')
for s in SEEDS:
    n=sum(f'OCRDA_v7_4_4_FROZEN_MODEL_{m.upper()}_SEED_{s}.zip' in found for m in METHODS)
    print(f'seed {s}: {n}/9')
missing=sorted(expected-found)
if missing:
    print('MISSING:'); [print(' -',x) for x in missing]
