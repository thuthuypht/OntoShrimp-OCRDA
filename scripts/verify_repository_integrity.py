from pathlib import Path
import csv, hashlib, sys

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'provenance'/'REPO_MANIFEST_SHA256.csv'

def sha256(p, chunk=1<<20):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(chunk),b''):
            h.update(b)
    return h.hexdigest()

if not MANIFEST.exists():
    raise SystemExit(f'Missing {MANIFEST}')

bad=[]; missing=[]; checked=0
with MANIFEST.open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        p=ROOT/r['relative_path']
        if not p.exists():
            missing.append(r['relative_path']); continue
        checked += 1
        got=sha256(p)
        if got != r['sha256']:
            bad.append((r['relative_path'],r['sha256'],got))

print(f'Checked: {checked}')
print(f'Missing: {len(missing)}')
print(f'Hash mismatches: {len(bad)}')
if missing:
    print('\nMissing files:')
    for x in missing: print(' -',x)
if bad:
    print('\nHash mismatches:')
    for x,e,g in bad: print(f' - {x}\n   expected {e}\n   got      {g}')
if missing or bad:
    sys.exit(1)
print('Repository integrity: PASS')
