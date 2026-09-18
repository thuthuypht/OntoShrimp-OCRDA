from pathlib import Path
import subprocess,sys
pkg_candidates=list(Path('/kaggle/input').rglob('external_eval_runner.py'))
if not pkg_candidates: raise RuntimeError('External-eval package not found in Kaggle Input')
pkg=sorted(pkg_candidates,key=lambda p:len(str(p)))[0]
subprocess.run([sys.executable,str(pkg),'--inventory-only'],check=False)
