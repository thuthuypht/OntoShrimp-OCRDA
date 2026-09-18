# OCRDA-v7.4.4 REGENERATE FROZEN 45 MODELS — Kaggle

Use this package only because the original formal-evaluation checkpoints were lost. It regenerates exactly one seed bundle per execution, nine frozen methods per seed, without Dataset 3 and without tuning.

## Inputs required for the first seed
1. This package.
2. OntoShrimp-SDBD.
3. OntoShrimp-TSBD.

Do **not** attach Dataset 3.

## Main runner
`regenerate_frozen_models_runner.py`

Use `KAGGLE_ONE_CELL_REGENERATE.txt`. After each execution download:
- `OCRDA_v7_4_4_REGENERATED_MODELS_SEED_<seed>.zip`
- `OCRDA_v7_4_4_REGEN_RESUME_LATEST.zip`
- `OCRDA_v7_4_4_REGEN_REPORTS_LATEST.zip`

If you close Kaggle, attach the latest lightweight resume **and all previously downloaded seed-model bundles** as Kaggle Inputs before the next run. Kaggle may unpack the outer seed bundle; the runner supports both zipped and unpacked forms.

After the fifth seed, the runner creates `OCRDA_v744_FROZEN_45_MODELS_BUNDLE.zip` if all five seed bundles/model ZIPs are visible. If not, attach the missing prior seed bundles and run `KAGGLE_ONE_CELL_BUILD_FINAL_BUNDLE.txt` (no GPU training).
