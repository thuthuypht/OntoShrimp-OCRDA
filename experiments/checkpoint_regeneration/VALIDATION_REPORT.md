# Validation report — OCRDA-v7.4.4 frozen checkpoint regeneration package

Validation performed before packaging:

- `regenerate_frozen_models_runner.py`: Python syntax/compile PASS.
- Frozen Full trainer SHA-256: `e771cc7301de2325df7b660d3c32e9007a0615940006da66925583bd6c60b054` — PASS against frozen manifest.
- Frozen ablation trainer SHA-256: `2432169e5abd52fec22a0ff122b29e3018ab018ae02c765b392dde1d9bb26690` — PASS.
- Frozen baseline trainer SHA-256: `452753d2764b34ad5fbbaaa1d82296db6bb7623b5e37b7598b4b9759474ee75f` — PASS.
- Frozen ontology SHA-256: `51b218118e051f4adb46b6ac92326ec25abae1af7bff00646cd484cdd9b81ccd` — PASS.
- All frozen `protocol_dev` file hashes — PASS.
- Baseline trainer `--help` import smoke test — PASS.
- Full OCRDA trainer `--help` import smoke test — PASS.
- Ablation trainer `--help` import smoke test — PASS.
- Kaggle one-cell launchers compile — PASS.
- Notebook JSON/nbformat load — PASS.
- Runner dry-run on empty input: frozen integrity guard PASS; Dataset-3 guard PASS; next seed correctly detected as 42.
- Synthetic seed-bundle round-trip: 9/9 direct compact checkpoints restored after outer ZIP extraction — PASS.
- Synthetic final bundle test: 45/45 direct compact checkpoints generated — PASS.
- Compatibility smoke test against the existing `OCRDA_v7_4_4_FROZEN_DATASET3_EXTERNAL_EVAL_Kaggle` model discovery code: 45/45 checkpoints discovered, 0 missing — PASS.

The compatibility bundle deliberately stores direct paths of the form `models/<method>/seed_<seed>/compact_model_fp16.pt`. This is robust to Kaggle auto-extraction and does not depend on parsing nested model-ZIP filenames.

No GPU training experiment was executed during package construction; actual 45-checkpoint regeneration must be run on Kaggle with the original SDBD/TSBD inputs.
