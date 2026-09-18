# Validation report

- **kaggle_frozen_suite_runner.py**: py_compile OK
- **summarize_suite.py**: py_compile OK
- **train_ocrda_v7_4_4.py**: py_compile OK
- **train_ocrda_v7_4_4_ablation.py**: py_compile OK
- **train_baselines_frozen_eval.py**: py_compile OK
- **Batch_A_Kaggle.ipynb**: notebook JSON OK
- **Batch_B_Kaggle.ipynb**: notebook JSON OK
- **frozen_full_hash**: PASS
- **ablation_diff_scope**: PASS: argparse method choices only
- **runner_batch_A_dry_run**: PASS
- **runner_batch_B_dry_run**: PASS
- **summarizer_real_v744_smoke**: PASS

Full trainer hash was verified against the frozen OCRDA-v7.4.4 source before packaging. Dataset 3 is not included.


## Patch validation: baseline-import-fix-1
- Baseline trainer old missing import removed.
- Baseline trainer imports included frozen v7.4.4 shared utilities.
- Full trainer SHA-256 remains `e771cc7301de2325df7b660d3c32e9007a0615940006da66925583bd6c60b054`.
- Baseline trainer SHA-256: `452753d2764b34ad5fbbaaa1d82296db6bb7623b5e37b7598b4b9759474ee75f`.
- Package version: 1.1.
