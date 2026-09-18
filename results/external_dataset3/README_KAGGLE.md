# OCRDA-v7.4.4 Frozen Dataset 3 External Evaluation

Protocol ID: `OCRDA-v7.4.4-frozen-dataset3-external-eval-v1`.

This package performs **inference/evaluation only** on the locked Dataset 3 balanced external test (150 images: 50 BG, 50 Healthy, 50 WSSV). It contains no training loop, no optimizer, no adaptation, and no checkpoint selection.

## Required Kaggle inputs
1. This package.
2. `Dataset3_Vietnam_v1` (the exact locked dataset).
3. Frozen model artifacts from the completed formal evaluation: **45 checkpoints = 9 methods × 5 seeds**. The accepted form is either the original `OCRDA_v7_4_4_FROZEN_MODEL_<METHOD>_SEED_<seed>.zip` files or Kaggle-unpacked `compact_model_fp16.pt` files preserving method/seed folders.

The runner **aborts if any frozen checkpoint is missing**. It never retrains a missing model.

## Integrity guards
- Exact Dataset 3 image SHA-256 verification using `DATASET3_LOCK_MANIFEST.csv` (manifest SHA-256 `d42febf3efe1af7aa9359f351708e57ea1d4e604272d6269d9f540f6bfb75662`).
- Exact 50/50/50 class count verification.
- Deterministic test transform copied from frozen v7.4.4.
- No test-time augmentation/adaptation.
- One seed bundle per execution; resumable reports.

## Primary endpoint
Macro-F1. Secondary endpoints: class F1, balanced accuracy, MCC, accuracy.
