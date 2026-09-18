# Dataset roles and release policy

## Dataset 1 — SDBD

Role: **source domain**.

Used for supervised source training, source validation, and source testing. The repository includes the protocol CSV manifests used by the frozen experiment under `manifests/dataset1_dataset2/`, but it does not redistribute the underlying image files.

Relevant manifests:

- `source_train_final.csv`
- `source_val_final.csv`
- `source_test_final.csv`
- `manifest_final.csv`

## Dataset 2 — TSBD

Role: **target domain during development**.

The frozen OCRDA protocol uses `target_adapt_final.csv` for target adaptation with labels hidden from training. `target_dev_exposed.csv` is used only for post-hoc development diagnostics and not for target-aware checkpoint selection in the frozen formal suite.

Relevant manifests:

- `target_adapt_final.csv`
- `target_dev_exposed.csv`
- `audit_excluded_target_records.json`
- `excluded_reason_summary.csv`

## Dataset 3 — Vietnam-curated external test

Role: **final external evaluation only**.

Locked composition:

- BG: 50 images
- Healthy: 50 images
- WSSV: 50 images
- Total: 150 images

The term **Vietnam-curated** (or Vietnam-assembled) is preferred. The package does not claim that every image was captured directly in Vietnam by the project team.

The exact locked image identities and SHA-256 hashes are stored in:

`manifests/dataset3/DATASET3_LOCK_MANIFEST.csv`

The cross-dataset duplicate audit is preserved in:

`manifests/dataset3_dedup/`

The final dedup summary recorded 150 retained Dataset 3 images, with no exact match, no high-confidence near duplicate, and no medium manual-review pair against Dataset 1/Dataset 2.

## Independence rule

Dataset 3 was locked before final external evaluation and was not used for:

- training;
- domain adaptation;
- hyperparameter tuning;
- checkpoint selection;
- seed selection;
- test-time adaptation.

## Why raw images are not in this repository

This GitHub-ready package intentionally excludes raw datasets. Before redistributing images publicly, verify the license, source provenance, consent/copyright terms, and redistribution rights for every contributing source. For reproducibility, the repository instead exposes split manifests, lock hashes, protocol metadata, and code.
