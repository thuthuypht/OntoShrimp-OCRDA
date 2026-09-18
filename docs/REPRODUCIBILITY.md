# Reproducibility guide

## Environment

Core dependencies are listed in the repository-level `requirements.txt`.

Recommended execution environment:

- Python 3.10+
- CUDA-capable PyTorch environment for training
- GPU recommended for the frozen 45-run formal suite
- CPU or GPU may be used for lightweight integrity checks

## Seeds and methods

Frozen seeds:

`42, 123, 2026, 3407, 7777`

Methods:

1. `source_only`
2. `coral`
3. `mmd`
4. `ccda_no_ontology`
5. `relation_blind`
6. `shuffled_ontology`
7. `ocrda_no_reliability`
8. `ocrda_no_topology`
9. `full_ocrda`

## Stage A — formal frozen evaluation on Dataset 1 / Dataset 2

Use:

- `notebooks/01_Frozen_Baselines_Batch_A_Kaggle.ipynb`
- `notebooks/02_Frozen_Full_Ablations_Batch_B_Kaggle.ipynb`

or the corresponding one-cell launchers in `experiments/formal_eval/`.

The code expects Dataset 1/2 image roots plus the protocol CSV files preserved under `manifests/dataset1_dataset2/`.

The frozen configuration is recorded in `manifests/FROZEN_MANIFEST.json`. Do not alter the frozen protocol when reproducing the reported tables.

## Stage B — checkpoint regeneration (only if original frozen weights are unavailable)

The original formal-run checkpoint files were not retained. The later Dataset 3 external test therefore used checkpoints regenerated once from the already-frozen OCRDA-v7.4.4 protocol. This regeneration was performed before using Dataset 3 for model tuning or selection.

Use:

- `notebooks/03_Regenerate_Frozen_45_Models_Kaggle.ipynb`
- `src/regenerate_frozen_models_runner.py`

The regeneration workflow is documented under `experiments/checkpoint_regeneration/`.

For methodological transparency, regenerated checkpoints should not be described as byte-identical copies of the original formal 45-run weight files. They are one-time regenerated weights produced from the frozen protocol.

## Stage C — locked Dataset 3 external inference

Before any inference, verify:

1. the Dataset 3 lock manifest;
2. exactly 150 files, 50 per class;
3. all SHA-256 checks;
4. all 45 frozen checkpoints are present;
5. the primary endpoint lock is unchanged.

Use:

- `notebooks/04_Dataset3_External_Eval_Kaggle.ipynb`
- `src/external_eval_runner.py`

The external runner performs inference only. The final external manifest records that training and tuning were not performed during this stage.

## Integrity checking

Run:

```bash
python scripts/verify_repository_integrity.py
```

This checks every file against `provenance/REPO_MANIFEST_SHA256.csv` (except the manifest itself).

## Results

Formal Dataset 1/2 development/frozen-suite outputs:

`results/development/formal_45_runs/`

Final locked Dataset 3 outputs:

`results/external_dataset3/`

The summary CSVs are intended as the canonical machine-readable tables for manuscript reporting.

## Reproducibility limits

GPU kernels, CUDA/cuDNN versions, library versions, and nondeterministic low-level operations may produce small differences across reruns even when seeds are fixed. Do not repeatedly rerun and select the best stochastic realization. A reproduction should preserve the predeclared protocol and report observed variation.
