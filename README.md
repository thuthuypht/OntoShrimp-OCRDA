# OntoShrimp-OCRDA

Reproducibility package for the **frozen OCRDA-v7.4.4 shrimp-disease domain-adaptation experiments**.

This repository organizes the code, ontology, split manifests, frozen-protocol records, formal baseline/ablation results, and the locked Dataset 3 external-evaluation results used in the final experimental workflow. Raw image datasets and large model checkpoints are **not redistributed** here.

## Experimental scope

Three dataset roles are represented:

1. **Dataset 1 — SDBD (source domain):** supervised source training, source validation, and source test.
2. **Dataset 2 — TSBD (target domain):** unlabeled target adaptation plus a separate target-development diagnostic partition.
3. **Dataset 3 — Vietnam-curated external test:** a locked balanced external set of 150 images (BG 50, Healthy 50, WSSV 50), used only after the model/protocol was frozen.

The frozen study uses five seeds: **42, 123, 2026, 3407, 7777** and nine methods:

`source_only`, `coral`, `mmd`, `ccda_no_ontology`, `relation_blind`, `shuffled_ontology`, `ocrda_no_reliability`, `ocrda_no_topology`, and `full_ocrda`.

## Repository layout

```text
OntoShrimp_OCRDA_GitHub_Ready/
├── README.md
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── docs/
│   ├── DATASETS.md
│   ├── REPRODUCIBILITY.md
│   ├── FROZEN_PROTOCOL.md
│   └── GITHUB_UPLOAD_GUIDE.md
├── ontology/
│   └── ShrimpOntology.owl
├── src/
│   ├── train_ocrda_v7_4_4.py
│   ├── train_ocrda_v7_4_4_ablation.py
│   ├── train_baselines_frozen_eval.py
│   ├── kaggle_frozen_suite_runner.py
│   ├── regenerate_frozen_models_runner.py
│   ├── external_eval_runner.py
│   ├── inference_model.py
│   └── utils/
├── notebooks/
├── experiments/
├── manifests/
├── results/
│   ├── development/formal_45_runs/
│   └── external_dataset3/
└── provenance/
```

## Frozen formal development results

Target-development results are diagnostic only; Dataset 3 was not used here.

| Method | Target-dev Macro-F1 mean±SD | BG F1 | Healthy F1 | WSSV F1 | Source F1 |
|---|---:|---:|---:|---:|---:|
| source_only | 0.534 ± 0.052 | 0.307 | 0.621 | 0.676 | 0.777 |
| coral | 0.546 ± 0.044 | 0.316 | 0.624 | 0.699 | 0.782 |
| mmd | 0.597 ± 0.064 | 0.216 | 0.849 | 0.725 | 0.768 |
| ccda_no_ontology | 0.533 ± 0.131 | 0.336 | 0.578 | 0.686 | 0.801 |
| relation_blind | 0.544 ± 0.130 | 0.288 | 0.648 | 0.697 | 0.761 |
| shuffled_ontology | 0.514 ± 0.129 | 0.219 | 0.630 | 0.695 | 0.768 |
| ocrda_no_reliability | 0.562 ± 0.124 | 0.313 | 0.660 | 0.713 | 0.773 |
| ocrda_no_topology | 0.556 ± 0.106 | 0.296 | 0.665 | 0.708 | 0.773 |
| full_ocrda | 0.547 ± 0.104 | 0.308 | 0.644 | 0.688 | 0.783 |

See `results/development/formal_45_runs/summary/` for the machine-readable tables.

## Locked Dataset 3 external evaluation

The primary endpoint was locked before external inference. Dataset 3 was not used for training, adaptation, hyperparameter tuning, checkpoint selection, or seed selection.

| Method | External Macro-F1 mean±SD | BG F1 | Healthy F1 | WSSV F1 | MCC |
|---|---:|---:|---:|---:|---:|
| source_only | 0.399 ± 0.048 | 0.567 | 0.393 | 0.239 | 0.174 |
| coral | 0.392 ± 0.046 | 0.547 | 0.385 | 0.245 | 0.145 |
| mmd | 0.303 ± 0.059 | 0.481 | 0.218 | 0.208 | 0.073 |
| ccda_no_ontology | 0.403 ± 0.016 | 0.535 | 0.437 | 0.237 | 0.185 |
| relation_blind | 0.388 ± 0.054 | 0.530 | 0.395 | 0.239 | 0.163 |
| shuffled_ontology | 0.361 ± 0.052 | 0.529 | 0.375 | 0.178 | 0.130 |
| ocrda_no_reliability | 0.384 ± 0.038 | 0.531 | 0.382 | 0.238 | 0.154 |
| ocrda_no_topology | 0.379 ± 0.053 | 0.539 | 0.371 | 0.226 | 0.144 |
| full_ocrda | 0.377 ± 0.041 | 0.515 | 0.368 | 0.249 | 0.139 |

See `results/external_dataset3/summary/` for complete mean/SD/CI tables and per-seed results.

## Reproduction sequence

For a clean reproduction, follow this order:

```text
Dataset 1 + Dataset 2
        ↓
formal frozen baseline/ablation suite
        ↓
freeze protocol/model definition
        ↓
(regenerate missing frozen checkpoints if necessary)
        ↓
lock Dataset 3 and primary endpoint
        ↓
Dataset 3 inference only
```

Kaggle notebooks and one-cell launchers are included under `notebooks/` and `experiments/`.

## Data and model weights

Raw datasets and the multi-gigabyte frozen checkpoint bundle are intentionally excluded from this GitHub-ready package. See `docs/DATASETS.md` and `docs/REPRODUCIBILITY.md` for expected inputs and integrity checks.

## Important interpretation note

This repository preserves the observed results; it does not rewrite them into a claim of across-the-board superiority. The external evaluation is substantially harder than the development diagnostic setting, and results should be reported as observed.

## Citation

See `CITATION.cff`. Replace the placeholder repository URL and paper metadata after the public GitHub repository and final manuscript details are available.

## License

No explicit open-source license is assigned by this package. Before making the repository public, the authors should select a license compatible with the code, ontology, and third-party dependencies/data terms.
