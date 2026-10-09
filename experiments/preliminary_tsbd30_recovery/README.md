# TSBD 30-case Bootstrap Recovery Package

## Purpose

This folder is a recovery/verification package for the **developmental 30-case TigerShrimpBD (TSBD) experiment** discussed in the manuscript.

A repository audit found no original case-level prediction file, bootstrap notebook/script, or console output for this specific preliminary experiment. Therefore, the historical bootstrap numbers currently appearing in the manuscript must be treated as **unverified until the original 30 case-level predictions are recovered or regenerated**.

## Historical manuscript values (NOT independently verified)

| Quantity | Reported value |
|---|---:|
| Number of cases | 30 |
| YOLO-only point Macro-F1 | 0.580 |
| YOLO-only accuracy | 0.567 (17/30) |
| YOLO+Ontology point Macro-F1 | 0.570 |
| YOLO+Ontology accuracy | 0.600 (18/30) |
| Bootstrap resamples | 2,000 |
| Bootstrap mean Macro-F1, YOLO+Ontology | 0.556 |
| Bootstrap mean Macro-F1, YOLO-only | 0.564 |
| Mean paired difference (Ontology - YOLO-only) | -0.008 |
| Reported p-value | 0.53 |
| Reported 95% interval | spans zero |

These are **historical manuscript values only**. They are not a substitute for source output.

## Why predictions must not be reconstructed from aggregate numbers

Macro-F1 and accuracy do not uniquely determine the 30 per-case predictions. Many different prediction tables can yield similar aggregate metrics but produce different paired bootstrap confidence intervals and p-values. Therefore, do **not** fabricate or reverse-engineer a prediction table to force the historical values.

## Files

1. `README.md` — this guide.
2. `tsbd_30_predictions_TEMPLATE.csv` — 30-row template for genuine recovered/rerun predictions.
3. `bootstrap_30case_verification.py` — standalone verification script.
4. `bootstrap_30case_verification.ipynb` — Jupyter notebook version.
5. `manuscript_values_UNVERIFIED.csv` — machine-readable historical values.
6. `bootstrap_30case_results_EXPECTED.txt` — historical expected values, explicitly unverified.
7. `Kaggle_OneCell_TSBD30_Bootstrap.ipynb` — one-cell Kaggle workflow.

## Required prediction file

Prefer the filename `tsbd_30_predictions.csv` with exactly these columns:

```text
image_id,true_label,yolo_only_pred,yolo_ontology_pred
```

Recommended labels: `BG`, `Healthy`, `WSSV`.

Do not fill the prediction columns from aggregate metrics. Use original saved predictions or rerun the original YOLO-only and YOLO+Ontology pipelines on the exact same 30 images.

## Local verification

```bash
pip install numpy pandas scikit-learn
python bootstrap_30case_verification.py --input tsbd_30_predictions.csv --bootstrap 2000 --seed 42
```

The script writes:

- `bootstrap_30case_results.csv`
- `bootstrap_30case_results.txt`
- `bootstrap_30case_resamples.csv`

## Scientific interpretation

- If genuine predictions reproduce the historical values to rounding tolerance, retain the bootstrap statement and archive the generated outputs.
- If they differ, report the recomputed values.
- If genuine predictions cannot be recovered or regenerated, remove the inferential bootstrap claim and report only the descriptive 30-case point estimates.

## Important note on p = 0.53

The exact original p-value algorithm was not preserved. The verification code reports a transparent two-sided empirical paired-bootstrap sign p-value:

`2 × min(P(diff <= 0), P(diff >= 0))`

where `diff = MacroF1(YOLO+Ontology) - MacroF1(YOLO-only)`.
