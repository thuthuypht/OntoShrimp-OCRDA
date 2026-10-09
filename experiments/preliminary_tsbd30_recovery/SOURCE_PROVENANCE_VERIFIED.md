# Provenance — recovered TSBD 30-case predictions

## Recovery result

The original thesis-code archive contains all material needed to recover the 30-case prediction table without fabricating any prediction:

- `mendeley_test/`: exactly 30 test images (10 Black Gill, 10 WSSV, 10 Healthy);
- `eval_results_clean.json`: 30 per-image records with ground truth, detections, YOLO-only prediction, heuristic prediction, and YOLO+Ontology prediction;
- `eval_3_systems_kaggle.py`: the evaluation program that generated `eval_results_clean.json`;
- `bootstrap_test.py`: the original 2,000-resample bootstrap procedure;
- `models_caitien/yolo11x_1024_best.pt`: the documented improved YOLO11x model used by the 30-case clean evaluation.

The set of the 30 image filenames in `eval_results_clean.json` exactly matches the 30 image files under `mendeley_test/`.

## Recovered file

`tsbd_30_predictions.csv` is a direct row-wise extraction from `eval_results_clean.json`. No labels or predictions were inferred from aggregate metrics.

Original-label mapping: `BlackGillDisease` = manuscript `BG`; `WSSV` = `WSSV`; `Healthy` = `Healthy`.

## Bootstrap verification

The original procedure uses N=30, 2,000 paired resamples with replacement, Python `random.seed(42)`, Macro-F1 over `WSSV`, `BlackGillDisease`, `Healthy`, difference = Ontology - YOLO-only, percentile 95% CI, and a **one-sided p-value** equal to the fraction of bootstrap differences <= 0.

Exact recovery:

- YOLO+Ontology bootstrap mean Macro-F1 = `0.555848148` -> **0.556**
- YOLO-only bootstrap mean Macro-F1 = `0.563905064` -> **0.564**
- mean paired difference = `-0.008056916` -> **-0.008**
- paired-difference 95% CI = `[-0.168478681, +0.145202765]` -> spans zero
- original one-sided p-value = `0.5315` -> **0.53**

Therefore the manuscript warning can be removed once these source artifacts are archived with the revision.

## SHA-256 of core source artifacts

- `eval_results_clean.json`: `72f174092e7074fada7c3397dbfe41e9b289f538a4e38f2206efb0f2c7354a98`
- `eval_3_systems_kaggle.py`: `e62116b0cf39bb890fe078c835503b72193cf2dabf0efb6883bb0fec5d3e6e9b`
- `bootstrap_test.py`: `809d86f12870ad9133f460a7c2e017a56fbe33f9b226e0992a8da65683da2160`
- `models_caitien/yolo11x_1024_best.pt`: `5b741d7d81caceaf2e120ba4437246bfcd764b25aa2f9f8cf429fcb45d5bad5b`
- recovered `tsbd_30_predictions.csv`: `8fdee57cf3099002fdca99d480acde839ab9d9bddc5217e8cfec6b5d673c407d`

See `mendeley_test_30_manifest_sha256.csv` for hashes of all 30 images.
