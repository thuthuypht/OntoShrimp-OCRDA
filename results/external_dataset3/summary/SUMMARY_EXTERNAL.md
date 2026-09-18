# Dataset 3 Final External Evaluation

Protocol: `OCRDA-v7.4.4-frozen-dataset3-external-eval-v1`

> Dataset 3 is a locked final external test. These results must not be used for further tuning or model selection.

| Method | Macro-F1 mean±SD | BG F1 | Healthy F1 | WSSV F1 |
|---|---:|---:|---:|---:|
| source_only | 0.399 ± 0.048 | 0.567 | 0.393 | 0.239 |
| coral | 0.392 ± 0.046 | 0.547 | 0.385 | 0.245 |
| mmd | 0.303 ± 0.059 | 0.481 | 0.218 | 0.208 |
| ccda_no_ontology | 0.403 ± 0.016 | 0.535 | 0.437 | 0.237 |
| relation_blind | 0.388 ± 0.054 | 0.530 | 0.395 | 0.239 |
| shuffled_ontology | 0.361 ± 0.052 | 0.529 | 0.375 | 0.178 |
| ocrda_no_reliability | 0.384 ± 0.038 | 0.531 | 0.382 | 0.238 |
| ocrda_no_topology | 0.379 ± 0.053 | 0.539 | 0.371 | 0.226 |
| full_ocrda | 0.377 ± 0.041 | 0.515 | 0.368 | 0.249 |
