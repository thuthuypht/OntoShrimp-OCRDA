# OCRDA-v7.4.4 FROZEN Formal Baseline/Ablation Summary

> Full OCRDA-v7.4.4 is frozen. target-dev is diagnostic/development only. Dataset 3 is not used here.

| Method | n | Target-dev Macro-F1 mean±SD | BG F1 | Healthy F1 | WSSV F1 | Source F1 |
|---|---:|---:|---:|---:|---:|---:|
| source_only | 5 | 0.534 ± 0.052 | 0.307 | 0.621 | 0.676 | 0.777 |
| coral | 5 | 0.546 ± 0.044 | 0.316 | 0.624 | 0.699 | 0.782 |
| mmd | 5 | 0.597 ± 0.064 | 0.216 | 0.849 | 0.725 | 0.768 |
| ccda_no_ontology | 5 | 0.533 ± 0.131 | 0.336 | 0.578 | 0.686 | 0.801 |
| relation_blind | 5 | 0.544 ± 0.130 | 0.288 | 0.648 | 0.697 | 0.761 |
| shuffled_ontology | 5 | 0.514 ± 0.129 | 0.219 | 0.630 | 0.695 | 0.768 |
| ocrda_no_reliability | 5 | 0.562 ± 0.124 | 0.313 | 0.660 | 0.713 | 0.773 |
| ocrda_no_topology | 5 | 0.556 ± 0.106 | 0.296 | 0.665 | 0.708 | 0.773 |
| full_ocrda | 5 | 0.547 ± 0.104 | 0.308 | 0.644 | 0.688 | 0.783 |