# Frozen formal experiment matrix

| Batch | Method | Target adaptation | Ontology | Frozen Full hyperparameters? | Source-warm fairness |
|---|---|---|---|---|---|
| A | source_only | No | No | N/A baseline | Shared pure visual source cache within seed |
| A | coral | Deep CORAL | No | N/A baseline | Shared pure visual source cache within seed |
| A | mmd | RBF-MMD | No | N/A baseline | Shared pure visual source cache within seed |
| A | ccda_no_ontology | Class-conditional DA | No | N/A baseline | Shared pure visual source cache within seed |
| B | full_ocrda | OCRDA-v7.4.4 | Full recognition-specific ontology | **YES, byte-identical frozen trainer** | Creates exact OCRDA source-warm cache |
| B | relation_blind | OCRDA-v7.4.4 | Relation IDs collapsed | **YES except named ablation** | Reuses same Full cache |
| B | shuffled_ontology | OCRDA-v7.4.4 | Fixed anchor permutation | **YES except named ablation** | Reuses same Full cache |
| B | ocrda_no_reliability | OCRDA-v7.4.4 | Yes | **YES except named ablation** | Reuses same Full cache |
| B | ocrda_no_topology | OCRDA-v7.4.4 | Topology/rank disabled | **YES except named ablation** | Reuses same Full cache |

Seeds: 42, 123, 2026, 3407, 7777.
