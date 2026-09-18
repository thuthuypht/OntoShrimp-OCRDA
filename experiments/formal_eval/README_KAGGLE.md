# OCRDA-v7.4.4 FROZEN Baselines + Ablations

Formal evaluation package after locking Full OCRDA-v7.4.4. It does **not** change Full OCRDA hyperparameters.

## Recommended order
1. Batch A: source_only, CORAL, MMD, CCDA-no-ontology.
2. Batch B: frozen Full OCRDA + relation_blind + shuffled_ontology + no_reliability + no_topology. Full always runs first for each seed and its exact source-warm cache is reused by all OCRDA ablations.

Each execution processes one seed bundle, then stops. Download REPORT + RESUME + MODEL ZIPs before rerunning.

Do not attach Dataset 3.
