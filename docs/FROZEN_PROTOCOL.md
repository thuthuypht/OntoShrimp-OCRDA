# Frozen OCRDA-v7.4.4 protocol

The final method used for formal comparison is **OCRDA-v7.4.4**.

The authoritative frozen configuration is stored in `manifests/FROZEN_MANIFEST.json`. The manifest records five seeds, the formal method matrix, trainer SHA-256 values, ontology SHA-256, and dataset-protocol file hashes.

The formal suite contains:

- Batch A: `source_only`, `coral`, `mmd`, `ccda_no_ontology`
- Batch B: `full_ocrda`, `relation_blind`, `shuffled_ontology`, `ocrda_no_reliability`, `ocrda_no_topology`

For the full model, the frozen runner records 30 epochs, 8 warm-up epochs, at most 72 adaptation steps per epoch, batch size 8, and target-development reporting as diagnostic only.

Dataset 3 is excluded from model development and formal baseline/ablation training. Its use is governed separately by `experiments/dataset3_external_test/FINAL_EXTERNAL_TEST_POLICY.md` and `manifests/dataset3/PRIMARY_ENDPOINT_LOCK.json`.

The source code in `src/` was extracted from the frozen Kaggle packages used in the experiment. Generated Python bytecode caches are intentionally omitted.
