# Dataset 3 policy during checkpoint regeneration

**Do not attach Dataset3_Vietnam_v1 to the regeneration notebook.**

The sole purpose of this package is to regenerate lost weights under a protocol already frozen before Dataset 3 is opened. The runner aborts if it detects Dataset 3 by common dataset names or locked-test metadata. Dataset 3 is attached only later, in the separate inference-only external-evaluation notebook together with the completed 45-model bundle.
