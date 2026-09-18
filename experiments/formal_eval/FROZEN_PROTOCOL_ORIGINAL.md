# OCRDA-v7.4.4 Frozen protocol

Full OCRDA-v7.4.4 is frozen before formal baseline/ablation evaluation.

## Invariants
- Exact Full trainer SHA256: `e771cc7301de2325df7b660d3c32e9007a0615940006da66925583bd6c60b054`
- Exact ontology SHA256: `51b218118e051f4adb46b6ac92326ec25abae1af7bff00646cd484cdd9b81ccd`
- Full command overrides: epochs=30, warmup=8, max adaptation steps/epoch=72, batch=8, workers=2.
- All unspecified Full hyperparameters remain the defaults embedded in the byte-identical frozen trainer.
- Seeds: 42, 123, 2026, 3407, 7777.
- target-dev labels are diagnostic only and never used for checkpoint selection.
- Dataset 3 remains locked.

## Ablation integrity
`train_ocrda_v7_4_4_ablation.py` is derived from the exact frozen trainer. The only textual source-code modification is expansion of the argparse method choices; the relation-blind, shuffled-ontology, no-reliability, and no-topology branches already existed in the frozen source. See `FULL_TRAINER_DIFF.patch`.

## Fair initialization
Within each Batch-B seed bundle, Full OCRDA runs first and creates the exact source-warm cache. Every OCRDA ablation for that seed is required to load that same cache. The runner hard-fails otherwise.
