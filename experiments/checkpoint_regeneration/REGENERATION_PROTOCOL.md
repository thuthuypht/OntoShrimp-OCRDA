# OCRDA-v7.4.4 Frozen Checkpoint Regeneration Protocol

This package does **not** develop or tune a new method. It regenerates checkpoint weights that were lost after the formal 45-run evaluation.

## Locked invariants
- Full OCRDA-v7.4.4 trainer, ablation trainer, baseline trainer, ontology, and protocol CSV/JSON files are hash-checked against the frozen formal package.
- Seeds are fixed: 42, 123, 2026, 3407, 7777.
- The same frozen command overrides are used: Full OCRDA epochs=30, warm-up=8, max adaptation steps/epoch=72, batch=8, workers=2; baselines source epochs=8, adaptation epochs=22, max steps=72, batch=8, workers=2.
- ImageNet-pretrained ConvNeXt-Tiny is required. The runner aborts rather than silently switching to non-pretrained weights.
- Each missing method/seed is regenerated once. The target-dev metric is audit/diagnostic only; it never triggers a rerun or model choice.
- Dataset 3 is forbidden and is guarded against in Kaggle Inputs.

## Fair initialization
For each seed, `full_ocrda` creates the common OCRDA source-warm cache; all four OCRDA ablations reuse that exact cache. The four visual baselines share the same pure visual source-warm cache for that seed.

## Outputs
After each seed: one outer `OCRDA_v7_4_4_REGENERATED_MODELS_SEED_<seed>.zip` containing nine direct `models/<method>/seed_<seed>/compact_model_fp16.pt` checkpoints, per-model metrics, and SHA-256 hashes. A separate lightweight resume/report is also created.

After all five seed bundles are visible: `OCRDA_v744_FROZEN_45_MODELS_BUNDLE.zip`, directly intended for the locked Dataset 3 external-evaluation workflow.
