# Patch note — baseline import fix

The first frozen-evaluation package could fail at Batch A startup with `ModuleNotFoundError: No module named train_ocrda_v7_4_2`.

Cause: `core/train_baselines_frozen_eval.py` referenced an old shared-module name that was not included in this package.

Fix: the baseline trainer now imports the shared dataset/transforms/VisualModel/evaluation utilities from the included frozen `train_ocrda_v7_4_4.py`. No Full OCRDA-v7.4.4 code or hyperparameter is changed. The Full trainer SHA-256 remains frozen and is verified by the runner.
