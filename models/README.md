# Included inference models

| Folder | Model | Why included | Frozen threshold |
|---|---|---|---:|
| `hist-gradient-boosting/` | scikit-learn histogram gradient boosting | Default CSV/browser demo | 0.5562889424776457 |
| `xgboost/` | XGBoost | Validation-selected family across all eight models | 0.5477622151374817 |

Each bundle has the trained weights, `selection.json` (features, scaler, threshold, versions and SHA-256), and `comparison.json` (metrics for available inference families). Both use all seven channels and 203 derived features. Together the weight files are about 8 MB. No training dataset is needed for inference.

Use scikit-learn **1.6.1** for the pickle checkpoint and XGBoost **3.0.5** for the native XGBoost checkpoint. Load only trusted model bundles. Inference checks the weight hash against the frozen selection record; this detects changes, not a malicious replacement of both files.

These are compact inference bundles, not complete training-run directories: per-event scores and training-event ID arrays are intentionally omitted. Use a newly trained full output directory for the comparison script or dataset browser. Model weights are unchanged from the original training runs; the XGBoost bundle's metadata lists only the included XGBoost family. Historical training arguments remain recorded in `selection.json`.

For all eight models' measured results, see [results/README.md](../results/README.md). For regeneration of full runs, see [docs/training.md](../docs/training.md).
