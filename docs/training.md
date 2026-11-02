# Train and compare the models

Start with the [README setup](../README.md). Training is optional: the included checkpoints already run CSV predictions. Full training takes considerably longer than the demo.

## 1. Install the training tools

Inside the activated environment:

```bash
python -m pip install -e ".[dev,vzcrash,plots,boosted]"
python -m pytest
```

Expected result: software tests pass. Test fixtures are artificial and check implementation behavior; they do not measure crash-detection quality.

Historical direct versions are recorded in [`requirements/`](../requirements/). CPU preparation used PyArrow 23.0.1. The historical GPU environment recorded PyArrow 19.0.0, but that version failed to read the raw corpus; **do not use it for preparation**. Keep the supported preparation dependency from this project's install (`pyarrow>=21,<24`). These files are experiment records, not universal lockfiles.

CPU is the default. To train on an NVIDIA GPU, install an appropriate CUDA-enabled PyTorch build for your machine before using `--device cuda`. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected result for GPU training: `True`. A GPU is optional; do not pass `--device cuda` when it prints `False`.

## 2. Obtain the real dataset

Open the [VZCrash provider page](https://huggingface.co/datasets/vzc-research-chapter/VZCrash), review the terms, and complete the provider's access process using your own account. The data are CC BY-NC 4.0. Then authenticate locally:

```bash
hf auth login
python -m crashdetect.vzcrash download
```

Enter a suitable Hugging Face read token in the local login prompt, not a source file. The downloader pins revision `42e1482d005c0e545379d6cfa4939486c9fe4249`, downloads all official shards, verifies available provider checksums, and writes `data/vzcrash/raw/download.json`. The full download is approximately 7.74 GB; allow around 20 GB or more for data and outputs.

**Expected result:** the raw directory contains the official shards and download inventory. Git ignores this directory.

## 3. Prepare and audit the recordings

```bash
python -m crashdetect.vzcrash prepare --sensors all --exclude-duplicate-groups
```

The pipeline converts to SI units, holds GPS speed on the 100 Hz grid, checks labels and sensor alignment, excludes every member of exact-input duplicate groups, and fits normalization on training data only. Provider partition assignments are retained for the remaining records.

| Partition | Usable model inputs | Eligible events, including unavailable sensors |
|---|---:|---:|
| Training | 137,357 | 137,362 |
| Validation | 27,032 | 27,032 |
| Test | 24,041 | 24,042 |

**Expected result:** memory-mapped arrays and `catalog.json` under `data/vzcrash/prepared`. See the [audit](data-audit.md) for exclusions and denominator details. All future commands use this directory by default. Use fresh preparation output paths for new experiments; never mix catalogs and models.

## 4. Begin with the physical baseline

```bash
python -m crashdetect.vzcrash benchmark --models physical --output artifacts/physical
```

This uses maximum horizontal acceleration and a validation-selected threshold. It does not fit an ML model. The physical baseline is also saved alongside the learned families in subsequent benchmark runs.

**Expected output files:** `selection.json`, `comparison.json`, per-event score files, and `report.md`. The recorded baseline achieved 89.20% test AP; see [all measured results](../results/README.md).

## 5. Train basic tree models

```bash
python -m crashdetect.vzcrash benchmark --models decision_tree random_forest --forest-trees 300 --threads 4 --output artifacts/classical
```

Decision trees learn rules over the 203 clip features. Random forests combine many trees. The saved report includes each candidate's validation selection and frozen test results. These families train on CPU.

## 6. Train boosted tree models

```bash
python -m crashdetect.vzcrash benchmark --models trees --threads 4 --output artifacts/trees
python -m crashdetect.vzcrash benchmark --models xgboost catboost --boost-rounds 1000 --threads 4 --output artifacts/boosted
```

Histogram gradient boosting compares iteration counts 250/500/1,000 and leaf counts 15/31. XGBoost and CatBoost compare depths 4/6/8 at 1,000 rounds. Candidate selection uses validation performance. Add `--device cuda` to the second command for the GPU setup used by the recorded boosted experiment; inference is frozen to CPU. Device and library differences can affect exact numerical results, including GPU CatBoost nondeterminism.

## 7. Train neural models

Run from the same prepared data:

```bash
python -m crashdetect.vzcrash benchmark --models inception_time --epochs 8 --patience 3 --batch-size 16 --threads 4 --output artifacts/inception
python -m crashdetect.vzcrash benchmark --models cnn_gru cnn_transformer --epochs 8 --patience 3 --batch-size 64 --threads 4 --output artifacts/neural
```

These defaults use CPU. Add `--device cuda --amp` for the InceptionTime GPU configuration used in the recorded experiment. The original CNN–GRU and CNN–Transformer experiments used CPU. All three completed eight epochs: the selected checkpoints were epoch 6, 6 and 7 for InceptionTime, CNN–GRU and CNN–Transformer respectively.

Validation AP selects the neural checkpoint; validation recall and precision select its decision threshold. Each model accepts variable-length complete clips; padded samples are masked. A single InceptionTime network was trained, not the multi-network ensemble sometimes associated with that name.

For a small practice run, add `--max-train 10000 --epochs 2` with a fresh output directory. This intentionally changes the experiment and **will not reproduce the full-data results**. Validation and test remain complete. Training can take a long time, especially on CPU; no paid compute service is started by these commands.

## 8. Compare full runs

```bash
python scripts/compare_vzcrash_runs.py --runs artifacts/classical artifacts/trees artifacts/boosted artifacts/inception artifacts/neural --output artifacts/comparison
```

**Expected output:** `report.md`, `comparison.json`, precision–recall and learning-curve plots, and operating-point plots. The script checks matching training IDs, prepared-data identity, held-out populations, sensor availability, and model hashes. It selects by validation precision subject to 95% recall, with validation AP as tiebreaker. It does not choose by test results.

Use **complete training output directories** here. The compact bundled `models/` folders omit training IDs and per-event scores and are intended for inference only. The historical measured comparison is preserved under [`results/`](../results/).

## 9. Predict with a model you trained

```bash
python scripts/predict_custom_csv.py --model artifacts/classical --family decision_tree --input examples/crash.csv
python scripts/predict_custom_csv.py --model artifacts/classical --family random_forest --input examples/crash.csv
python scripts/predict_custom_csv.py --model artifacts/boosted --family catboost --input examples/crash.csv
python scripts/predict_custom_csv.py --model artifacts/inception --family inception_time --input examples/crash.csv
python scripts/predict_custom_csv.py --model artifacts/neural --family cnn_gru --input examples/crash.csv
python scripts/predict_custom_csv.py --model artifacts/neural --family cnn_transformer --input examples/crash.csv
```

Passing `--family` uses that candidate's own frozen validation threshold. Without it, inference uses the selected family in the supplied directory. Keep the checkpoint, `selection.json` and `comparison.json` together. Only load trusted pickle checkpoints.

## Interpret the results carefully

The target is 95% recall on validation, not a guarantee on a future test set. Random forest illustrates this: test recall was 94.89%. AP and ROC AUC exclude unavailable inputs; eligible unavailable crashes count as misses in recall. Exact-input duplicate removal does not verify vehicle independence or remove all approximate duplicates. Repeated inspection of the same test population makes the comparisons exploratory. Continuous-driving false alarms/hour and impact-to-alert delay still require a separate study.
