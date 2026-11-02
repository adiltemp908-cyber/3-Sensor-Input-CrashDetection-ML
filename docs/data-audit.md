# VZCrash data audit

Audited locally on 2026-10-11 before model training. The complete provider snapshot is pinned to `42e1482d005c0e545379d6cfa4939486c9fe4249`. All 16 Parquet files match the provider's SHA-256 values. The download inventory and hashes are in `data/vzcrash/raw/download.json`.

## Effective experiment population

| Partition | Original events | Duplicate-group exclusions | Eligible events | Sensor-unavailable | Usable inputs | Eligible crashes |
|---|---:|---:|---:|---:|---:|---:|
| Training | 137,954 | 592 | 137,362 | 5 | 137,357 | 22,034 |
| Validation | 27,175 | 143 | 27,032 | 0 | 27,032 | 4,116 |
| Test | 24,174 | 132 | 24,042 | 1 | 24,041 | 4,093 |

One unavailable training event is a crash; the unavailable test event is a non-crash. Training uses 137,357 usable records, including 22,033 crashes. No synthetic examples are added. Near misses remain negative examples. The eligible test set contains 3,632 near misses and 16,317 normal-driving records.

## Exact duplicates

Hashing the converted seven-channel model inputs found 437 extra records in 430 groups, containing 867 total records. The comparison includes accelerometer, gyroscope, and nominally aligned GPS speed after SI conversion. "Identical" refers to model inputs, not a verified claim that every pair represents the same physical incident.

| Location of first record → repeated record | Extra records |
|---|---:|
| Training → training | 198 |
| Training → validation | 100 |
| Validation → validation | 14 |
| Training → test | 103 |
| Validation → test | 15 |
| Test → test | 7 |

No duplicate event IDs were found. Two repeats within one group conflict with the first record's class label. The exclusion rule removes every member of all identical-input groups, irrespective of label or model outcome. It is applied before normalization and training. No record is moved between partitions.

This changes the effective population relative to the paper's benchmark, so our measurements are not a direct reproduction of its reported scores. Vehicle IDs are unavailable. Exact-input deduplication does not establish absence of overlapping clips, approximate duplicates, or vehicle-level leakage.

The original research workspace retained detailed and aggregate audit files. They are omitted from this compact repository. Preparation independently repeats the group audit and stores every excluded ID in its catalog.

## Sensor quality and timing

The provider's actual arrays are time-major and variable length. Complete acceleration traces are retained. Most recordings have 1,600 samples, but assuming every clip has exactly that length would truncate or mishandle real records. Some GPS arrays include an extra endpoint; nominal zero-order hold covers the IMU trace and drops unused trailing values.

Six records have unequal acceleration and gyro lengths: training IDs `2317983`, `382387`, `517196`, `411733`, `76493`, and test ID `57776`. They are marked unavailable; no synthetic alignment or interpolation is invented. Provider timestamps and impact-onset labels are absent.

Duplicate exclusions are omitted from metric denominators because this experiment evaluates the retained population. In contrast, unavailable eligible crashes count as false negatives. AP/AUC use usable records only; FPR uses usable negatives. Every report records both exclusions and unavailable counts.

PyArrow 19.0.0 failed on these Parquet files with a repetition-level histogram error. The project `.venv` uses PyArrow 23.0.1. The optimized Arrow-to-NumPy reader was checked against the original conversion on real records and has a regression test for sliced variable-length arrays.

## Reproduce

```powershell
python -m crashdetect.vzcrash download
python -m crashdetect.vzcrash prepare --exclude-duplicate-groups --output data/vzcrash/prepared-new
```

Use a new output directory. Raw data and new training outputs under `artifacts/` are ignored by Git. Two small inference bundles are included under `models/`. Access remains governed by the user's Hugging Face consent and the dataset's CC BY-NC 4.0 license.
