# CSV input and the local browser

## Input checklist

Use the exact header `time_s,ax,ay,az,gx,gy,gz,speed`. Supply a complete, ordered event clip with 32–20,000 rows, no missing or non-finite values, and timestamps increasing by 0.01 seconds. Typical examples contain 1,600 rows. The supported range is an input bound, not a validation claim for every length.

| Input | Required representation | Conversion if needed |
|---|---|---|
| Acceleration | m/s², including gravity | g × 9.80665 |
| Angular velocity | radians/s | degrees/s × π / 180 |
| GPS speed | m/s, nonnegative | km/h ÷ 3.6 |
| Coordinates | X forward, Y left, Z up | Apply your documented sensor-to-vehicle rotation |
| Time | Relative seconds at 100 Hz | Align streams before writing the CSV |

The training pipeline holds GPS values on the IMU grid using nominal clip indices. This does not recover unknown real sensor arrival timestamps. Do not normalize each recording to unit amplitude: doing so removes information about impact magnitude. Missing sensors require a separately trained input configuration; invented zeros are not replacement measurements.

## Terminal prediction

```bash
python scripts/predict_custom_csv.py --input examples/crash.csv
python scripts/predict_custom_csv.py --model models/xgboost --input examples/crash.csv
```

XGBoost requires `python -m pip install "xgboost==3.0.5"`. Each command prints sample count and JSON with score, frozen threshold, decision and family. `crash_clip` means the completed clip is classified as a crash, not that an impact has just occurred.

## Browser prediction

```bash
python -m crashdetect.vzcrash.browser
```

The default mode needs only the bundled histogram model. Upload an example, click **Analyze CSV**, and expand **Show sensor graphs**. The table displays 50 rows per page; **Download CSV** exports all samples. Uploads are processed in memory without adding them to a dataset or writing a recording to disk. The 5 MB file limit applies to browser uploads.

Use `--port 8878` to choose another starting port or `--no-open` to keep the browser from opening automatically. The server binds to localhost. Stop it with Ctrl+C.

## Browse the full dataset after training

After completing the [training guide](training.md), use the full model output from that same prepared corpus:

```bash
python -m crashdetect.vzcrash.browser --prepared data/vzcrash/prepared --model artifacts/trees
```

This needs `catalog.json`, prepared arrays and metadata, and the matching model's `training_event_ids.npy`. Compact bundled inference models intentionally omit the training ID list. Dataset browsing checks the catalog hash against the model's training record.

Search by provider event ID or one-based usable-record number. Event IDs are not row numbers. The browser separates provider labels, original partitions, actual training membership, excluded duplicates, sensor failures, and predictions. Example IDs: 19088 is a test normal-driving event; 2492132 is a test crash. Training examples are not independent test evidence.

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| Python is not found | Install Python and reopen the terminal; try `py` on Windows or `python3` on macOS/Linux. |
| PowerShell blocks activation | Run `.\.venv\Scripts\python.exe` directly instead of `python`. |
| `No module named ...` | Install into the interpreter you are actually using with `python -m pip install -e .`. |
| Saved scikit-learn version warning | Install the recorded version: `python -m pip install scikit-learn==1.6.1`. |
| CSV header error | Match the exact eight column names and order; use a comma delimiter. |
| Timestamp / finite-value error | Correct source data and alignment; do not silently fill missing readings. |
| Model changed after threshold was frozen | Restore the original trusted bundle or retrain and recalibrate; do not edit the hash to bypass validation. |
| Dataset browser cannot open | Use `--prepared` with the complete output of a model trained on that exact prepared dataset. |
| Training output already exists | Choose a fresh `--output` directory so old results remain intact. |
| Dataset access denied | Complete the provider's access process and run `hf auth login` locally. Never commit your token. |
