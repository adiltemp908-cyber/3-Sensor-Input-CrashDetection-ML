# Example CSV files

Each file contains 1,600 rows at 100 Hz (16 seconds). The header is `time_s,ax,ay,az,gx,gy,gz,speed`.

| File | Provenance | Ground truth |
|---|---|---|
| `normal_driving.csv` | VZCrash test event 19088 | Provider label 2: normal driving |
| `crash.csv` | VZCrash test event 2492132 | Provider label 0: crash |
| `synthetic_pulse.csv` | Artificial signal with a pulse beginning at 8 seconds | Unknown; not a real measured event |

The two real examples belong to the retained test population, not the training population. They were chosen to illustrate a positive and negative prediction; they are not a representative evaluation sample. The synthetic file was never used to obtain the reported model results.

Real samples are from **VZCrash: A Large-Scale IMU Dataset of Ego-Vehicle Crashes**, Tommaso Bianconcini, Henrique Piñeiro Monteagudo, Aurel Pjetri, Tomaso Trinci and Leonardo Taccari, ITSC 2026. Source: [VZCrash dataset](https://huggingface.co/datasets/vzc-research-chapter/VZCrash), revision `42e1482d005c0e545379d6cfa4939486c9fe4249`. These excerpts and their sensor-plot screenshots retain the source [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/); no provider endorsement is implied.

Changes from the provider representation: acceleration g → m/s², angular velocity degrees/s → radians/s, GPS speed km/h → m/s, nominal zero-order hold of speed onto 100 Hz, and export of the prepared float32 inputs as CSV with relative timestamps. CSV round trips were checked to reproduce the prepared float32 model inputs exactly. No pulse or other perturbation was added to either real example.

`predictions.json` records measured outputs from the two bundled models. The browser deliberately assigns uploaded files an unknown label; it does not trust their filenames or attach dataset membership automatically.
