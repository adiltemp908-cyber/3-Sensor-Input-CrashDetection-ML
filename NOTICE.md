# Dataset attribution and scope

This repository contains original project code, measured research results, trained model bundles, two small CSV excerpts from VZCrash, and an artificial example. It does not mirror the full dataset.

VZCrash is by Tommaso Bianconcini, Henrique Piñeiro Monteagudo, Aurel Pjetri, Tomaso Trinci and Leonardo Taccari. Citation: **VZCrash: A Large-Scale IMU Dataset of Ego-Vehicle Crashes**, 2026 IEEE International Conference on Intelligent Transportation Systems (ITSC).

- Dataset: https://huggingface.co/datasets/vzc-research-chapter/VZCrash
- Source revision: `42e1482d005c0e545379d6cfa4939486c9fe4249`
- Dataset license: [Creative Commons Attribution–NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
- Derived excerpts: `examples/normal_driving.csv`, `examples/crash.csv` and their sensor-plot screenshots under `docs/images/`.
- Changes: SI unit conversion, nominal 100 Hz GPS alignment, CSV export and visualization. Full provenance is in [examples/README.md](examples/README.md).

The data license is preserved for the derived excerpts; it is not a claim that every dependency or project file has that license. No separate permissive software license is granted by this notice. The models were trained using the noncommercial research dataset; this repository does not grant commercial data rights or claim provider endorsement. Obtain the full dataset through the provider and complete its access process yourself.

The synthetic example is an artificial input demonstration with no verified crash label. The measured comparison was produced from real retained VZCrash events; no synthetic examples were counted as evaluation evidence.
