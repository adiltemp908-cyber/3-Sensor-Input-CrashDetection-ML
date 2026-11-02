"""Small artificial fixtures verify software, not crash-detection performance."""
from argparse import Namespace
import json

import numpy as np
import pytest

from test_vzcrash import fake_prepared
from crashdetect.vzcrash.data import digest, write_json
from crashdetect.vzcrash.experiment import benchmark, predict_clip, predict_prepared
from crashdetect.vzcrash.boosted import EXTENSIONS


@pytest.mark.parametrize("family", ["xgboost", "catboost"])
def test_boosted_native_artifact_roundtrip_and_unavailable_events(tmp_path, family):
    pytest.importorskip(family)
    prepared, output = tmp_path / "prepared", tmp_path / "run"
    fake_prepared(prepared)
    catalog = json.loads((prepared / "catalog.json").read_text())
    for split in ("train", "validation", "test"):
        path = prepared / f"{split}_meta.npz"
        with np.load(path) as data:
            meta = dict(data)
        meta["valid"] = np.array([False] + [True] * 11)
        meta["eligible"] = np.ones(12, dtype=bool)
        meta["physical_score"][0] = np.nan
        np.savez_compressed(path, **meta)
        catalog["files"][path.name] = digest(path)
    write_json(prepared / "catalog.json", catalog)
    args = Namespace(prepared=str(prepared), output=str(output), sensors="accel", models=[family],
                     epochs=1, patience=1, batch_size=4, threads=2, learning_rate=.001,
                     seed=7, target_recall=.95, device="cpu", max_train=0, boost_rounds=12)
    benchmark(args)
    config = json.loads((output / "selection.json").read_text())
    comparison = json.loads((output / "comparison.json").read_text())
    assert config["training_events"] == 11
    assert comparison[family]["test"]["unavailable_events"] == 1
    history = comparison[family]["history"]
    best = max(history, key=lambda h: (h["validation"]["recall"] >= .95,
                                      h["validation"]["precision"], h["validation"]["average_precision"]))
    assert comparison[family]["validation"]["threshold"] == best["validation"]["threshold"]
    scores = predict_prepared(output, prepared, "test", family, config)
    assert np.isnan(scores[0])
    clip = np.load(prepared / "test_x.npy")[1]
    result = predict_clip(output, clip, family)
    assert result["score"] == pytest.approx(scores[1], abs=1e-7)
    assert result["threshold"] == comparison[family]["validation"]["threshold"]
    assert predict_clip(output, clip[:, :3], family)["score"] == pytest.approx(result["score"], abs=1e-7)
    changed = clip.copy()
    changed[:, 3:] += 10000
    assert predict_clip(output, changed, family)["score"] == pytest.approx(result["score"], abs=1e-7)
    (output / (family + EXTENSIONS[family])).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Model changed"):
        predict_clip(output, clip, family)
