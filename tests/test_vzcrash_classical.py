"""Artificial fixtures check saved baselines, not real crash performance."""
from argparse import Namespace
import json

import numpy as np
import pytest

from test_vzcrash import fake_prepared
from crashdetect.vzcrash import classical
from crashdetect.vzcrash.data import digest, write_json
from crashdetect.vzcrash.experiment import benchmark, predict_clip, predict_prepared


@pytest.mark.parametrize("family", ["decision_tree", "random_forest"])
def test_classical_roundtrip_selection_missing_inputs_and_integrity(tmp_path, family, monkeypatch):
    prepared, output = tmp_path / "prepared", tmp_path / "run"
    fake_prepared(prepared)
    # Adapt leaf counts only for this tiny fixture so the trees actually split.
    parameters = classical.candidate_parameters
    monkeypatch.setattr(classical, "candidate_parameters", lambda *a, **kw:
                        [{**p, "min_samples_leaf": 2} for p in parameters(*a, **kw)])
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
                     seed=7, target_recall=.95, device="cpu", max_train=0, forest_trees=8)
    benchmark(args)
    config = json.loads((output / "selection.json").read_text())
    comparison = json.loads((output / "comparison.json").read_text())
    assert config["training_events"] == 11
    assert comparison[family]["test"]["unavailable_events"] == 1
    history = comparison[family]["history"]
    assert len(history) == 3
    best = max(history, key=lambda h: (h["validation"]["recall"] >= .95,
                                      h["validation"]["precision"], h["validation"]["average_precision"]))
    assert comparison[family]["validation"]["threshold"] == best["validation"]["threshold"]
    scores = predict_prepared(output, prepared, "test", family, config)
    assert np.isnan(scores[0])
    assert len(np.unique(scores[1:])) > 1
    clip = np.load(prepared / "test_x.npy")[1]
    result = predict_clip(output, clip, family)
    assert result["score"] == pytest.approx(scores[1], abs=1e-12)
    assert result["threshold"] == comparison[family]["validation"]["threshold"]
    assert predict_clip(output, clip[:, :3], family)["score"] == pytest.approx(result["score"], abs=1e-12)
    changed = clip.copy()
    changed[:, 3:] += 10000
    assert predict_clip(output, changed, family)["score"] == pytest.approx(result["score"], abs=1e-12)
    (output / f"{family}.pkl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Model changed"):
        predict_clip(output, clip, family)
