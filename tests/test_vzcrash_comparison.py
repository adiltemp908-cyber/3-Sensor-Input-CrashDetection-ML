"""Comparison safeguards use artificial metrics, never performance evidence."""
import numpy as np
import pytest

from crashdetect.vzcrash.data import digest, write_json
from scripts.compare_vzcrash_runs import compare_runs


def completed_run(path, family, validation_precision, test_precision):
    path.mkdir()
    weights = path / f"{family}.pt"
    weights.write_bytes(b"test fixture, not model weights")
    write_json(path / "selection.json", {
        "catalog_sha256": "fixture", "source_revision": "fixture", "columns": [0, 1, 2],
        "scaler": {"mean": [0]*7, "std": [1]*7}, "target_recall": .95,
        "training_events": 2, "model_sha256": {weights.name: digest(weights)}})
    np.save(path / "training_event_ids.npy", [1, 2])
    write_json(path / "comparison.json", {family: {
        "validation": {"recall": .96, "precision": validation_precision, "average_precision": .9},
        "test": {"precision": test_precision}, "history": []}})
    for split in ("validation", "test"):
        np.savez(path / f"{family}_{split}_scores.npz", event_id=[3, 4], y=[0, 1],
                 eligible=[True, True], score=[.1, .9])


def test_comparison_selects_validation_winner_and_verifies_populations(tmp_path):
    gru, transformer = tmp_path / "gru", tmp_path / "transformer"
    completed_run(gru, "cnn_gru", .9, .8)
    completed_run(transformer, "cnn_transformer", .8, .99)
    result = compare_runs([gru, transformer])
    assert result["selected"] == "cnn_gru"  # Better test precision must not drive selection.
    with pytest.raises(ValueError, match="Ambiguous repeated"):
        compare_runs([gru, gru])
    np.savez(transformer / "cnn_transformer_test_scores.npz", event_id=[3, 5], y=[0, 1],
             eligible=[True, True], score=[.1, .9])
    with pytest.raises(ValueError, match="populations"):
        compare_runs([gru, transformer])


def test_comparison_rejects_different_training_set_and_changed_weights(tmp_path):
    gru, transformer = tmp_path / "gru", tmp_path / "transformer"
    completed_run(gru, "cnn_gru", .9, .8)
    completed_run(transformer, "cnn_transformer", .8, .99)
    np.save(transformer / "training_event_ids.npy", [1, 5])
    with pytest.raises(ValueError, match="identical prepared"):
        compare_runs([gru, transformer])
    np.save(transformer / "training_event_ids.npy", [1, 2])
    (transformer / "cnn_transformer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Model changed"):
        compare_runs([gru, transformer])
