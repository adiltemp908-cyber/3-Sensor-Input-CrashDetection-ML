"""Artificial fixtures test software only. They are not crash performance evidence."""
from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from crashdetect.vzcrash.data import REPO, batch_rows, decode, duplicate_groups, feature_vector, digest, load_catalog, prepare, write_json
from crashdetect.vzcrash.experiment import threshold_for_recall, metrics, benchmark, predict_clip, predict_prepared
from crashdetect.vzcrash.model import ClipNet, make_model


def sample_row(label=0):
    return {"label": label, "gsensor": np.tile([[0.], [0.], [1.]], (1, 1600)),
            "gyro": np.full((3, 1600), 180.), "gps_speed": np.arange(16) * 3.6}


def test_provider_units_labels_and_past_only_speed():
    x, y = decode(sample_row())
    assert y == 1 and x.shape == (1600, 7)
    np.testing.assert_allclose(x[:, 2], 9.80665)
    np.testing.assert_allclose(x[:, 3:6], np.pi)
    assert x[99, 6] == 0 and x[100, 6] == pytest.approx(1)
    row = sample_row(1)
    row["gps_speed"][10:] = 999
    changed, y = decode(row)
    assert y == 0
    np.testing.assert_array_equal(changed[:1000, 6], x[:1000, 6])
    assert decode(sample_row(2))[1] == 0


def test_malformed_provider_data_rejected():
    row = sample_row()
    row["gsensor"] = np.zeros((3, 1599))
    with pytest.raises(ValueError):
        decode(row)
    row = sample_row()
    row["gyro"][0, 0] = np.nan
    with pytest.raises(ValueError):
        decode(row)
    row = sample_row(3)
    with pytest.raises(ValueError):
        decode(row)


def test_arrow_fixed_width_reader_preserves_sliced_event_boundaries():
    pa = pytest.importorskip("pyarrow")
    signals = [np.arange(n * 3, dtype=np.float32).reshape(n, 3).tolist() for n in (1600, 1501, 1600)]
    batch = pa.record_batch({
        "event_id": pa.array([10, 11, 12]), "label": pa.array([0, 1, 2]),
        "gsensor": pa.array(signals, type=pa.list_(pa.list_(pa.float32(), 3))),
        "gyro": pa.array(signals, type=pa.list_(pa.list_(pa.float32(), 3))),
        "gps_speed": pa.array([[0.] * 16] * 3, type=pa.list_(pa.float32())),
    }).slice(1, 2)
    for fast, reference in zip(batch_rows(batch), batch.to_pylist()):
        assert fast["event_id"] == reference["event_id"]
        np.testing.assert_array_equal(decode(fast)[0], decode(reference)[0])


def raw_fixture(path, duplicate=False):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    (path / "data").mkdir(parents=True)
    files = []
    for number, split in enumerate(("train", "validation", "test")):
        rows = []
        for i in range(2):
            row = sample_row(i)
            row["gsensor"][0] = number * 10 + i
            row.update(event_id=number*2+i, vehiclesize="Light", gyro_is_hd=False)
            row["gsensor"] = row["gsensor"].T
            row["gyro"] = row["gyro"].T
            if duplicate and split == "test" and i == 0:
                row["event_id"] = 0
            rows.append({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()})
        filename = f"data/{split}-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path / filename)
        files.append({"path": filename, "sha256": digest(path / filename)})
    write_json(path / "download.json", {"repo": REPO, "revision": "test-fixture", "files": files})


def test_real_schema_preparation_preserves_splits_and_training_scaler(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    raw_fixture(raw)
    catalog = prepare(raw, output)
    assert catalog["splits"]["test"] == {"events": 2, "crashes": 1, "noncrashes": 1}
    assert catalog["scaler"]["mean"][0] == pytest.approx(.5 * 9.80665)
    assert np.load(output / "test_meta.npz")["event_id"].tolist() == [4, 5]
    assert load_catalog(output) == catalog


def test_duplicate_event_across_partitions_rejected(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    raw_fixture(raw, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(raw, output)
    assert not (output / "catalog.json").exists()


def test_duplicate_groups_exclude_both_partitions_without_label_selection(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    raw_fixture(tmp_path)
    paths = {split: sorted((tmp_path / "data").glob(f"{split}-*.parquet")) for split in ("train", "validation", "test")}
    training = pq.read_table(paths["train"][0]).to_pylist()
    testing = pq.read_table(paths["test"][0]).to_pylist()
    testing[0] = {**training[0], "event_id": 4, "label": 2}
    pq.write_table(pa.Table.from_pylist(testing), paths["test"][0])
    excluded = duplicate_groups(paths, "all")
    assert set(excluded) == {0, 4}
    assert excluded[0]["split"] == "train" and excluded[4]["split"] == "test"


def test_duplicate_exclusion_is_distinct_from_unavailable_sensor_data():
    y, score = np.array([1, 1, 1, 0]), np.array([np.nan, np.nan, .9, .1])
    eligible = np.array([False, True, True, True])
    result = metrics(y, score, .5, eligible)
    assert result["original_events"] == 4 and result["events"] == 3
    assert result["excluded_duplicate_events"] == 1
    assert result["unavailable_crashes"] == 1 and result["recall"] == .5
    assert threshold_for_recall(y, score, .5, eligible) == .9


def test_threshold_ties_and_unmeasured_stream_metrics():
    y = np.array([1, 1, 0, 0])
    score = np.array([.8, .8, .8, .1])
    threshold = threshold_for_recall(y, score, 1.)
    result = metrics(y, score, threshold)
    assert result["tp"] == 2 and result["fp"] == 1
    assert result["recall"] == 1
    assert result["false_alerts_per_hour"] is None
    assert result["impact_to_alert_delay_s"] is None
    assert metrics([0, 0], [.1, .2], .5)["recall"] is None


@pytest.mark.parametrize("family", ["cnn_gru", "cnn_transformer"])
def test_clip_network_has_trainable_gradient(family):
    torch.set_num_threads(2)
    model = ClipNet(3, family)
    logits = model(torch.randn(2, 1600, 3))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
    assert logits.shape == (2,)
    assert model.encoder[0].weight.grad.abs().sum() > 0


def fake_prepared(path):
    path.mkdir()
    rng = np.random.default_rng(1)
    for split in ("train", "validation", "test"):
        x = rng.normal(size=(12, 1600, 7)).astype(np.float32)
        x[:, :, 6] = np.abs(x[:, :, 6])
        y = np.arange(12) % 2
        x[y == 1, 800:810, :3] += 10
        np.save(path / f"{split}_x.npy", x)
        np.savez_compressed(path / f"{split}_meta.npz", y=y, event_id=np.arange(12),
                            vehiclesize=np.array(["Light"]*12), gyro_is_hd=y.astype(bool), provider_label=np.where(y, 0, 2),
                            features=np.array([feature_vector(v) for v in x]),
                            physical_score=np.linalg.norm(x[:, :, :2], axis=2).max(1)/9.80665)
    write_json(path / "catalog.json", {"scaler": {"mean": [0.]*7, "std": [1.]*7},
                                       "channels": list("abcdefg"), "task": "artificial test fixture",
                                       "source": {"repo": "test-fixture", "revision": "test"},
                                       "files": {p.name: digest(p) for p in path.iterdir()}})


def test_end_to_end_local_artifacts_and_input_validation(tmp_path):
    prepared, output = tmp_path / "prepared", tmp_path / "models"
    fake_prepared(prepared)
    args = Namespace(prepared=str(prepared), output=str(output), sensors="accel", models=["trees", "cnn_gru", "cnn_transformer", "inception_time"],
                     epochs=1, patience=1, batch_size=4, threads=2, learning_rate=.001, seed=7,
                     target_recall=.95, device="cpu", max_train=0)
    result = benchmark(args)
    config = json.loads((output / "selection.json").read_text())
    assert config["deployment_validated"] is False
    assert result["test"]["events"] == 12
    x = np.load(prepared / "test_x.npy")
    comparison = json.loads((output / "comparison.json").read_text())
    original_selection = (output / "selection.json").read_bytes()
    for family in ("physical", "trees", "cnn_gru", "cnn_transformer", "inception_time"):
        explicit = predict_clip(output, x[0], family=family)
        assert explicit["model"] == family
        assert explicit["threshold"] == comparison[family]["validation"]["threshold"]
        assert explicit["crash_clip"] == (explicit["score"] >= explicit["threshold"])
        batch_score = predict_prepared(output, prepared, "test", family, config)
        assert explicit["score"] == pytest.approx(batch_score[0], abs=1e-6)
        assert (output / "selection.json").read_bytes() == original_selection
    with pytest.raises(ValueError, match="not in this benchmark"):
        predict_clip(output, x[0], family="missing")
    for family in ("physical", "trees", "cnn_gru", "cnn_transformer", "inception_time"):
        config["selected"] = family
        write_json(output / "selection.json", config)
        batch_score = predict_prepared(output, prepared, "test", family, config)
        single = predict_clip(output, x[0])
        assert single["score"] == pytest.approx(batch_score[0], abs=1e-6)
        assert predict_clip(output, x[0, :, :3])["score"] == pytest.approx(single["score"], abs=1e-6)
        # An accelerometer-only model must ignore unused sensor values.
        changed = x[0].copy()
        changed[:, 3:] += 1000
        assert predict_clip(output, changed)["score"] == pytest.approx(single["score"], abs=1e-6)
    with pytest.raises(ValueError):
        predict_clip(output, np.zeros((10, 7)))
    with pytest.raises(ValueError):
        benchmark(args)
    (output / "inception_time.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Model changed"):
        predict_clip(output, x[0])
    (prepared / "test_meta.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        load_catalog(prepared)


@pytest.mark.parametrize("family", ["cnn_transformer", "cnn_gru", "inception_time"])
def test_variable_length_padding_is_invariant(family):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    net = make_model(7, family).eval()
    short = torch.randn(1, 1501, 7)
    mixed = torch.randn(2, 4096, 7)
    mixed[0, :1501] = short[0]
    mixed[0, 1501:] = 0
    with torch.inference_mode():
        a = net(short)
        b = net(mixed, torch.tensor([1501, 4096]))[:1]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_variable_records_and_missing_speed_coverage():
    row = sample_row()
    row["gsensor"] = row["gsensor"][:, :1420]
    row["gyro"] = row["gyro"][:, :1420]
    row["gps_speed"] = row["gps_speed"][:15]
    x, _ = decode(row)
    assert x.shape == (1420, 7)
    row["gps_speed"] = row["gps_speed"][:10]
    with pytest.raises(ValueError, match="cover"):
        decode(row)
    assert decode(row, sensors="imu")[0].shape == (1420, 7)


def test_unusable_crashes_remain_in_recall_denominator():
    y = np.array([1, 1, 0, 0])
    scores = np.array([.9, np.nan, .1, np.nan])
    result = metrics(y, scores, .5)
    assert result["recall"] == .5
    assert result["fn"] == 1
    assert result["unavailable_events"] == 2
    assert result["tn"] == 1
    assert result["noncrashes"] == 2
    assert threshold_for_recall(y, scores, .95) == .9
