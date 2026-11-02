"""Download and prepare official VZCrash partitions without changing their membership."""
from pathlib import Path
import hashlib
import json

import numpy as np

REPO = "vzc-research-chapter/VZCrash"
REVISION = "42e1482d005c0e545379d6cfa4939486c9fe4249"
SPLITS = ("train", "validation", "test")
CHANNELS = ["ax_m_s2", "ay_m_s2", "az_m_s2", "gx_rad_s", "gy_rad_s", "gz_rad_s", "gps_m_s"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def download(destination, revision=REVISION):
    from huggingface_hub import HfApi, get_token, hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    if not get_token():
        raise ValueError("Accept the VZCrash conditions in your browser, then run hf auth login in your own terminal. Never paste a token into chat.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(REPO, revision=revision, files_metadata=True)
    files = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.endswith(".parquet") or s.rfilename in ("README.md", "explore.ipynb", "pr_curve.ipynb"))
    expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs is not None}
    inventory = []
    try:
        # Read the card first to fail quickly if consent is still missing.
        hf_hub_download(REPO, "README.md", repo_type="dataset", revision=info.sha, local_dir=destination)
        for filename in files:
            print(f"Downloading {filename}", flush=True)
            path = hf_hub_download(REPO, filename, repo_type="dataset", revision=info.sha, local_dir=destination,
                                   force_download=not (destination / filename).exists())
            checksum = digest(path)
            if filename in expected and checksum != expected[filename]:
                raise ValueError(f"Provider checksum mismatch for {filename}; download again before use")
            inventory.append({"path": filename, "bytes": Path(path).stat().st_size, "sha256": checksum})
    except GatedRepoError as exc:
        raise ValueError("VZCrash access has not been granted for the locally logged-in account. Complete dataset consent using that account.") from exc
    write_json(destination / "download.json", {"repo": REPO, "revision": info.sha, "license": "CC-BY-NC-4.0", "files": inventory})


def decode(row, sensors="all"):
    """Provider units -> SI. Keep gravity and impact amplitude. No per-event scaling."""
    label = row["label"]
    if label not in (0, 1, 2):
        raise ValueError(f"Unknown provider label: {label}")
    signals = []
    for name, factor in (("gsensor", 9.80665), ("gyro", np.pi / 180)):
        if name == "gyro" and sensors == "accel":
            signals.append(np.zeros_like(signals[0]))
            continue
        values = np.asarray(row[name], dtype=np.float32)
        # Provider notebook confirms time-major arrays and variable event lengths.
        if values.ndim == 2 and values.shape[0] == 3 and values.shape[1] != 3:
            values = values.T
        if values.ndim != 2 or values.shape[1] != 3 or not 32 <= len(values) <= 20000 or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected finite 100 Hz three-axis data (32..20000 samples), got {values.shape}")
        if signals and len(values) != len(signals[0]):
            raise ValueError("Accelerometer/gyro lengths differ; do not invent alignment")
        signals.append(values * factor)
    n = len(signals[0])
    if sensors == "all":
        speed = np.asarray(row["gps_speed"], dtype=np.float32)
        if speed.ndim != 1 or not np.isfinite(speed).all() or (speed < 0).any() or len(speed)*100 < n:
            raise ValueError("gps_speed: finite nonnegative 1 Hz values must cover the complete IMU clip; consider IMU-only preparation")
        speed = np.repeat(speed / 3.6, 100)[:n]
    else:
        speed = np.zeros(n, np.float32)
    # Nominal synchronized clip clock: zero-order hold; never interpolate future GPS.
    # This is a clip-classification convention, not a verified online arrival clock.
    signals.append(speed[:, None])
    x = np.concatenate(signals, axis=1)
    return x, int(label == 0)


def feature_vector(x):
    """Fixed clip statistics for a CPU-friendly nonlinear baseline."""
    x = np.asarray(x, dtype=np.float32)
    derivatives = np.diff(x, axis=0) * 100.0
    blocks = [x.mean(0), x.std(0), x.min(0), x.max(0),
              np.quantile(x, .05, axis=0), np.quantile(x, .95, axis=0),
              np.abs(derivatives).max(0), derivatives.std(0)]
    # Coarse temporal context retains speed changes and pre/post-event dynamics.
    for segment in np.array_split(x, 4):
        blocks.extend((segment.mean(0), segment.std(0), np.abs(segment).max(0)))
    # Sustained pulses distinguish impacts/deceleration from short vibration peaks.
    cumulative = np.vstack((np.zeros((1, 7)), np.cumsum(x, axis=0, dtype=np.float64)))
    for width in (10, 30, 100, 300):
        width = min(width, len(x))
        rolling = (cumulative[width:] - cumulative[:-width]) / width
        blocks.extend((rolling.min(0), rolling.max(0)))
    blocks.append(x[-1] - x[0])
    return np.concatenate(blocks).astype(np.float32)


def batch_rows(batch):
    """Keep large sensor arrays in NumPy rather than allocating millions of floats."""
    import pyarrow as pa

    scalar_names = [name for name in batch.schema.names if name not in ("gsensor", "gyro", "gps_speed")]
    scalars = batch.select(scalar_names).to_pylist()
    for i, row in enumerate(scalars):
        for name in ("gsensor", "gyro", "gps_speed"):
            cell = batch.column(batch.schema.get_field_index(name))[i]
            if not cell.is_valid:
                row[name] = None
                continue
            values = cell.values
            if pa.types.is_fixed_size_list(values.type) and values.null_count == 0:
                width = values.type.list_size
                flat = values.values.slice(values.offset * width, len(values) * width)
                row[name] = flat.to_numpy(zero_copy_only=False).reshape(-1, width)
            elif pa.types.is_floating(values.type):
                row[name] = values.to_numpy(zero_copy_only=False)
            else:
                row[name] = values.to_pylist()
        yield row


def duplicate_groups(paths, sensors):
    """Exclude every member of identical-signal groups, without selecting by label."""
    import pyarrow.parquet as pq

    seen_ids, seen_signals, excluded = set(), {}, {}
    for split, entries in paths.items():
        for path in entries:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch_rows(batch):
                    event_id = int(row["event_id"])
                    if event_id in seen_ids:
                        raise ValueError(f"Duplicate event ID: {event_id}")
                    seen_ids.add(event_id)
                    try:
                        values, _ = decode(row, sensors)
                    except ValueError:
                        continue
                    key = hashlib.sha256(values.tobytes()).digest()
                    member = {"event_id": event_id, "split": split}
                    if key in seen_signals:
                        first = seen_signals[key]
                        excluded[first["event_id"]] = {**first, "group_first_event_id": first["event_id"]}
                        excluded[event_id] = {**member, "group_first_event_id": first["event_id"]}
                    else:
                        seen_signals[key] = member
        print(f"Duplicate audit complete: {split}", flush=True)
    return excluded


def prepare(raw, output, sensors="all", exclude_duplicate_groups=False):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    raw, output = Path(raw), Path(output)
    if output.exists():
        raise ValueError("Choose a new prepared directory; existing preparations are never overwritten")
    source = json.loads((raw / "download.json").read_text(encoding="utf-8"))
    if source.get("repo") != REPO:
        raise ValueError("Expected a VZCrash download inventory")
    inventory = {item["path"]: item for item in source["files"]}
    paths = {split: sorted((raw / "data").glob(f"{split}-*.parquet")) for split in SPLITS}
    expected_paths = {name for name in inventory if name.endswith(".parquet")}
    observed_paths = {p.relative_to(raw).as_posix() for entries in paths.values() for p in entries}
    if expected_paths != observed_paths or not all(paths.values()):
        raise ValueError("Missing/unexpected shards; run the complete downloader first")
    for entries in paths.values():
        for path in entries:
            if digest(path) != inventory[path.relative_to(raw).as_posix()]["sha256"]:
                raise ValueError(f"Checksum mismatch: {path.name}")
    excluded = duplicate_groups(paths, sensors) if exclude_duplicate_groups else {}
    output.mkdir(parents=True)
    catalog = {"source": source, "channels": CHANNELS, "sample_hz": 100, "samples": "variable", "available_sensors": sensors,
               "task": "variable-length event-clip classification", "split_basis": "official provider partitions with documented duplicate exclusions; vehicle IDs not released",
               "speed_alignment": "nominal clip-index zero-order hold, 100 IMU samples per GPS value", "splits": {},
               "duplicate_policy": "exclude every member of exact-signal groups" if exclude_duplicate_groups else "error",
               "duplicate_exclusions": list(excluded.values())}
    seen_ids, seen_signals = set(), set()
    sums = np.zeros(7, np.float64)
    squares = np.zeros(7, np.float64)
    count = 0
    for split, entries in paths.items():
        n = sum(pq.ParquetFile(path).metadata.num_rows for path in entries)
        lengths = []
        for path in entries:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["gsensor"]):
                lengths.extend(pc.list_value_length(batch.column(0)).to_pylist())
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        x = np.lib.format.open_memmap(output / f"{split}_x.npy", mode="w+", dtype="float32", shape=(int(offsets[-1]), 7))
        ys, ids, sizes, hd, labels, features, peaks, valid, rejected, eligible = [], [], [], [], [], [], [], [], [], []
        i = 0
        for path in entries:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch_rows(batch):
                    event_id = int(row["event_id"])
                    if row["label"] not in (0, 1, 2):
                        raise ValueError("Unknown label")
                    target = int(row["label"] == 0)
                    try:
                        values, target = decode(row, sensors)
                        usable = True
                    except ValueError as exc:
                        # Storage placeholder only; this event is never fed to a model.
                        values = np.zeros((int(lengths[i]), 7), np.float32)
                        usable = False
                        rejected.append({"event_id": event_id, "label": int(row["label"]), "reason": str(exc)})
                    signal_hash = hashlib.sha256(values.tobytes()).digest()
                    eligible.append(event_id not in excluded)
                    usable = usable and eligible[-1]
                    if event_id in seen_ids or (usable and signal_hash in seen_signals):
                        raise ValueError(f"Duplicate event or exact signal: {event_id}; review before training")
                    seen_ids.add(event_id)
                    if usable:
                        seen_signals.add(signal_hash)
                    x[offsets[i]:offsets[i+1]] = values
                    ys.append(target)
                    ids.append(event_id)
                    labels.append(int(row["label"]))
                    sizes.append(str(row["vehiclesize"]))
                    hd.append(bool(row["gyro_is_hd"]))
                    valid.append(usable)
                    features.append(feature_vector(values) if usable else np.zeros(203, dtype=np.float32))
                    peaks.append(float(np.linalg.norm(values[:, :2], axis=1).max() / 9.80665) if usable else np.nan)
                    if split == "train" and usable:
                        sums += values.sum(axis=0, dtype=np.float64)
                        squares += np.square(values.astype(np.float64)).sum(axis=0)
                        count += len(values)
                    i += 1
            print(f"Prepared {split}: {i}/{n}", flush=True)
        x.flush()
        del x
        if len(set(ys)) != 2:
            raise ValueError(f"Both crash and non-crash classes required in {split}")
        if len(set(np.array(ys)[np.array(valid)])) != 2:
            raise ValueError(f"Both usable classes required in {split}")
        np.savez_compressed(output / f"{split}_meta.npz", y=np.array(ys), event_id=np.array(ids),
                            vehiclesize=np.array(sizes), gyro_is_hd=np.array(hd), provider_label=np.array(labels),
                            features=np.array(features), physical_score=np.array(peaks), offsets=offsets,
                            valid=np.array(valid), eligible=np.array(eligible))
        catalog["splits"][split] = {"events": n, "crashes": sum(ys), "noncrashes": n - sum(ys)}
        catalog.setdefault("rejected", {})[split] = rejected
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-8))
    catalog["scaler"] = {"mean": mean.tolist(), "std": std.tolist(), "fitted_on": "train only"}
    catalog["files"] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "catalog.json", catalog)
    return catalog


def load_catalog(path, verify=True):
    path = Path(path)
    catalog = json.loads((path / "catalog.json").read_text(encoding="utf-8"))
    if verify:
        for name, expected in catalog["files"].items():
            if digest(path / name) != expected:
                raise ValueError(f"Prepared file changed: {name}")
    return catalog


def sensor_indices(sensors):
    return {"accel": list(range(3)), "imu": list(range(6)), "all": list(range(7))}[sensors]
