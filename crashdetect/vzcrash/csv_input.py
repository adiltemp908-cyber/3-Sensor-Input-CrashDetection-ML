"""Read an already aligned CSV clip; do not infer units or fabricate sensors."""
import numpy as np

COLUMNS = ("time_s", "ax", "ay", "az", "gx", "gy", "gz", "speed")


def read_clip(path):
    rows = np.genfromtxt(path, delimiter=",", names=True,
                         encoding="utf-8-sig", ndmin=1)
    if rows.dtype.names != COLUMNS:
        raise ValueError("CSV header must be exactly: " + ",".join(COLUMNS))
    if not 32 <= len(rows) <= 20000:
        raise ValueError("Expected 32..20000 samples; a 16-second recording has 1600.")
    times = rows["time_s"]
    clip = np.column_stack([rows[name] for name in COLUMNS[1:]]).astype(np.float32)
    if not np.isfinite(times).all() or not np.isfinite(clip).all():
        raise ValueError("Every cell must contain a finite number; missing values are not filled.")
    if not np.allclose(np.diff(times), .01, rtol=0, atol=.0001):
        raise ValueError("Rows must be ordered at 100 Hz: timestamps increase by 0.01 seconds.")
    if (clip[:, 6] < 0).any():
        raise ValueError("GPS speed must be nonnegative and expressed in m/s.")
    return clip
