"""Predict one complete, already aligned recording in documented SI units."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crashdetect.vzcrash.experiment import predict_clip
from crashdetect.vzcrash.csv_input import read_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/hist-gradient-boosting")
    parser.add_argument("--family", choices=["physical", "trees", "decision_tree", "random_forest", "cnn_gru", "cnn_transformer", "inception_time", "xgboost", "catboost"])
    args = parser.parse_args()
    try:
        clip = read_clip(args.input)
        result = predict_clip(args.model, clip, family=args.family)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Input/model error: {exc}\n")
    print(f"Read {len(clip)} samples and 7 sensor channels.")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
