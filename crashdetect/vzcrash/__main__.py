import argparse
import json

import numpy as np

from .data import REVISION, download, prepare
from .experiment import benchmark, predict_clip


def main(argv=None):
    parser = argparse.ArgumentParser(description="VZCrash research workflow: official event clips, configurable sensor inputs")
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="Download after user-completed consent and local hf auth login")
    get.add_argument("--output", default="data/vzcrash/raw")
    get.add_argument("--revision", default=REVISION)
    prep = commands.add_parser("prepare", help="Validate and convert the complete official partitions")
    prep.add_argument("--raw", default="data/vzcrash/raw")
    prep.add_argument("--output", default="data/vzcrash/prepared")
    prep.add_argument("--sensors", choices=["accel", "imu", "all"], default="all")
    prep.add_argument("--exclude-duplicate-groups", action="store_true",
                      help="Exclude every member of exact-signal groups from training and evaluation; retain an audit")
    train = commands.add_parser("benchmark", help="Select on validation and evaluate frozen models on test")
    train.add_argument("--prepared", default="data/vzcrash/prepared")
    train.add_argument("--output", default="artifacts/vzcrash")
    train.add_argument("--models", nargs="+", choices=["physical", "trees", "decision_tree", "random_forest", "cnn_gru", "cnn_transformer", "inception_time", "xgboost", "catboost"],
                       default=["trees", "cnn_gru", "cnn_transformer"])
    train.add_argument("--sensors", choices=["accel", "imu", "all"], default="all")
    train.add_argument("--boost-rounds", type=int, default=1000, help="Trees per XGBoost/CatBoost candidate; depths 4, 6, 8")
    train.add_argument("--forest-trees", type=int, default=300, help="Trees per random-forest candidate")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=.0005)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--target-recall", type=float, default=.95)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--amp", action="store_true", help="CUDA mixed-precision training; evaluation remains float32")
    train.add_argument("--max-train", type=int, default=0, help="0 uses all training events; nonzero is explicitly a subset experiment")
    predict = commands.add_parser("predict", help="Classify a complete .npy clip in documented SI units")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--family", choices=["physical", "trees", "decision_tree", "random_forest", "cnn_gru", "cnn_transformer", "inception_time", "xgboost", "catboost"],
                         help="Use a specific benchmark candidate and its frozen validation threshold")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = download(args.output, args.revision)
        elif args.command == "prepare":
            catalog = prepare(args.raw, args.output, args.sensors, args.exclude_duplicate_groups)
            result = {"output": args.output, "splits": catalog["splits"],
                      "duplicate_exclusions": len(catalog["duplicate_exclusions"]),
                      "sensor_unavailable": {split: len(rows) for split, rows in catalog["rejected"].items()}}
        elif args.command == "benchmark":
            result = benchmark(args)
        else:
            result = predict_clip(args.model, np.load(args.input, allow_pickle=False), args.family)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"VZCRASH: {exc}\n")
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
