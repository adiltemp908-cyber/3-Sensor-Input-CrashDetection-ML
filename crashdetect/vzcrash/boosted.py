"""Optional boosted-tree alternatives on the existing numerical features."""
import json
from pathlib import Path
import time

import numpy as np


EXTENSIONS = {"xgboost": ".ubj", "catboost": ".cbm"}
DEPTHS = (4, 6, 8)


def create_classifier(family, depth, iterations, seed, device, threads):
    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=iterations, max_depth=depth, learning_rate=.05,
            reg_lambda=3., reg_alpha=0., min_child_weight=1,
            subsample=.9, colsample_bytree=.9, max_bin=256,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            device=device, n_jobs=threads, random_state=seed)
    if family == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=iterations, depth=depth, learning_rate=.05,
            l2_leaf_reg=3., loss_function="Logloss", boosting_type="Plain",
            bootstrap_type="Bernoulli", subsample=.9, border_count=128,
            random_seed=seed, thread_count=threads, task_type="GPU" if device == "cuda" else "CPU",
            **({"devices": "0", "gpu_ram_part": .7} if device == "cuda" else {}),
            use_best_model=False, allow_writing_files=False, verbose=False)
    raise ValueError(f"Unknown boosted family: {family}")


def load_classifier(directory, family, threads=4):
    if family == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(Path(directory) / (family + EXTENSIONS[family]))
        model.set_params(device="cpu", n_jobs=threads)
    elif family == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(thread_count=threads)
        model.load_model(str(Path(directory) / (family + EXTENSIONS[family])))
    else:
        raise ValueError(f"Unknown boosted family: {family}")
    return model


def probabilities(model, family, features, threads=4):
    features = np.ascontiguousarray(features, dtype=np.float32)
    if family == "catboost":
        return model.predict_proba(features, thread_count=threads, task_type="CPU")[:, 1]
    return model.predict_proba(features)[:, 1]


def train_boosted(family, train, val, indices, feature_columns, args):
    from .experiment import metrics, threshold_for_recall
    from .data import write_json

    training = np.ascontiguousarray(train["features"][indices][:, feature_columns], dtype=np.float32)
    valid = val.get("valid", np.ones(len(val["y"]), bool))
    validation = np.ascontiguousarray(val["features"][valid][:, feature_columns], dtype=np.float32)
    history, best_key, best_score = [], None, None
    output = Path(args.output)
    for depth in DEPTHS:
        started = time.perf_counter()
        print(json.dumps({"model": family, "phase": "training", "depth": depth,
                          "iterations": args.boost_rounds, "training_events": len(indices)}), flush=True)
        model = create_classifier(family, depth, args.boost_rounds, args.seed, args.device, args.threads)
        model.fit(training, train["y"][indices])
        if family == "xgboost":
            # Freeze inference on CPU, avoiding GPU/host-array fallback differences.
            model.set_params(device="cpu")
        score = np.full(len(val["y"]), np.nan)
        score[valid] = probabilities(model, family, validation, args.threads)
        threshold = threshold_for_recall(val["y"], score, args.target_recall, val.get("eligible"))
        result = metrics(val["y"], score, threshold, val.get("eligible"))
        record = {"depth": depth, "iterations": args.boost_rounds,
                  "validation": result, "seconds": time.perf_counter() - started}
        history.append(record)
        key = (result["recall"] >= args.target_recall, result["precision"], result["average_precision"])
        if best_key is None or key > best_key:
            best_key, best_score = key, score.copy()
            model.save_model(str(output / (family + EXTENSIONS[family])))
        write_json(output / f"{family}_history.json", history)
        print(json.dumps({"model": family, "depth": depth, "seconds": record["seconds"],
                          "validation_ap": result["average_precision"],
                          "validation_precision": result["precision"], "validation_recall": result["recall"]}), flush=True)
        del model
    return best_score, history
