"""Decision-tree and random-forest baselines on the existing clip features."""
import json
import pickle
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def candidate_parameters(family, seed=42, threads=4, forest_trees=300):
    common = {"criterion": "gini", "random_state": seed, "class_weight": None}
    if family == "decision_tree":
        return [{**common, "max_depth": depth, "min_samples_leaf": 20}
                for depth in (8, 16, None)]
    if family == "random_forest":
        if forest_trees < 1:
            raise ValueError("Forest tree count must be positive")
        return [{**common, "n_estimators": forest_trees, "n_jobs": threads,
                 "bootstrap": True, "max_features": "sqrt", "max_depth": depth,
                 "min_samples_leaf": leaf}
                for depth, leaf in ((16, 5), (None, 5), (None, 1))]
    raise ValueError(f"Unknown classical family: {family}")


def train_classical(family, train, val, indices, feature_columns, args):
    from .experiment import metrics, threshold_for_recall
    from .data import write_json

    training = np.ascontiguousarray(train["features"][indices][:, feature_columns], dtype=np.float32)
    valid = val.get("valid", np.ones(len(val["y"]), bool))
    validation = np.ascontiguousarray(val["features"][valid][:, feature_columns], dtype=np.float32)
    history, best_key, best_score = [], None, None
    output = Path(args.output)
    candidates = candidate_parameters(family, args.seed, args.threads, getattr(args, "forest_trees", 300))
    # Persist the complete search space before fitting; no test data enters this function.
    write_json(output / f"{family}_plan.json", candidates)
    constructor = DecisionTreeClassifier if family == "decision_tree" else RandomForestClassifier
    for parameters in candidates:
        started = time.perf_counter()
        print(json.dumps({"model": family, "phase": "training", "parameters": parameters,
                          "training_events": len(indices)}), flush=True)
        model = constructor(**parameters)
        model.fit(training, train["y"][indices])
        score = np.full(len(val["y"]), np.nan)
        score[valid] = model.predict_proba(validation)[:, 1]
        threshold = threshold_for_recall(val["y"], score, args.target_recall, val.get("eligible"))
        result = metrics(val["y"], score, threshold, val.get("eligible"))
        record = {"parameters": model.get_params(), "validation": result,
                  "seconds": time.perf_counter() - started}
        history.append(record)
        key = (result["recall"] >= args.target_recall, result["precision"], result["average_precision"])
        if best_key is None or key > best_key:
            best_key, best_score = key, score.copy()
            with (output / f"{family}.pkl").open("wb") as stream:
                pickle.dump(model, stream)
        write_json(output / f"{family}_history.json", history)
        print(json.dumps({"model": family, "parameters": parameters, "seconds": record["seconds"],
                          "validation_ap": result["average_precision"],
                          "validation_precision": result["precision"], "validation_recall": result["recall"]}), flush=True)
        del model
    return best_score, history
