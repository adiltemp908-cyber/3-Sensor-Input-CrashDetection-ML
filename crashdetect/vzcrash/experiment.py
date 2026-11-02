"""Select on validation, freeze the policy, then measure the held-out event clips."""
from pathlib import Path
import json
import pickle
import random
import time
from importlib.metadata import version

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader, Dataset

from .data import digest, feature_vector, load_catalog, sensor_indices, write_json
from .model import make_model


def threshold_for_recall(y, score, recall_target=.95, eligible=None):
    if not 0 < recall_target <= 1:
        raise ValueError("Recall target must be in (0, 1]")
    y, score = np.asarray(y), np.asarray(score)
    if eligible is not None:
        y, score = y[eligible], score[eligible]
    valid = np.isfinite(score)
    positive_total = int((y == 1).sum())
    positive_valid = int(((y == 1) & valid).sum())
    if not positive_valid:
        raise ValueError("No usable validation crashes")
    recall_target = min(1., recall_target * positive_total / positive_valid)
    y, score = y[valid], score[valid]
    if len(set(y.tolist())) != 2:
        raise ValueError("Threshold selection needs both usable classes")
    p, r, thresholds = precision_recall_curve(y, score)
    candidates = np.flatnonzero(r[:-1] >= recall_target)
    # Maximize precision subject to recall, then prefer the higher threshold.
    best = max(candidates, key=lambda i: (p[i], thresholds[i]))
    return float(thresholds[best])


def wilson(successes, n):
    if n == 0:
        return None
    z = 1.95996398454
    p = successes / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    radius = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def metrics(y, score, threshold, eligible=None):
    y, score = np.asarray(y), np.asarray(score)
    original_events = len(y)
    if eligible is not None:
        y, score = y[eligible], score[eligible]
    valid = np.isfinite(score)
    pred = valid & (score >= threshold)
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    fn, tn = int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred & valid).sum())
    recall = tp/(tp+fn) if tp+fn else None
    fpr = fp/(fp+tn) if fp+tn else None
    prior = .0002  # Illustration only, not an estimate for this user's deployment.
    denominator = (prior*recall + (1-prior)*fpr) if recall is not None and fpr is not None else 0
    both_valid = len(set(y[valid].tolist())) == 2
    return {"events": len(y), "original_events": original_events,
            "excluded_duplicate_events": original_events - len(y), "crashes": tp+fn, "noncrashes": int((y == 0).sum()),
            "unavailable_events": int((~valid).sum()), "unavailable_crashes": int(((y == 1) & ~valid).sum()),
            "availability": float(valid.mean()), "ranking_metrics_conditional_on_usable_data": True,
            "average_precision": float(average_precision_score(y[valid], score[valid])) if both_valid else None,
            "roc_auc": float(roc_auc_score(y[valid], score[valid])) if both_valid else None,
            "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else None, "recall": recall, "false_positive_rate": fpr,
            "recall_95pct_wilson": wilson(tp, tp+fn), "fpr_95pct_wilson": wilson(fp, fp+tn),
            "ci_caveat": "Event-binomial intervals; vehicle correlations cannot be accounted for without vehicle IDs.",
            "illustrative_precision_at_crash_prior_0_0002": prior*recall/denominator if denominator else None,
            "prior_caveat": "Assumes unchanged sensitivity/specificity; not a deployment measurement.",
            "false_alerts_per_hour": None, "impact_to_alert_delay_s": None}


class Clips(Dataset):
    def __init__(self, path, split, columns, mean, std, indices=None):
        self.x = np.load(Path(path) / f"{split}_x.npy", mmap_mode="r")
        meta = np.load(Path(path) / f"{split}_meta.npz", allow_pickle=False)
        self.y = meta["y"]
        self.offsets = meta["offsets"] if "offsets" in meta else None
        self.valid = meta["valid"] if "valid" in meta else np.ones(len(self.y), bool)
        self.columns, self.mean, self.std = columns, np.array(mean)[columns], np.array(std)[columns]
        self.indices = np.flatnonzero(self.valid) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        raw = self.x[index] if self.offsets is None else self.x[self.offsets[index]:self.offsets[index+1]]
        x = (raw[:, self.columns] - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(self.y[index], dtype=torch.float32)


def collate_clips(batch):
    x, y = zip(*batch)
    lengths = torch.tensor([len(v) for v in x], dtype=torch.long)
    return torch.nn.utils.rnn.pad_sequence(x, batch_first=True), torch.stack(y), lengths


def score_neural(net, loader, device):
    net.eval()
    result = []
    with torch.inference_mode():
        for x, _, lengths in loader:
            result.append(net(x.to(device), lengths.to(device)).sigmoid().cpu().numpy())
    score = np.full(len(loader.dataset.y), np.nan)
    score[loader.dataset.indices] = np.concatenate(result)
    return score


def train_neural(path, family, columns, catalog, indices, args):
    scaler = catalog["scaler"]
    training = Clips(path, "train", columns, **{k: scaler[k] for k in ("mean", "std")}, indices=indices)
    validation = Clips(path, "validation", columns, **{k: scaler[k] for k in ("mean", "std")})
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_clips)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0, collate_fn=collate_clips)
    net = make_model(len(columns), family).to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=.01)
    amp = getattr(args, "amp", False)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # Natural prevalence BCE avoids an additional artificial prior shift from oversampling.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_ap, best_state, stale, history = -1., None, 0, []
    for epoch in range(args.epochs):
        start = time.perf_counter()
        last_progress, seen = start, 0
        net.train()
        loss_sum = 0.
        for x, y, lengths in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=amp):
                loss = loss_fn(net(x, lengths.to(args.device)), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            loss_sum += float(loss.detach()) * len(x)
            seen += len(x)
            now = time.perf_counter()
            if now - last_progress >= 30:
                print(json.dumps({"model": family, "epoch": epoch+1, "phase": "training",
                                  "events_seen": seen, "training_events": len(training),
                                  "loss": loss_sum/seen, "seconds": now-start}), flush=True)
                last_progress = now
        score = score_neural(net, val_loader, args.device)
        ap = float(average_precision_score(validation.y[validation.valid], score[validation.valid]))
        record = {"epoch": epoch+1, "loss": loss_sum/len(training), "validation_ap": ap, "seconds": time.perf_counter()-start}
        history.append(record)
        print(json.dumps({"model": family, **record}), flush=True)
        if ap > best_ap:
            best_ap, stale = ap, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save(best_state, Path(args.output) / f"{family}.pt")
        else:
            stale += 1
        write_json(Path(args.output) / f"{family}_history.json", history)
        if stale >= args.patience:
            break
    net.load_state_dict(best_state)
    return net, score_neural(net, val_loader, args.device), history


def benchmark(args):
    if getattr(args, "amp", False) and args.device != "cuda":
        raise ValueError("AMP requires --device cuda")
    if args.device == "cuda" and any(f in ("cnn_gru", "cnn_transformer", "inception_time") for f in args.models) and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    if getattr(args, "boost_rounds", 1000) < 1:
        raise ValueError("Boosting rounds must be positive")
    if getattr(args, "forest_trees", 300) < 1:
        raise ValueError("Forest tree count must be positive")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.threads < 1:
        raise ValueError("Epochs, batch size, patience, and threads must be positive")
    if not 0 < args.target_recall <= 1 or args.learning_rate <= 0:
        raise ValueError("Invalid recall or learning rate")
    output, path = Path(args.output), Path(args.prepared)
    if output.exists():
        raise ValueError("Choose a new benchmark output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    catalog = load_catalog(path)
    columns = sensor_indices(args.sensors)
    if not set(columns) <= set(sensor_indices(catalog.get("available_sensors", "all"))):
        raise ValueError("Requested sensors were not included during preparation")
    train = dict(np.load(path / "train_meta.npz", allow_pickle=False))
    val = dict(np.load(path / "validation_meta.npz", allow_pickle=False))
    indices = np.flatnonzero(train.get("valid", np.ones(len(train["y"]), bool)))
    if args.max_train:
        if args.max_train < 2:
            raise ValueError("max-train must be at least 2")
        # A training-only uniform subset; never select easy examples or alter held-out sets.
        indices = np.sort(np.random.default_rng(args.seed).choice(indices, min(args.max_train, len(indices)), replace=False))
    if len(set(train["y"][indices])) != 2:
        raise ValueError("Training subset must contain both classes")
    output.mkdir(parents=True)
    comparison = {}
    feature_columns = [i for i in range(train["features"].shape[1]) if i % 7 in columns]
    families = list(dict.fromkeys(["physical"] + args.models))
    for family in families:
        history = []
        if family == "physical":
            score, model = val["physical_score"], None
        elif family == "trees":
            from threadpoolctl import threadpool_limits
            best_key = None
            with threadpool_limits(limits=args.threads):
                valid = val.get("valid", np.ones(len(val["y"]), bool))
                for leaves, iterations in ((15, 250), (31, 500), (31, 1000)):
                    trial = HistGradientBoostingClassifier(max_iter=iterations, max_leaf_nodes=leaves, l2_regularization=1.,
                                                           learning_rate=.06, early_stopping=False, random_state=args.seed)
                    trial.fit(train["features"][indices][:, feature_columns], train["y"][indices])
                    trial_score = np.full(len(val["y"]), np.nan)
                    trial_score[valid] = trial.predict_proba(val["features"][valid][:, feature_columns])[:, 1]
                    trial_threshold = threshold_for_recall(val["y"], trial_score, args.target_recall, val.get("eligible"))
                    trial_result = metrics(val["y"], trial_score, trial_threshold, val.get("eligible"))
                    key = (trial_result["recall"] >= args.target_recall, trial_result["precision"], trial_result["average_precision"])
                    history.append({"leaves": leaves, "iterations": iterations, "validation": trial_result})
                    print(json.dumps({"model": "trees", "leaves": leaves, "iterations": iterations,
                                      "validation_ap": trial_result["average_precision"], "validation_precision": trial_result["precision"]}), flush=True)
                    if best_key is None or key > best_key:
                        model, score, best_key = trial, trial_score, key
            with (output / f"{family}.pkl").open("wb") as stream:
                pickle.dump(model, stream)
        elif family in ("decision_tree", "random_forest"):
            from .classical import train_classical
            score, history = train_classical(family, train, val, indices, feature_columns, args)
        elif family in ("xgboost", "catboost"):
            from .boosted import train_boosted
            score, history = train_boosted(family, train, val, indices, feature_columns, args)
        else:
            torch.manual_seed(args.seed)
            model, score, history = train_neural(path, family, columns, catalog, indices, args)
            torch.save(model.cpu().state_dict(), output / f"{family}.pt")
        threshold = threshold_for_recall(val["y"], score, args.target_recall, val.get("eligible"))
        comparison[family] = {"validation": metrics(val["y"], score, threshold, val.get("eligible")), "history": history}
        comparison[family]["validation"]["recall_target_met"] = comparison[family]["validation"]["recall"] >= args.target_recall
        np.savez_compressed(output / f"{family}_validation_scores.npz", score=score, y=val["y"], event_id=val["event_id"],
                            eligible=val.get("eligible", np.ones(len(score), bool)))
        print(json.dumps({"model": family, "validation": comparison[family]["validation"]}), flush=True)
    # Test scores do not participate in model, epoch, or threshold selection.
    selected = max(families, key=lambda name: (comparison[name]["validation"]["recall_target_met"], comparison[name]["validation"]["precision"],
                                               comparison[name]["validation"]["average_precision"]))
    config = {"selected": selected, "sensors": args.sensors, "columns": columns, "feature_columns": feature_columns,
              "scaler": catalog["scaler"], "threshold": comparison[selected]["validation"]["threshold"],
              "sample_hz": 100, "samples": "variable", "channels": catalog["channels"],
              "source_revision": catalog["source"]["revision"], "catalog_sha256": digest(path / "catalog.json"),
              "training_events": len(indices), "seed": args.seed, "target_recall": args.target_recall,
              "task": catalog["task"], "deployment_validated": False,
              "arguments": vars(args), "versions": {name: version(name) for name in
                  ("torch", "numpy", "scikit-learn", "pyarrow", "huggingface-hub")}}
    for family in ("xgboost", "catboost"):
        if family in families:
            config["versions"][family] = version(family)
    config["model_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.suffix in (".pt", ".pkl", ".ubj", ".cbm")}
    write_json(output / "selection.json", config)
    np.save(output / "training_event_ids.npy", train["event_id"][indices])
    # Freeze every candidate and its validation threshold before touching test outcomes.
    test = dict(np.load(path / "test_meta.npz", allow_pickle=False))
    for family in families:
        score = predict_prepared(output, path, "test", family, config, args.batch_size, args.device)
        threshold = comparison[family]["validation"]["threshold"]
        report = metrics(test["y"], score, threshold, test.get("eligible"))
        report["recall_target_met"] = report["recall"] >= args.target_recall
        cohorts = {}
        for field in ("vehiclesize", "gyro_is_hd", "provider_label"):
            for value in np.unique(test[field]):
                mask = test[field] == value
                eligible = test["eligible"][mask] if "eligible" in test else None
                cohorts[f"{field}={value}"] = metrics(test["y"][mask], score[mask], threshold, eligible)
        comparison[family].update(test=report, test_cohorts=cohorts)
        np.savez_compressed(output / f"{family}_test_scores.npz", score=score, y=test["y"], event_id=test["event_id"],
                            eligible=test.get("eligible", np.ones(len(score), bool)))
    write_json(output / "comparison.json", comparison)
    lines = ["# VZCrash event-clip benchmark", "", f"Selected using validation only: **{selected}**.", "",
             "| Model | Validation AP | Test AP | Test recall | Test precision | Test FPR |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in comparison.items():
        v, t = result["validation"], result["test"]
        lines.append(f"| {name} | {v['average_precision']:.4f} | {t['average_precision']:.4f} | {t['recall']:.4f} | {t['precision']:.4f} | {t['false_positive_rate']:.6f} |")
    lines += ["", "Unavailable recordings are excluded from training. Unavailable test crashes count as misses in recall; AP/AUC use usable recordings only.",
              f"Exact-duplicate group members excluded from test evaluation: {comparison[selected]['test']['excluded_duplicate_events']}.",
              f"Unavailable test events: {comparison[selected]['test']['unavailable_events']}; unavailable crashes: {comparison[selected]['test']['unavailable_crashes']}.",
              "These are selected variable-length event clips, not continuous driving. False alarms/hour and impact-to-alert delay are unmeasured.",
              "Scores are not calibrated real-world probabilities. Provider crash labels also include accidental roadway departure.",
              "Original partition assignments are retained, with documented duplicate exclusions. Vehicle IDs are not released; exact cross-partition duplicates were found and excluded.",
              f"Training events used: {len(indices)}. Sensors: {args.sensors}. Source: https://huggingface.co/datasets/{catalog['source']['repo']} .",
              "CC BY-NC 4.0 data. Research prototype; passenger-car and target-hardware performance require separate validation."]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"selected": selected, "test": comparison[selected]["test"]}


def predict_prepared(output, prepared, split, family, config, batch_size=64, device="cpu"):
    verify_model(output, family, config)
    meta = dict(np.load(Path(prepared) / f"{split}_meta.npz", allow_pickle=False))
    if family == "physical":
        return meta["physical_score"]
    if family in ("xgboost", "catboost"):
        from .boosted import load_classifier, probabilities
        model = load_classifier(output, family)
        valid = meta.get("valid", np.ones(len(meta["y"]), bool))
        score = np.full(len(meta["y"]), np.nan)
        score[valid] = probabilities(model, family, meta["features"][valid][:, config["feature_columns"]])
        return score
    if family in ("trees", "decision_tree", "random_forest"):
        with (Path(output) / f"{family}.pkl").open("rb") as stream:
            model = pickle.load(stream)  # Only locally trained/trusted artifacts.
        valid = meta.get("valid", np.ones(len(meta["y"]), bool))
        score = np.full(len(meta["y"]), np.nan)
        score[valid] = model.predict_proba(meta["features"][valid][:, config["feature_columns"]])[:, 1]
        return score
    net = make_model(len(config["columns"]), family).to(device)
    net.load_state_dict(torch.load(Path(output) / f"{family}.pt", map_location=device, weights_only=True))
    ds = Clips(prepared, split, config["columns"], config["scaler"]["mean"], config["scaler"]["std"])
    return score_neural(net, DataLoader(ds, batch_size=batch_size, collate_fn=collate_clips), device)


def verify_model(path, family, config):
    if family == "physical":
        return
    name = family + {"trees": ".pkl", "decision_tree": ".pkl", "random_forest": ".pkl",
                     "xgboost": ".ubj", "catboost": ".cbm"}.get(family, ".pt")
    if digest(Path(path) / name) != config["model_sha256"][name]:
        raise ValueError("Model changed after validation threshold was frozen")


def predict_clip(model_dir, x, family=None):
    """One complete clip in canonical SI units and vehicle coordinates; no live alert claim."""
    path = Path(model_dir)
    config = json.loads((path / "selection.json").read_text(encoding="utf-8"))
    if family is not None:
        comparison = json.loads((path / "comparison.json").read_text(encoding="utf-8"))
        if family not in comparison:
            raise ValueError(f"Model family {family!r} is not in this benchmark")
        config["selected"] = family
        config["threshold"] = comparison[family]["validation"]["threshold"]
    x = np.asarray(x, dtype=np.float32)
    columns = config["columns"]
    if x.ndim == 2 and x.shape[1] == len(columns) and len(columns) != 7:
        # Internal padding only: discarded channels are never used by this trained model.
        supplied = x
        x = np.zeros((len(x), 7), dtype=np.float32)
        x[:, columns] = supplied
    if x.ndim != 2 or x.shape[1] != 7 or not 32 <= len(x) <= 20000 or not np.isfinite(x[:, columns]).all() or (6 in columns and (x[:, 6] < 0).any()):
        raise ValueError(f"Input must be a finite (T, {len(columns)}) selected-sensor or (T, 7) SI clip, T=32..20000; GPS speed must be nonnegative")
    family = config["selected"]
    verify_model(path, family, config)
    if family == "physical":
        score = float(np.linalg.norm(x[:, :2], axis=1).max() / 9.80665)
    elif family in ("xgboost", "catboost"):
        from .boosted import load_classifier, probabilities
        model = load_classifier(path, family)
        features = feature_vector(x)[config["feature_columns"]]
        score = float(probabilities(model, family, features[None])[0])
    elif family in ("trees", "decision_tree", "random_forest"):
        with (path / f"{family}.pkl").open("rb") as stream:
            model = pickle.load(stream)
        features = feature_vector(x)[config["feature_columns"]]
        score = float(model.predict_proba(features[None])[0, 1])
    else:
        net = make_model(len(columns), family)
        net.load_state_dict(torch.load(path / f"{family}.pt", map_location="cpu", weights_only=True))
        net.eval()
        scaled = ((x - np.array(config["scaler"]["mean"])) / np.array(config["scaler"]["std"]))[:, columns]
        with torch.inference_mode():
            score = float(net(torch.from_numpy(scaled.astype(np.float32))[None]).sigmoid()[0])
    return {"score": score, "threshold": config["threshold"], "crash_clip": score >= config["threshold"],
            "model": family, "deployment_validated": False, "task": "event-clip classification"}
