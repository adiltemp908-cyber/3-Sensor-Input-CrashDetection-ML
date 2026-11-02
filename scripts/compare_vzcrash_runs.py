"""Compare completed runs on identical populations, selecting by validation only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crashdetect.vzcrash.experiment import verify_model
from crashdetect.vzcrash.data import write_json


LABELS = {"physical": "Peak acceleration", "trees": "Boosted trees",
          "decision_tree": "Decision tree", "random_forest": "Random forest",
          "cnn_gru": "CNN-GRU", "cnn_transformer": "CNN-Transformer",
          "inception_time": "InceptionTime (single network)", "xgboost": "XGBoost", "catboost": "CatBoost"}


def compare_runs(runs):
    candidates, populations = {}, {}
    reference, training_ids = None, None
    for run in map(Path, runs):
        config = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        comparison = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
        identity = {key: config[key] for key in (
            "catalog_sha256", "source_revision", "columns", "scaler", "target_recall", "training_events")}
        ids = np.sort(np.load(run / "training_event_ids.npy", allow_pickle=False))
        if reference is None:
            reference, training_ids = identity, ids
        elif identity != reference or not np.array_equal(ids, training_ids):
            raise ValueError("Runs must use identical prepared data, sensors, training events, and recall target")
        for family, result in comparison.items():
            verify_model(run, family, config)
            for split in ("validation", "test"):
                with np.load(run / f"{family}_{split}_scores.npz", allow_pickle=False) as scores:
                    population = {key: scores[key] for key in ("event_id", "y", "eligible")}
                    population["usable"] = np.isfinite(scores["score"])
                if split in populations:
                    if any(not np.array_equal(populations[split][key], value) for key, value in population.items()):
                        raise ValueError(f"{split} event populations or availability differ")
                else:
                    populations[split] = population
            if family in candidates:
                # The physical baseline is shared. Never pick a repeated trained family using test results.
                if family != "physical" or result != candidates[family]["metrics"]:
                    raise ValueError(f"Ambiguous repeated candidate: {family}")
                continue
            candidates[family] = {"run": str(run.resolve()), "metrics": result}
    if not candidates:
        raise ValueError("At least one completed run is required")
    selected = max(candidates, key=lambda family: (
        candidates[family]["metrics"]["validation"]["recall"] >= reference["target_recall"],
        candidates[family]["metrics"]["validation"]["precision"],
        candidates[family]["metrics"]["validation"]["average_precision"]))
    return {"selected": selected, "selection_basis": "Validation precision at target recall, then validation AP",
            "common_configuration": reference, "candidates": candidates,
            "evaluation_caveat": "The test population was previously examined for the tree baseline; this is an exploratory comparison, not fresh independent validation."}


def render_report(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", result)
    selected = result["selected"]
    lines = ["# VZCrash model alternatives", "",
             f"Selected by validation: **{LABELS[selected]}**. {result['selection_basis']}.", "",
             "| Model | Validation AP | Validation precision | Test AP | Test recall | Test precision | Test FPR | Misses | False positives |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for family, candidate in result["candidates"].items():
        v, t = candidate["metrics"]["validation"], candidate["metrics"]["test"]
        lines.append(f"| {LABELS[family]} | {v['average_precision']:.2%} | {v['precision']:.2%} | "
                     f"{t['average_precision']:.2%} | {t['recall']:.2%} | {t['precision']:.2%} | "
                     f"{t['false_positive_rate']:.2%} | {t['fn']} | {t['fp']} |")
        with np.load(Path(candidate["run"]) / f"{family}_test_scores.npz", allow_pickle=False) as scores:
            usable = scores["eligible"] & np.isfinite(scores["score"])
            precision, recall, _ = precision_recall_curve(scores["y"][usable], scores["score"][usable])
        curve, = axes[0].plot(recall, precision, linewidth=2,
                             label=f"{LABELS[family]}: AP {t['average_precision']:.2%}")
        axes[0].scatter(t["recall"], t["precision"], color=curve.get_color(), s=35, zorder=4)
        history = candidate["metrics"]["history"]
        if history and "epoch" in history[0]:
            axes[1].plot([h["epoch"] for h in history], [h["validation_ap"] for h in history],
                         marker="o", color=curve.get_color(), label=LABELS[family])
    axes[0].set(xlim=(.8, 1.001), ylim=(.5, 1.005), xlabel="Crash recall",
                ylabel="Precision", title="Test precision–recall (high-recall region)")
    axes[1].set(xlabel="Epoch", ylabel="Validation average precision", title="Neural learning curves")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(loc="lower left", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle("VZCrash · Model comparison", fontsize=16, fontweight="bold")
    count = result["common_configuration"]["training_events"]
    fig.supxlabel(f"{count:,} training events · same audited splits and sensors · thresholds frozen on validation\n"
                  "Exploratory event-clip comparison; continuous-driving false alarms and detection delay unmeasured.", fontsize=9)
    fig.savefig(output / "comparison.png", dpi=170)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    # Direct operating-point comparison complements the ranking curves, especially
    # for coarse decision-tree scores whose thresholds can overshoot target recall.
    families = list(result["candidates"])
    labels = [LABELS[f] for f in families]
    test_metrics = [result["candidates"][f]["metrics"]["test"] for f in families]
    colors = ["#b86c19" if f in ("decision_tree", "random_forest") else
              "#777777" if f == "physical" else "#087f8c" for f in families]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, .55 * len(families) + 1.5)),
                             layout="constrained", sharey=True)
    positions = np.arange(len(families))
    bars = axes[0].barh(positions, [100*t["precision"] for t in test_metrics], color=colors)
    axes[0].bar_label(bars, labels=[f"{t['precision']:.2%}" for t in test_metrics], padding=4, fontsize=9)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 110), xlabel="Test precision (%)", title="Precision at frozen thresholds")
    axes[0].set_xticks([0, 25, 50, 75, 100])
    bars = axes[1].barh(positions, [t["fp"] for t in test_metrics], color=colors)
    axes[1].bar_label(bars, labels=[f"{t['fp']:,}  |  recall {t['recall']:.2%}" for t in test_metrics],
                     padding=4, fontsize=9)
    axes[1].set(xlim=(0, max(t["fp"] for t in test_metrics)*1.65),
                xlabel="False-positive event clips", title="False positives and corresponding recall")
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("VZCrash · Test operating points", fontsize=16, fontweight="bold")
    fig.supxlabel("Thresholds target at least 95% recall on validation; test recall can differ.\n"
                  "Orange: decision-tree and random-forest baselines. Event clips; false alerts/hour unmeasured.", fontsize=9)
    fig.savefig(output / "operating_points.png", dpi=170)
    fig.savefig(output / "operating_points.svg")
    plt.close(fig)
    lines += ["", f"All candidates use {count:,} training recordings and the same audited held-out populations.", "",
              result["evaluation_caveat"], "",
              "AP and precision describe selected event clips. Scores are not calibrated deployment probabilities. "
              "Continuous-driving false alarms/hour and impact-to-alert delay remain unmeasured.", "",
              "![Comparison](comparison.png)", "", "![Test operating points](operating_points.png)",
              "", "## Neural training", ""]
    for family, candidate in result["candidates"].items():
        if candidate["metrics"]["history"] and "epoch" in candidate["metrics"]["history"][0]:
            history = candidate["metrics"]["history"]
            best = max(history, key=lambda record: record["validation_ap"])
            lines.append(f"- {LABELS[family]}: {len(history)} epochs; best epoch {best['epoch']}; "
                         f"recorded epoch wall time {sum(h['seconds'] for h in history)/60:.1f} minutes (includes pauses). "
                         f"Artifacts: `{candidate['run']}`.")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_runs(args.runs)
    render_report(result, args.output)
    print(json.dumps({"selected": result["selected"], "report": str(args.output / "report.md")}))


if __name__ == "__main__":
    main()
