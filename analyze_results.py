"""Audit stored predictions and generate experiment evidence, without refitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_experiment import evaluate, sha256, summarize


LABELS = {"bayesian_clip_dino": "Bayesian CLIP + DINO",
          "bayesian_clip_only": "Bayesian CLIP only",
          "ridge_clip": "CLIP ridge probe", "zero_shot_clip": "Zero-shot CLIP"}
COLORS = {"bayesian_clip_dino": "#007f73", "bayesian_clip_only": "#63778f",
          "ridge_clip": "#c15745", "zero_shot_clip": "#292929"}


def audit(document, feature_dir):
    if document["status"] != "complete":
        raise ValueError("experiment is not complete")
    for filename, expected in document["source_sha256"].items():
        if sha256(filename) != expected:
            raise ValueError(f"source changed since the experiment: {filename}")
    for filename, expected in document["feature_sha256"].items():
        if sha256(feature_dir / filename) != expected:
            raise ValueError(f"feature cache changed: {filename}")
    train = torch.load(feature_dir / "trainval.pt", weights_only=True)
    test = torch.load(feature_dir / "test.pt", weights_only=True)
    labels = train["labels"]
    classes = len(document["settings"]["classes"])
    support_by_run, seen = {}, set()
    expected = {("zero_shot_clip", 0, None)}
    for method in ("bayesian_clip_dino", "bayesian_clip_only", "ridge_clip"):
        expected.update((method, shot, seed) for shot in document["settings"]["shots"]
                        for seed in document["settings"]["seeds"])
    for row in document["runs"]:
        identifier = row["method"], row["shots"], row["seed"]
        if identifier in seen:
            raise ValueError("duplicate evaluation")
        seen.add(identifier)
        if sha256(row["prediction_file"]) != row["prediction_sha256"]:
            raise ValueError("prediction file hash mismatch")
        cache = torch.load(row["prediction_file"], weights_only=True)
        if not torch.equal(cache["labels"], test["labels"]):
            raise ValueError("test label order mismatch")
        variance = cache["variance"].double() if cache["variance"].numel() else None
        metrics = evaluate(cache["probabilities"].double(), cache["labels"], variance)
        for key, value in metrics.items():
            if value is not None and abs(value - row[key]) > 1e-5:
                raise ValueError(f"metric mismatch: {identifier} {key}")
        if row["shots"]:
            support = row["support_indices"]
            fit, val = row["tuning_fit_indices"], row["tuning_validation_indices"]
            if len(set(support)) != classes * row["shots"]:
                raise ValueError("incorrect support budget or duplicated examples")
            if any(index < 0 or index >= len(labels) for index in support):
                raise ValueError("support index outside trainval")
            if torch.bincount(labels[support], minlength=classes).tolist() != [row["shots"]] * classes:
                raise ValueError("support is not class-balanced")
            if set(fit) & set(val) or set(fit) | set(val) != set(support):
                raise ValueError("invalid tuning partition")
            key = row["shots"], row["seed"]
            sample = (support, fit, val)
            if key in support_by_run and support_by_run[key] != sample:
                raise ValueError("methods did not share identical labeled data")
            support_by_run[key] = sample
        if row["method"].startswith("bayesian"):
            hp = row["hyperparameters"]
            required = "marginal_likelihood" if row["shots"] == 1 else "predictive_likelihood"
            if hp["objective"] != required or len(hp["loss_history"]) != document["settings"]["optimization_steps"] + 1:
                raise ValueError("wrong GP optimization objective or steps")
    if seen != expected:
        raise ValueError("missing or unexpected experiment runs")
    if summarize(document["runs"]) != document["summary"]:
        raise ValueError("stored aggregate metrics differ from per-run metrics")
    return {"status": "passed", "evaluations": len(seen),
            "checks": ["source and feature SHA-256", "prediction SHA-256 and label order",
                       "metrics recomputed from stored float32 predictions (tolerance 1e-5)",
                       "complete method/shot/seed coverage", "stratified support label budget",
                       "disjoint fit/validation within support", "identical support across methods",
                       "correct GP tuning objective and iteration count",
                       "aggregate mean and sample standard deviation recomputed"],
            "test_predictions": len(test["labels"]),
            "audit_script_sha256": sha256(__file__)}


def plot_comparison(document, directory):
    summary = document["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    baseline = next(r for r in summary if r["method"] == "zero_shot_clip")
    for ax, metric, title, scale in zip(
        axes, ["accuracy", "macro_f1", "ece"], ["Accuracy (%)", "Macro-F1 (%)", "ECE (10 bins)"], [100, 100, 1]
    ):
        for method in ("bayesian_clip_dino", "bayesian_clip_only", "ridge_clip"):
            values = sorted([r for r in summary if r["method"] == method], key=lambda r: r["shots"])
            ax.errorbar([r["shots"] for r in values], [scale * r[metric + "_mean"] for r in values],
                        yerr=[scale * r[metric + "_std"] for r in values], marker="o", capsize=3,
                        color=COLORS[method], label=LABELS[method], linewidth=1.7)
        ax.axhline(scale * baseline[metric + "_mean"], color=COLORS["zero_shot_clip"],
                   linestyle="--", linewidth=1.4, label=LABELS["zero_shot_clip"])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
        ax.set_xlabel("Labeled images per class")
        ax.set_ylabel(title)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Oxford-IIIT Pet: 3,669 test images / three support seeds", fontsize=13)
    fig.text(0.5, 0.012, "Error bars: sample standard deviation. GP/ridge ECE uses clipped, normalized regression scores.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(directory / "comparison.png", dpi=180)
    plt.close(fig)


def plot_learning_curves(document, directory):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, shots in zip(axes, [1, 16]):
        for row in document["runs"]:
            if row["method"] == "bayesian_clip_dino" and row["shots"] == shots:
                history = row["hyperparameters"]["loss_history"]
                ax.plot(range(len(history)), history, label=f"Seed {row['seed']}", linewidth=1.5)
        ax.set_title(f"{shots}-shot / " + ("marginal likelihood" if shots == 1 else "predictive likelihood"))
        ax.set_xlabel("Adam update")
        ax.set_ylabel("Gaussian NLL per label scalar")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=180)
    plt.close(fig)


def plot_uncertainty(document, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    rows = [r for r in document["summary"] if r["method"] == "bayesian_clip_dino"]
    rows.sort(key=lambda r: r["shots"])
    axes[0].errorbar([r["shots"] for r in rows], [r["mean_variance_mean"] for r in rows],
                     yerr=[r["mean_variance_std"] for r in rows], marker="o", capsize=3, color="#007f73")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    axes[0].set_xlabel("Labeled images per class")
    axes[0].set_ylabel("Mean latent GP variance")
    axes[0].set_title("Bayesian CLIP + DINO")
    coverages = torch.linspace(0.2, 1, 9).tolist()
    for shots, color in zip([1, 4, 16], ["#63778f", "#c15745", "#007f73"]):
        curves = []
        for row in document["runs"]:
            if row["method"] != "bayesian_clip_dino" or row["shots"] != shots:
                continue
            cache = torch.load(row["prediction_file"], weights_only=True)
            correct = cache["probabilities"].argmax(dim=1).eq(cache["labels"])
            ordering = cache["variance"].argsort()
            curves.append([100 * float(correct[ordering[:max(1, int(len(correct) * c))]].double().mean())
                           for c in coverages])
        axes[1].plot([100 * c for c in coverages], torch.tensor(curves).mean(dim=0).tolist(),
                     marker="o", label=f"{shots}-shot", color=color)
    axes[1].set_xlabel("Coverage: lowest-variance predictions retained (%)")
    axes[1].set_ylabel("Accuracy on retained predictions (%)")
    axes[1].set_title("Selective prediction (three-seed average)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(directory / "uncertainty.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment.json")
    parser.add_argument("--features", default="artifacts/features")
    args = parser.parse_args()
    result_path = Path(args.results)
    document = json.loads(result_path.read_text())
    directory = result_path.parent
    checked = audit(document, Path(args.features))
    annotations = Path("data/oxford-iiit-pet/annotations")
    if annotations.exists():
        train_ids = {line.split()[0] for line in (annotations / "trainval.txt").read_text().splitlines() if line.strip()}
        test_ids = {line.split()[0] for line in (annotations / "test.txt").read_text().splitlines() if line.strip()}
        if train_ids & test_ids:
            raise ValueError("trainval/test image IDs overlap")
        if len(train_ids) != document["settings"]["trainval_count"] or len(test_ids) != document["settings"]["test_count"]:
            raise ValueError("image ID counts do not match cached splits")
        checked["checks"].append("official trainval/test image IDs are disjoint and match cache counts")
        checked["split_sha256"] = {name: sha256(annotations / name) for name in ["trainval.txt", "test.txt"]}
    plot_comparison(document, directory)
    plot_learning_curves(document, directory)
    plot_uncertainty(document, directory)
    (directory / "validation.json").write_text(json.dumps(checked, indent=2) + "\n")
    print(json.dumps(checked, indent=2))
    print("Figures saved:", str(directory / "comparison.png"), str(directory / "learning_curves.png"),
          str(directory / "uncertainty.png"))


if __name__ == "__main__":
    main()
