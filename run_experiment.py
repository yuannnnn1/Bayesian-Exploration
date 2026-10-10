"""Reproducible low-shot evaluation; tune exclusively inside the support budget."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from bayesian_exploration import (clip_prior, expected_calibration_error, gp_posterior,
                                  posterior_probabilities, tune_hyperparameters)


def choose_support(labels, shots, seed):
    if shots < 1:
        raise ValueError("shots must be positive")
    # Shuffle once per class to make support sets nested across shot counts.
    indices = []
    for label in labels.unique(sorted=True).tolist():
        candidates = (labels == label).nonzero(as_tuple=True)[0].tolist()
        random.Random(seed * 1000 + label).shuffle(candidates)
        if shots > len(candidates):
            raise ValueError("not enough samples in a class")
        indices.extend(candidates[:shots])
    return torch.tensor(indices, dtype=torch.long)


def split_support(labels, seed):
    fit, val = [], []
    rng = random.Random(seed)
    for label in labels.unique(sorted=True).tolist():
        members = (labels == label).nonzero(as_tuple=True)[0].tolist()
        rng.shuffle(members)
        count = len(members) // 2
        val.extend(members[:count])
        fit.extend(members[count:])
    return torch.tensor(fit, dtype=torch.long), torch.tensor(val, dtype=torch.long)


def macro_f1(predictions, labels, num_classes):
    matrix = torch.bincount(labels * num_classes + predictions, minlength=num_classes**2)
    matrix = matrix.reshape(num_classes, num_classes).double()
    denominator = matrix.sum(dim=0) + matrix.sum(dim=1)
    return float((2 * matrix.diagonal() / denominator.clamp_min(1)).mean())


def evaluate(probabilities, labels, variance=None):
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(dim=1), torch.ones(len(labels), dtype=probabilities.dtype), atol=1e-6
    ):
        raise RuntimeError("invalid categorical probabilities")
    prediction = probabilities.argmax(dim=1)
    correct = prediction.eq(labels)
    result = {"accuracy": float(correct.double().mean()),
              "macro_f1": macro_f1(prediction, labels, probabilities.shape[1]),
              "nll": float(-probabilities[torch.arange(len(labels)), labels].clamp_min(1e-12).log().mean()),
              "ece": expected_calibration_error(probabilities, labels),
              "mean_variance": None}
    if variance is not None:
        if not torch.isfinite(variance).all() or (variance < 0).any():
            raise RuntimeError("invalid GP posterior variance")
        result["mean_variance"] = float(variance.mean())
        result["variance_on_correct"] = float(variance[correct].mean()) if correct.any() else None
        result["variance_on_errors"] = float(variance[~correct].mean()) if (~correct).any() else None
        kept = variance.argsort()[:max(1, int(0.8 * len(labels)))]
        result["accuracy_at_80pct_variance_coverage"] = float(correct[kept].double().mean())
    return result


def ridge_scores(x_train, y_train, x_test, regularization):
    x_train, x_test, y_train = x_train.double(), x_test.double(), y_train.double()
    system = x_train @ x_train.T + regularization * torch.eye(len(x_train), dtype=torch.float64)
    factor = torch.linalg.cholesky(system)
    return x_test @ x_train.T @ torch.cholesky_solve(y_train, factor)


def tune_ridge(features, targets, fit, val):
    candidates = [1e-4, 1e-2, 0.1, 1.0, 10.0]
    if len(val) == 0:
        return {"regularization": 0.01, "selection": "fixed_for_1_shot", "validation_mse": []}
    scores = []
    for regularization in candidates:
        prediction = ridge_scores(features[fit], targets[fit], features[val], regularization)
        scores.append(float((prediction - targets[val]).square().mean()))
    selected = min(range(len(scores)), key=lambda i: scores[i])
    return {"regularization": candidates[selected], "selection": "support_validation_mse",
            "validation_mse": [{"regularization": r, "mse": s} for r, s in zip(candidates, scores)]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows):
    summary = []
    groups = sorted({(row["method"], row["shots"]) for row in rows})
    metrics = ["accuracy", "macro_f1", "nll", "ece", "mean_variance", "seconds",
               "accuracy_at_80pct_variance_coverage"]
    for method, shots in groups:
        group = [row for row in rows if row["method"] == method and row["shots"] == shots]
        item = {"method": method, "shots": shots, "runs": len(group)}
        for metric in metrics:
            values = [r[metric] for r in group if r.get(metric) is not None]
            item[metric + "_mean"] = statistics.mean(values) if values else None
            item[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        summary.append(item)
    return summary


def validate_cache(train, test, metadata):
    classes = len(metadata["classes"])
    for split in (train, test):
        labels = split["labels"]
        if labels.dtype != torch.int64 or sorted(labels.unique().tolist()) != list(range(classes)):
            raise ValueError("class labels must be contiguous and match the text class order")
        for key in ("clip", "resnet"):
            features = split[key]
            if features.ndim != 2 or len(features) != len(labels) or not torch.isfinite(features).all():
                raise ValueError("invalid cached features")
            if not torch.allclose(features.norm(dim=-1), torch.ones(len(labels)), atol=1e-4):
                raise ValueError("cached feature rows must be normalized")
    if metadata["text_features"].shape != (classes, train["clip"].shape[1]):
        raise ValueError("invalid text feature shape")
    if not metadata.get("force_quick_gelu", False):
        raise ValueError("regenerate text metadata with prepare_features.py --text-only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="artifacts/features")
    parser.add_argument("--output", default="results/experiment.json")
    parser.add_argument("--predictions", default="artifacts/predictions")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 0 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("invalid steps, threads, or duplicate seeds")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    features_path, output = Path(args.features), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_dir = Path(args.predictions)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    train = torch.load(features_path / "trainval.pt", weights_only=True)
    test = torch.load(features_path / "test.pt", weights_only=True)
    metadata = torch.load(features_path / "metadata.pt", weights_only=True)
    validate_cache(train, test, metadata)
    labels, test_labels = train["labels"], test["labels"]
    classes = len(metadata["classes"])
    text = F.normalize(metadata["text_features"].double(), dim=-1)
    similarities_train = F.normalize(train["clip"].double(), dim=-1) @ text.T
    similarities_test = F.normalize(test["clip"].double(), dim=-1) @ text.T
    methods = {"bayesian_clip_dino": ["clip", "resnet"], "bayesian_clip_only": ["clip"]}
    settings = {
        "dataset": "Oxford-IIIT Pet", "classes": metadata["classes"],
        "trainval_count": len(labels), "test_count": len(test_labels),
        "trainval_class_counts": torch.bincount(labels).tolist(),
        "test_class_counts": torch.bincount(test_labels).tolist(),
        "shots": args.shots, "seeds": args.seeds, "optimization_steps": args.steps,
        "adam_lr": 0.01, "scheduler": "cosine", "kernel": "sum of normalized RBF deep kernels",
        "validation": "1:1 stratified split inside the support budget; no extra labeled data",
        "one_shot_objective": "GP marginal likelihood; ridge lambda fixed to 0.01",
        "final_fit": "all support samples, after hyperparameter selection",
        "sampling": "nested support sets per seed",
        "prior": "gamma * softmax(tau * CLIP text-image cosine similarity)",
        "initial_hyperparameters": {"lengthscales": 1.0, "noise_variance": 0.01, "tau": 100, "gamma": 1},
        "probability_mapping": "GP/ridge: clamp scores at 1e-8 and normalize; zero-shot: softmax(tau=100)",
        "probability_limit": "mapped GP categorical NLL/ECE are conventions, not the Gaussian predictive likelihood",
        "uncertainty": "latent GP variance, independent of class; not an OOD benchmark",
        "ece_bins": 10, "aggregate": "mean and sample standard deviation across seeds",
        "clip_metadata": {k: v for k, v in metadata.items() if k not in ("classes", "text_features")},
        "paper_url": "https://arxiv.org/html/2404.00312v1",
        "deviations": ["two encoders rather than CLIP+MoCo+DINO",
                       "scalar lengthscale per encoder", "deterministic lengthscale initialization",
                       "single text prompt per class", "support-only validation on OxfordPets",
                       "ridge label-regression probe rather than paper's ensemble baselines"],
    }
    source_files = ["bayesian_exploration.py", "run_experiment.py", "prepare_features.py"]
    document = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                                "platform": platform.platform(), "architecture": platform.machine(),
                                "device": "cpu", "threads": args.threads},
                "settings": settings,
                "source_sha256": {p: sha256(p) for p in source_files},
                "feature_sha256": {p: sha256(features_path / p) for p in ("trainval.pt", "test.pt", "metadata.pt")},
                "runs": [], "summary": []}
    log_stream = output.with_suffix(".log").open("w", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_stream.write(message + "\n")
        log_stream.flush()

    def save():
        document["summary"] = summarize(document["runs"])
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    def record(method, shots, seed, probabilities, variance, timer, extra):
        row = {"method": method, "shots": shots, "seed": seed, **evaluate(probabilities, test_labels, variance),
               "seconds": time.perf_counter() - timer, **extra}
        name = f"{method}_{shots}shot_seed{seed}.pt"
        prediction_path = prediction_dir / name
        cache = {"labels": test_labels, "probabilities": probabilities.float(),
                 "variance": variance.float() if variance is not None else torch.empty(0)}
        torch.save(cache, prediction_path)
        row["prediction_file"] = str(prediction_path)
        row["prediction_sha256"] = sha256(prediction_path)
        document["runs"].append(row)
        save()
        log(f"{method:22s} shots={shots:2d} seed={str(seed):4s} accuracy={row['accuracy']:.6f} "
            f"macro_f1={row['macro_f1']:.6f} nll={row['nll']:.4f} ece={row['ece']:.4f} "
            f"seconds={row['seconds']:.3f}")

    try:
        log(f"Dataset: {len(labels)} trainval / {len(test_labels)} test / {classes} classes")
        log("Hyperparameter selection uses support samples only; final prediction uses all support samples.")
        timer = time.perf_counter()
        record("zero_shot_clip", 0, None, clip_prior(similarities_test), None, timer,
               {"hyperparameters": {"temperature": 100.0}})
        for shots in args.shots:
            for seed in args.seeds:
                support = choose_support(labels, shots, seed)
                targets = F.one_hot(labels[support], num_classes=classes).double()
                fit, val = split_support(labels[support], seed)
                shared = {"support_indices": support.tolist(),
                          "tuning_fit_indices": support[fit].tolist(),
                          "tuning_validation_indices": support[val].tolist()}
                for method, keys in methods.items():
                    timer = time.perf_counter()
                    train_features = [train[key][support] for key in keys]
                    test_features = [test[key] for key in keys]
                    hyperparameters = tune_hyperparameters(
                        train_features, similarities_train[support], targets, fit, val,
                        steps=args.steps, seed=seed)
                    prior_train = clip_prior(similarities_train[support], hyperparameters["temperature"],
                                             hyperparameters["prior_scale"])
                    prior_test = clip_prior(similarities_test, hyperparameters["temperature"],
                                            hyperparameters["prior_scale"])
                    with torch.no_grad():
                        posterior = gp_posterior(train_features, test_features, targets, prior_train, prior_test,
                                                 noise=math.sqrt(hyperparameters["noise_variance"]),
                                                 lengthscale=hyperparameters["lengthscales"])
                        probabilities = posterior_probabilities(posterior.mean)
                    if not torch.equal(probabilities.argmax(dim=1), posterior.mean.argmax(dim=1)):
                        raise RuntimeError("probability mapping changed raw GP argmax")
                    record(method, shots, seed, probabilities, posterior.variance, timer,
                           {**shared, "hyperparameters": hyperparameters,
                            "negative_score_fraction": float((posterior.mean < 0).double().mean())})
                timer = time.perf_counter()
                ridge = tune_ridge(train["clip"][support], targets, fit, val)
                scores = ridge_scores(train["clip"][support], targets, test["clip"], ridge["regularization"])
                record("ridge_clip", shots, seed, posterior_probabilities(scores), None, timer,
                       {**shared, "hyperparameters": ridge})
        document["status"] = "complete"
        document["total_seconds"] = time.perf_counter() - started
        document["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        summary_path = output.with_name("summary.csv")
        with summary_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(document["summary"][0]))
            writer.writeheader()
            writer.writerows(document["summary"])
        log(f"Completed {len(document['runs'])} evaluations in {document['total_seconds']:.3f} seconds.")
    except Exception as error:
        document["status"] = "failed"
        document["error"] = str(error)
        save()
        log(f"FAILED: {error}")
        raise
    finally:
        log_stream.close()


if __name__ == "__main__":
    main()
