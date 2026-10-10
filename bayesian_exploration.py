"""Exact label-regression GP with frozen deep kernels (Miao et al., CVPR 2024)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class GPResult:
    mean: torch.Tensor
    variance: torch.Tensor


def squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.double(), dim=-1)
    y = F.normalize(y.double(), dim=-1)
    return (2.0 - 2.0 * (x @ y.T)).clamp_min(0.0)


def rbf_kernel(x, y, lengthscale=1.0):
    if float(lengthscale) <= 0:
        raise ValueError("lengthscale must be positive")
    return torch.exp(-squared_distances(x, y) / (2.0 * lengthscale**2))


def kernel_from_distances(distances, lengthscales):
    if len(distances) != len(lengthscales) or not distances:
        raise ValueError("one lengthscale is required per feature space")
    return sum(torch.exp(-d / (2.0 * scale**2)) for d, scale in zip(distances, lengthscales))


def additive_deep_kernel(features: Sequence[torch.Tensor], other, lengthscale=1.0):
    if not features or len(features) != len(other):
        raise ValueError("matching nonempty feature spaces are required")
    scales = [lengthscale] * len(features) if isinstance(lengthscale, (float, int)) else lengthscale
    return kernel_from_distances([squared_distances(a, b) for a, b in zip(features, other)], scales)


def clip_prior(similarities, temperature=100.0, scale=1.0):
    """Equation 8: gamma * softmax(tau * image-text cosine similarities)."""
    return scale * F.softmax(similarities.double() * temperature, dim=-1)


def gp_posterior(train_features, test_features, train_targets, prior_train, prior_test,
                 noise=0.1, lengthscale=1.0):
    """Equation 3; noise is a standard deviation, and variance is latent variance."""
    k_train = additive_deep_kernel(train_features, train_features, lengthscale)
    k_cross = additive_deep_kernel(test_features, train_features, lengthscale)
    eye = torch.eye(len(k_train), dtype=k_train.dtype, device=k_train.device)
    factor = torch.linalg.cholesky(k_train + (noise**2 + 1e-8) * eye)
    residual = train_targets.double() - prior_train.double()
    mean = prior_test.double() + k_cross @ torch.cholesky_solve(residual, factor)
    projected = torch.linalg.solve_triangular(factor, k_cross.T, upper=False)
    # Every normalized RBF kernel has diagonal 1; avoid a full test-test matrix.
    variance = (len(train_features) - projected.square().sum(dim=0)).clamp_min(0.0)
    return GPResult(mean, variance)


def gaussian_nll(residual, covariance):
    """Joint Gaussian NLL summed over independent class outputs, per label scalar."""
    n, c = residual.shape
    factor = torch.linalg.cholesky(covariance)
    quadratic = (residual * torch.cholesky_solve(residual, factor)).sum()
    logdet = 2.0 * factor.diagonal().log().sum()
    return (quadratic + c * logdet + n * c * math.log(2 * math.pi)) / (2 * n * c)


def tune_hyperparameters(features, similarities, targets, fit_indices, val_indices,
                         steps=100, seed=0):
    """Adam/cosine decay; validation uses the full joint predictive covariance."""
    torch.manual_seed(seed)
    distances = [squared_distances(x, x) for x in features]
    use_validation = len(val_indices) > 0
    log_scales = torch.nn.Parameter(torch.zeros(len(features), dtype=torch.float64))
    log_noise_variance = torch.nn.Parameter(torch.tensor(math.log(0.01), dtype=torch.float64))
    log_tau = torch.nn.Parameter(torch.tensor(math.log(100.0), dtype=torch.float64))
    log_gamma = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    parameters = [log_scales, log_noise_variance, log_tau, log_gamma]
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    targets = targets.double()
    history, best = [], None
    for step in range(steps + 1):
        scales = log_scales.exp()
        noise_variance = log_noise_variance.exp()
        prior = clip_prior(similarities, log_tau.exp(), log_gamma.exp())
        k = kernel_from_distances(distances, scales)
        fit_kernel = k[fit_indices][:, fit_indices]
        eye = torch.eye(len(fit_indices), dtype=torch.float64)
        system = fit_kernel + (noise_variance + 1e-8) * eye
        residual = targets[fit_indices] - prior[fit_indices]
        if use_validation:
            factor = torch.linalg.cholesky(system)
            cross = k[val_indices][:, fit_indices]
            mean = prior[val_indices] + cross @ torch.cholesky_solve(residual, factor)
            val_eye = torch.eye(len(val_indices), dtype=torch.float64)
            covariance = k[val_indices][:, val_indices] - cross @ torch.cholesky_solve(cross.T, factor)
            covariance = (covariance + covariance.T) / 2 + (noise_variance + 1e-8) * val_eye
            loss = gaussian_nll(targets[val_indices] - mean, covariance)
        else:
            loss = gaussian_nll(residual, system)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite GP tuning loss")
        value = float(loss.detach())
        history.append(value)
        if best is None or value < best[0]:
            best = (value, [p.detach().clone() for p in parameters], step)
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("invalid GP tuning gradient")
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            log_scales.clamp_(-4, 4)
            log_noise_variance.clamp_(math.log(1e-4), math.log(10.0))
            log_tau.clamp_(0, math.log(300.0))
            log_gamma.clamp_(-4, 2)
    _, fitted, best_step = best
    return {"lengthscales": fitted[0].exp().tolist(),
            "noise_variance": float(fitted[1].exp()),
            "temperature": float(fitted[2].exp()), "prior_scale": float(fitted[3].exp()),
            "objective": "predictive_likelihood" if use_validation else "marginal_likelihood",
            "best_step": best_step, "loss_history": history}


def expected_calibration_error(probabilities, labels, bins=10):
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    error = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = ((confidence >= lower) if i == 0 else (confidence > lower)) & (confidence <= upper)
        if mask.any():
            error += mask.double().mean() * (confidence[mask].mean() - correct[mask].double().mean()).abs()
    return float(error)


def posterior_probabilities(mean):
    """A reporting convention, not an exact GP categorical predictive likelihood.

    Label regression may yield negative, non-normalized scores. Clip them to a
    positive floor and normalize for categorical NLL/ECE; retain raw GP variance.
    """
    positive = mean.clamp_min(1e-8)
    return positive / positive.sum(dim=-1, keepdim=True)
