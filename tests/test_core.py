"""Checks against analytical reference formulas and the support-label budget."""
import unittest

import torch
import torch.nn.functional as F

from bayesian_exploration import (additive_deep_kernel, clip_prior, gaussian_nll,
                                  gp_posterior, posterior_probabilities, tune_hyperparameters)
from run_experiment import choose_support, macro_f1, split_support


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = F.normalize(torch.randn(6, 4, dtype=torch.float64), dim=1)
        self.z = F.normalize(torch.randn(3, 4, dtype=torch.float64), dim=1)

    def test_posterior_matches_full_matrix_reference(self):
        targets = F.one_hot(torch.tensor([0, 1, 0, 1, 0, 1]), 2).double()
        m_train, m_test = torch.randn(6, 2).double(), torch.randn(3, 2).double()
        scales = [0.6, 1.1]
        x, z = [self.x, self.x], [self.z, self.z]
        result = gp_posterior(x, z, targets, m_train, m_test, noise=0.2, lengthscale=scales)
        k = additive_deep_kernel(x, x, scales)
        cross = additive_deep_kernel(z, x, scales)
        inv = torch.linalg.inv(k + (0.2**2 + 1e-8) * torch.eye(6).double())
        reference_mean = m_test + cross @ inv @ (targets - m_train)
        reference_variance = (additive_deep_kernel(z, z, scales) - cross @ inv @ cross.T).diagonal()
        self.assertTrue(torch.allclose(result.mean, reference_mean, atol=1e-10))
        self.assertTrue(torch.allclose(result.variance, reference_variance, atol=1e-10))

    def test_kernel_is_sum_and_positive_semidefinite(self):
        k = additive_deep_kernel([self.x, self.x], [self.x, self.x])
        self.assertTrue(torch.allclose(k.diagonal(), torch.full((6,), 2.0).double()))
        self.assertGreater(float(torch.linalg.eigvalsh(k).min()), -1e-10)

    def test_gaussian_nll_matches_distribution(self):
        residual = torch.randn(6, 2).double()
        covariance = self.x @ self.x.T + torch.eye(6).double() * 0.2
        distribution = torch.distributions.MultivariateNormal(torch.zeros(6).double(), covariance)
        reference = -distribution.log_prob(residual.T).sum() / residual.numel()
        self.assertTrue(torch.allclose(gaussian_nll(residual, covariance), reference))

    def test_prior_and_probability_mapping(self):
        prior = clip_prior(torch.tensor([[0.4, 0.1]]), temperature=3, scale=2)
        reference = 2 * F.softmax(torch.tensor([[1.2, 0.3]]).double(), dim=1)
        self.assertTrue(torch.allclose(prior, reference))
        p = posterior_probabilities(torch.tensor([[-0.1, 0.5, 0.8]]))
        self.assertEqual(int(p.argmax()), 2)
        self.assertAlmostEqual(float(p.sum()), 1)

    def test_sampling_is_nested_and_validation_uses_no_extra_labels(self):
        labels = torch.arange(3).repeat_interleave(10)
        small, large = choose_support(labels, 2, 4), choose_support(labels, 4, 4)
        self.assertTrue(set(small.tolist()).issubset(large.tolist()))
        fit, val = split_support(labels[large], 4)
        self.assertFalse(set(fit.tolist()) & set(val.tolist()))
        self.assertEqual(set(fit.tolist()) | set(val.tolist()), set(range(len(large))))
        self.assertEqual(torch.bincount(labels[large[fit]]).tolist(), [2, 2, 2])
        self.assertEqual(torch.bincount(labels[large[val]]).tolist(), [2, 2, 2])

    def test_both_tuning_objectives_are_finite_and_improve(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        targets = F.one_hot(labels, 2).double()
        similarities = self.x[:, :2]
        for fit, val in [(torch.arange(6), torch.empty(0, dtype=torch.long)),
                         (torch.tensor([0, 1, 3, 4]), torch.tensor([2, 5]))]:
            settings = tune_hyperparameters([self.x], similarities, targets, fit, val, steps=3)
            history = settings["loss_history"]
            self.assertTrue(all(torch.isfinite(torch.tensor(history))))
            self.assertLessEqual(min(history), history[0])

    def test_macro_f1_handles_missing_predictions(self):
        self.assertAlmostEqual(macro_f1(torch.tensor([0, 0]), torch.tensor([0, 1]), 2), 1 / 3)


if __name__ == "__main__":
    unittest.main()
