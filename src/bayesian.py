"""
Bayesian A/B Testing
Implements conjugate prior models for experiment analysis.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class BayesianResult:
    prob_b_better: float
    expected_loss_a: float
    expected_loss_b: float
    credible_interval: Tuple[float, float]
    posterior_mean_a: float
    posterior_mean_b: float
    risk_threshold_met: bool


class BayesianABTest:
    """
    Bayesian inference for A/B tests using conjugate priors.
    - Proportions: Beta-Binomial model
    - Continuous: Normal-Normal model
    """

    def __init__(self, n_simulations: int = 100000):
        self.n_simulations = n_simulations
        np.random.seed(42)

    def beta_binomial(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        loss_threshold: float = 0.001,
    ) -> BayesianResult:
        """
        Beta-Binomial conjugate model for conversion rate testing.
        Uses uninformative prior by default (Beta(1,1) = Uniform).
        """
        alpha_a = prior_alpha + control_conversions
        beta_a = prior_beta + (control_total - control_conversions)

        alpha_b = prior_alpha + treatment_conversions
        beta_b = prior_beta + (treatment_total - treatment_conversions)

        samples_a = np.random.beta(alpha_a, beta_a, self.n_simulations)
        samples_b = np.random.beta(alpha_b, beta_b, self.n_simulations)

        prob_b_better = (samples_b > samples_a).mean()

        loss_a = np.maximum(samples_b - samples_a, 0).mean()
        loss_b = np.maximum(samples_a - samples_b, 0).mean()

        diff_samples = samples_b - samples_a
        ci_lower = np.percentile(diff_samples, 2.5)
        ci_upper = np.percentile(diff_samples, 97.5)

        posterior_mean_a = alpha_a / (alpha_a + beta_a)
        posterior_mean_b = alpha_b / (alpha_b + beta_b)

        return BayesianResult(
            prob_b_better=round(prob_b_better, 4),
            expected_loss_a=round(loss_a, 6),
            expected_loss_b=round(loss_b, 6),
            credible_interval=(round(ci_lower, 6), round(ci_upper, 6)),
            posterior_mean_a=round(posterior_mean_a, 6),
            posterior_mean_b=round(posterior_mean_b, 6),
            risk_threshold_met=min(loss_a, loss_b) < loss_threshold,
        )

    def normal_normal(
        self,
        control_mean: float,
        control_std: float,
        control_n: int,
        treatment_mean: float,
        treatment_std: float,
        treatment_n: int,
        prior_mean: float = 0,
        prior_std: float = 1000,
    ) -> BayesianResult:
        """Normal-Normal conjugate model for continuous metrics."""
        prior_precision = 1 / prior_std**2

        precision_a = control_n / control_std**2
        posterior_precision_a = prior_precision + precision_a
        posterior_mean_a = (prior_precision * prior_mean + precision_a * control_mean) / posterior_precision_a
        posterior_std_a = 1 / np.sqrt(posterior_precision_a)

        precision_b = treatment_n / treatment_std**2
        posterior_precision_b = prior_precision + precision_b
        posterior_mean_b = (prior_precision * prior_mean + precision_b * treatment_mean) / posterior_precision_b
        posterior_std_b = 1 / np.sqrt(posterior_precision_b)

        samples_a = np.random.normal(posterior_mean_a, posterior_std_a, self.n_simulations)
        samples_b = np.random.normal(posterior_mean_b, posterior_std_b, self.n_simulations)

        prob_b_better = (samples_b > samples_a).mean()
        loss_a = np.maximum(samples_b - samples_a, 0).mean()
        loss_b = np.maximum(samples_a - samples_b, 0).mean()

        diff_samples = samples_b - samples_a
        ci_lower = np.percentile(diff_samples, 2.5)
        ci_upper = np.percentile(diff_samples, 97.5)

        return BayesianResult(
            prob_b_better=round(prob_b_better, 4),
            expected_loss_a=round(loss_a, 4),
            expected_loss_b=round(loss_b, 4),
            credible_interval=(round(ci_lower, 4), round(ci_upper, 4)),
            posterior_mean_a=round(posterior_mean_a, 4),
            posterior_mean_b=round(posterior_mean_b, 4),
            risk_threshold_met=True,
        )

    def compute_lift_distribution(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
    ) -> Dict:
        """Compute the posterior distribution of relative lift."""
        alpha_a = 1 + control_conversions
        beta_a = 1 + (control_total - control_conversions)
        alpha_b = 1 + treatment_conversions
        beta_b = 1 + (treatment_total - treatment_conversions)

        samples_a = np.random.beta(alpha_a, beta_a, self.n_simulations)
        samples_b = np.random.beta(alpha_b, beta_b, self.n_simulations)

        lift_samples = (samples_b - samples_a) / samples_a

        return {
            "mean_lift": round(float(lift_samples.mean()), 4),
            "median_lift": round(float(np.median(lift_samples)), 4),
            "ci_95": (
                round(float(np.percentile(lift_samples, 2.5)), 4),
                round(float(np.percentile(lift_samples, 97.5)), 4),
            ),
            "prob_positive_lift": round(float((lift_samples > 0).mean()), 4),
            "prob_lift_gt_5pct": round(float((lift_samples > 0.05).mean()), 4),
            "prob_lift_gt_10pct": round(float((lift_samples > 0.10).mean()), 4),
        }
