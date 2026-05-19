"""
Core A/B Experiment Analysis Class
Provides unified interface for frequentist and Bayesian analysis.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ExperimentResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    z_score: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    power: float
    effect_size: float


class ABExperiment:
    """
    Analyzes a two-sample proportion test (A/B test).
    Supports both one-sided and two-sided tests.
    """

    def __init__(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        alpha: float = 0.05,
        one_sided: bool = False,
    ):
        self.control_conversions = control_conversions
        self.control_total = control_total
        self.treatment_conversions = treatment_conversions
        self.treatment_total = treatment_total
        self.alpha = alpha
        self.one_sided = one_sided

        self.control_rate = control_conversions / control_total
        self.treatment_rate = treatment_conversions / treatment_total

    def analyze(self) -> ExperimentResult:
        p_c = self.control_rate
        p_t = self.treatment_rate
        n_c = self.control_total
        n_t = self.treatment_total

        p_pool = (self.control_conversions + self.treatment_conversions) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))

        z_score = (p_t - p_c) / se

        if self.one_sided:
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
        z_crit = stats.norm.ppf(1 - self.alpha / (1 if self.one_sided else 2))
        ci_lower = (p_t - p_c) - z_crit * se_diff
        ci_upper = (p_t - p_c) + z_crit * se_diff

        absolute_lift = p_t - p_c
        relative_lift = absolute_lift / p_c if p_c > 0 else 0

        h = 2 * (np.arcsin(np.sqrt(p_t)) - np.arcsin(np.sqrt(p_c)))
        n_harmonic = 2 * n_c * n_t / (n_c + n_t)
        ncp = h * np.sqrt(n_harmonic)
        power = 1 - stats.norm.cdf(z_crit - ncp)

        return ExperimentResult(
            control_rate=p_c,
            treatment_rate=p_t,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            p_value=p_value,
            z_score=z_score,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            power=power,
            effect_size=h,
        )

    def check_srm(self, expected_ratio: float = 0.5) -> dict:
        """Sample Ratio Mismatch detection."""
        total = self.control_total + self.treatment_total
        expected_control = total * expected_ratio
        expected_treatment = total * (1 - expected_ratio)

        chi2 = ((self.control_total - expected_control) ** 2 / expected_control +
                (self.treatment_total - expected_treatment) ** 2 / expected_treatment)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return {
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p_value, 4),
            "has_srm": p_value < 0.01,
            "actual_ratio": round(self.control_total / total, 4),
            "expected_ratio": expected_ratio,
        }


class ContinuousExperiment:
    """Analyzes A/B test with continuous metrics (revenue, time on page, etc.)."""

    def __init__(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        alpha: float = 0.05,
    ):
        self.control = np.array(control_values)
        self.treatment = np.array(treatment_values)
        self.alpha = alpha

    def welch_t_test(self) -> dict:
        t_stat, p_value = stats.ttest_ind(self.treatment, self.control, equal_var=False)

        mean_diff = self.treatment.mean() - self.control.mean()
        se = np.sqrt(self.treatment.var() / len(self.treatment) + self.control.var() / len(self.control))

        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (mean_diff - z_crit * se, mean_diff + z_crit * se)

        n_c, n_t = len(self.control), len(self.treatment)
        s_c, s_t = self.control.std(), self.treatment.std()
        s_pooled = np.sqrt(((n_c - 1) * s_c**2 + (n_t - 1) * s_t**2) / (n_c + n_t - 2))
        cohens_d = mean_diff / s_pooled

        return {
            "control_mean": round(float(self.control.mean()), 4),
            "treatment_mean": round(float(self.treatment.mean()), 4),
            "mean_difference": round(mean_diff, 4),
            "relative_lift": round(mean_diff / self.control.mean() * 100, 2) if self.control.mean() != 0 else 0,
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "confidence_interval": (round(ci[0], 4), round(ci[1], 4)),
            "is_significant": p_value < self.alpha,
            "cohens_d": round(cohens_d, 4),
        }

    def mann_whitney_test(self) -> dict:
        u_stat, p_value = stats.mannwhitneyu(
            self.treatment, self.control, alternative="two-sided"
        )
        n1, n2 = len(self.treatment), len(self.control)
        effect_size = u_stat / (n1 * n2)

        return {
            "u_statistic": float(u_stat),
            "p_value": round(float(p_value), 4),
            "is_significant": p_value < self.alpha,
            "rank_biserial_r": round(2 * effect_size - 1, 4),
        }
