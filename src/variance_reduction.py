"""
CUPED - Controlled-experiment Using Pre-Experiment Data
Variance reduction technique for A/B experiments.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats


class CUPED:
    """
    Implements CUPED (Controlled-experiment Using Pre-Experiment Data).

    CUPED reduces variance in A/B test metrics by leveraging pre-experiment
    covariates. This increases sensitivity without requiring more samples.

    Reference: Deng et al. (2013) - "Improving the Sensitivity of Online
    Controlled Experiments by Utilizing Pre-Experiment Data"
    """

    def __init__(self):
        self.theta: Optional[float] = None
        self.variance_reduction_pct: float = 0.0

    def apply_cuped(
        self,
        y: np.ndarray,
        x: np.ndarray,
        x_mean: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply CUPED adjustment.

        Parameters:
            y: Post-experiment metric values
            x: Pre-experiment covariate values (same users)
            x_mean: Population mean of x (if known), otherwise uses sample mean

        Returns:
            Adjusted metric values with reduced variance
        """
        if x_mean is None:
            x_mean = x.mean()

        cov_xy = np.cov(y, x)[0, 1]
        var_x = np.var(x, ddof=1)

        self.theta = cov_xy / var_x if var_x > 0 else 0

        y_adjusted = y - self.theta * (x - x_mean)

        original_var = np.var(y, ddof=1)
        adjusted_var = np.var(y_adjusted, ddof=1)
        self.variance_reduction_pct = (1 - adjusted_var / original_var) * 100

        return y_adjusted

    def analyze_with_cuped(
        self,
        control_post: np.ndarray,
        control_pre: np.ndarray,
        treatment_post: np.ndarray,
        treatment_pre: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Run full A/B analysis with CUPED variance reduction.
        Compares results with and without CUPED.
        """
        all_pre = np.concatenate([control_pre, treatment_pre])
        x_mean = all_pre.mean()

        control_adjusted = self.apply_cuped(control_post, control_pre, x_mean)
        treatment_adjusted = self.apply_cuped(treatment_post, treatment_pre, x_mean)

        t_stat_raw, p_value_raw = stats.ttest_ind(treatment_post, control_post, equal_var=False)
        t_stat_cuped, p_value_cuped = stats.ttest_ind(treatment_adjusted, control_adjusted, equal_var=False)

        mean_diff_raw = treatment_post.mean() - control_post.mean()
        mean_diff_cuped = treatment_adjusted.mean() - control_adjusted.mean()

        se_raw = np.sqrt(
            np.var(control_post, ddof=1) / len(control_post) +
            np.var(treatment_post, ddof=1) / len(treatment_post)
        )
        se_cuped = np.sqrt(
            np.var(control_adjusted, ddof=1) / len(control_adjusted) +
            np.var(treatment_adjusted, ddof=1) / len(treatment_adjusted)
        )

        z_crit = stats.norm.ppf(1 - alpha / 2)

        return {
            "without_cuped": {
                "mean_difference": round(float(mean_diff_raw), 4),
                "standard_error": round(float(se_raw), 4),
                "t_statistic": round(float(t_stat_raw), 4),
                "p_value": round(float(p_value_raw), 4),
                "ci_95": (
                    round(float(mean_diff_raw - z_crit * se_raw), 4),
                    round(float(mean_diff_raw + z_crit * se_raw), 4),
                ),
                "is_significant": p_value_raw < alpha,
            },
            "with_cuped": {
                "mean_difference": round(float(mean_diff_cuped), 4),
                "standard_error": round(float(se_cuped), 4),
                "t_statistic": round(float(t_stat_cuped), 4),
                "p_value": round(float(p_value_cuped), 4),
                "ci_95": (
                    round(float(mean_diff_cuped - z_crit * se_cuped), 4),
                    round(float(mean_diff_cuped + z_crit * se_cuped), 4),
                ),
                "is_significant": p_value_cuped < alpha,
            },
            "improvement": {
                "variance_reduction_pct": round(self.variance_reduction_pct, 2),
                "se_reduction_pct": round((1 - se_cuped / se_raw) * 100, 2),
                "theta": round(float(self.theta), 4),
                "correlation_xy": round(float(np.corrcoef(
                    np.concatenate([control_post, treatment_post]),
                    np.concatenate([control_pre, treatment_pre])
                )[0, 1]), 4),
            },
        }
