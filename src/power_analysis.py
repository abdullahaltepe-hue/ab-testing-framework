"""
Power Analysis & Sample Size Calculation
Determines required sample sizes for A/B experiments.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PowerResult:
    sample_size_per_variant: int
    total_sample_size: int
    estimated_duration_days: int
    achieved_power: float
    parameters: Dict


class PowerAnalyzer:
    """
    Calculates statistical power and required sample sizes
    for proportion and continuous metric experiments.
    """

    def required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        one_sided: bool = False,
        ratio: float = 1.0,
    ) -> int:
        """
        Sample size for a two-proportion z-test.
        Uses the arcsine transformation for better accuracy.
        """
        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect

        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        if one_sided:
            z_alpha = stats.norm.ppf(1 - alpha)
        else:
            z_alpha = stats.norm.ppf(1 - alpha / 2)

        z_beta = stats.norm.ppf(power)

        n = ((z_alpha + z_beta) / h) ** 2
        n = n * (1 + 1 / ratio) / 2

        return int(np.ceil(n))

    def required_sample_size_continuous(
        self,
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """Sample size for continuous metric (t-test)."""
        delta = minimum_detectable_effect
        sigma = baseline_std

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) * sigma / delta) ** 2
        return int(np.ceil(n))

    def estimate_duration(
        self,
        required_sample_size: int,
        daily_traffic: int,
        allocation_pct: float = 1.0,
        num_variants: int = 2,
    ) -> PowerResult:
        """Estimate experiment duration based on traffic."""
        effective_daily = daily_traffic * allocation_pct / num_variants
        duration = int(np.ceil(required_sample_size / effective_daily))

        return PowerResult(
            sample_size_per_variant=required_sample_size,
            total_sample_size=required_sample_size * num_variants,
            estimated_duration_days=duration,
            achieved_power=0.0,
            parameters={
                "daily_traffic": daily_traffic,
                "allocation_pct": allocation_pct,
                "num_variants": num_variants,
            },
        )

    def power_curve(
        self,
        baseline_rate: float,
        sample_sizes: List[int],
        alpha: float = 0.05,
        mde: Optional[float] = None,
    ) -> List[Dict]:
        """Generate power curve data for visualization."""
        if mde is None:
            mde = baseline_rate * 0.1

        p1 = baseline_rate
        p2 = baseline_rate + mde
        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        results = []
        for n in sample_sizes:
            ncp = h * np.sqrt(n)
            power = 1 - stats.norm.cdf(z_alpha - ncp)
            results.append({
                "sample_size": n,
                "power": round(power, 4),
                "is_adequate": power >= 0.80,
            })

        return results

    def sensitivity_analysis(
        self,
        baseline_rate: float,
        sample_size: int,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> Dict:
        """Calculate MDE for a given sample size."""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        h_detectable = (z_alpha + z_beta) / np.sqrt(sample_size)
        p1 = baseline_rate
        p2 = (np.sin(np.arcsin(np.sqrt(p1)) + h_detectable / 2)) ** 2
        mde = p2 - p1

        return {
            "baseline_rate": baseline_rate,
            "sample_size": sample_size,
            "minimum_detectable_effect": round(mde, 6),
            "relative_mde": round(mde / baseline_rate * 100, 2),
            "detectable_rate": round(p2, 6),
        }

    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni",
        alpha: float = 0.05,
    ) -> Dict:
        """Adjust alpha for multiple comparisons."""
        n_tests = len(p_values)
        sorted_pvals = sorted(enumerate(p_values), key=lambda x: x[1])

        if method == "bonferroni":
            adjusted_alpha = alpha / n_tests
            significant = [p < adjusted_alpha for p in p_values]

        elif method == "holm":
            significant = [False] * n_tests
            for rank, (idx, pval) in enumerate(sorted_pvals):
                threshold = alpha / (n_tests - rank)
                if pval < threshold:
                    significant[idx] = True
                else:
                    break

        elif method == "benjamini-hochberg":
            significant = [False] * n_tests
            for rank, (idx, pval) in enumerate(reversed(sorted_pvals)):
                actual_rank = n_tests - rank
                threshold = alpha * actual_rank / n_tests
                if pval <= threshold:
                    for r2, (idx2, _) in enumerate(sorted_pvals[:actual_rank]):
                        significant[idx2] = True
                    break

        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            "method": method,
            "original_alpha": alpha,
            "n_tests": n_tests,
            "results": [
                {"p_value": p, "significant": s}
                for p, s in zip(p_values, significant)
            ],
            "n_significant": sum(significant),
        }
