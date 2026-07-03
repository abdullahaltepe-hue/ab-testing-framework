"""
Statistical Correctness Tests for the A/B Testing Framework.

Tests verify that:
- Power analysis returns reasonable sample sizes
- Bayesian posterior updates correctly
- CUPED reduces variance as expected
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, "..")

from src.power_analysis import PowerAnalyzer
from src.bayesian import BayesianABTest
from src.variance_reduction import CUPED


# ---------------------------------------------------------------------------
# Power Analysis Tests
# ---------------------------------------------------------------------------

class TestPowerAnalysis:
    """Tests for PowerAnalyzer sample size calculations."""

    def setup_method(self):
        self.analyzer = PowerAnalyzer()

    def test_sample_size_increases_with_smaller_mde(self):
        """Smaller MDE should require larger sample sizes."""
        n_large_mde = self.analyzer.required_sample_size(
            baseline_rate=0.10, minimum_detectable_effect=0.02
        )
        n_small_mde = self.analyzer.required_sample_size(
            baseline_rate=0.10, minimum_detectable_effect=0.01
        )
        assert n_small_mde > n_large_mde
        # Roughly 4x larger for half the MDE (quadratic relationship)
        assert n_small_mde > 3 * n_large_mde

    def test_sample_size_increases_with_higher_power(self):
        """Higher power should require larger sample sizes."""
        n_80 = self.analyzer.required_sample_size(
            baseline_rate=0.10, minimum_detectable_effect=0.01, power=0.80
        )
        n_90 = self.analyzer.required_sample_size(
            baseline_rate=0.10, minimum_detectable_effect=0.01, power=0.90
        )
        assert n_90 > n_80

    def test_sample_size_reasonable_range(self):
        """Sample sizes should be in a reasonable range for typical parameters."""
        n = self.analyzer.required_sample_size(
            baseline_rate=0.05,
            minimum_detectable_effect=0.01,
            alpha=0.05,
            power=0.80,
        )
        # For 5% baseline, 1pp MDE: expect roughly 3000-8000 per group
        assert 2000 < n < 10000

    def test_one_sided_requires_fewer_samples(self):
        """One-sided test should require fewer samples than two-sided."""
        n_two_sided = self.analyzer.required_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.02,
            one_sided=False,
        )
        n_one_sided = self.analyzer.required_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.02,
            one_sided=True,
        )
        assert n_one_sided < n_two_sided

    def test_continuous_sample_size_scales_with_variance(self):
        """Higher variance should require more samples."""
        n_low_var = self.analyzer.required_sample_size_continuous(
            baseline_mean=100,
            baseline_std=10,
            minimum_detectable_effect=5,
        )
        n_high_var = self.analyzer.required_sample_size_continuous(
            baseline_mean=100,
            baseline_std=20,
            minimum_detectable_effect=5,
        )
        # Quadratic in std: doubling std -> 4x samples
        assert n_high_var > 3.5 * n_low_var

    def test_power_curve_reaches_high_power(self):
        """Power curve should approach 1.0 for large enough sample sizes."""
        curve = self.analyzer.power_curve(
            baseline_rate=0.05,
            sample_sizes=[100000],
            mde=0.01,
        )
        assert curve[0]["power"] > 0.99

    def test_power_curve_monotonically_increases(self):
        """Power should increase monotonically with sample size."""
        sample_sizes = [1000, 5000, 10000, 20000, 50000]
        curve = self.analyzer.power_curve(
            baseline_rate=0.05,
            sample_sizes=sample_sizes,
            mde=0.01,
        )
        powers = [d["power"] for d in curve]
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1]

    def test_sensitivity_analysis_consistency(self):
        """Sensitivity analysis MDE should be consistent with sample size calculation."""
        baseline = 0.05
        n = self.analyzer.required_sample_size(
            baseline_rate=baseline,
            minimum_detectable_effect=0.01,
            power=0.80,
        )
        sensitivity = self.analyzer.sensitivity_analysis(
            baseline_rate=baseline,
            sample_size=n,
            power=0.80,
        )
        # The MDE from sensitivity analysis should be close to 0.01
        assert abs(sensitivity["minimum_detectable_effect"] - 0.01) < 0.002

    def test_duration_estimation(self):
        """Duration estimation should be proportional to sample size / traffic."""
        result = self.analyzer.estimate_duration(
            required_sample_size=10000,
            daily_traffic=5000,
            allocation_pct=1.0,
            num_variants=2,
        )
        # 10000 per variant, 2500 per variant per day -> 4 days
        assert result.estimated_duration_days == 4
        assert result.total_sample_size == 20000


# ---------------------------------------------------------------------------
# Bayesian Tests
# ---------------------------------------------------------------------------

class TestBayesianABTest:
    """Tests for Bayesian posterior correctness."""

    def setup_method(self):
        self.bayes = BayesianABTest(n_simulations=200000)

    def test_equal_data_gives_50_50(self):
        """Equal conversion data should give ~50% probability for each."""
        result = self.bayes.beta_binomial(
            control_conversions=100,
            control_total=1000,
            treatment_conversions=100,
            treatment_total=1000,
        )
        assert 0.45 < result.prob_b_better < 0.55

    def test_clearly_better_treatment_detected(self):
        """A clearly better treatment should have high P(B > A)."""
        result = self.bayes.beta_binomial(
            control_conversions=50,
            control_total=1000,
            treatment_conversions=80,
            treatment_total=1000,
        )
        assert result.prob_b_better > 0.99

    def test_posterior_mean_approaches_mle(self):
        """With large data, posterior mean should approach MLE."""
        n = 10000
        conv = 500  # 5% rate
        result = self.bayes.beta_binomial(
            control_conversions=conv,
            control_total=n,
            treatment_conversions=conv,
            treatment_total=n,
        )
        # Posterior mean should be very close to 0.05
        assert abs(result.posterior_mean_a - 0.05) < 0.005
        assert abs(result.posterior_mean_b - 0.05) < 0.005

    def test_credible_interval_contains_true_difference(self):
        """95% CI should contain the true difference for well-specified model."""
        # True difference is 0 (same rates)
        result = self.bayes.beta_binomial(
            control_conversions=500,
            control_total=10000,
            treatment_conversions=500,
            treatment_total=10000,
        )
        ci_lower, ci_upper = result.credible_interval
        assert ci_lower < 0 < ci_upper

    def test_expected_loss_decreases_with_more_data(self):
        """Expected loss should decrease as we gather more data (same rates)."""
        result_small = self.bayes.beta_binomial(
            control_conversions=5,
            control_total=100,
            treatment_conversions=5,
            treatment_total=100,
        )
        result_large = self.bayes.beta_binomial(
            control_conversions=50,
            control_total=1000,
            treatment_conversions=50,
            treatment_total=1000,
        )
        # With equal rates, the total expected loss (a+b) shrinks with more data
        total_loss_small = result_small.expected_loss_a + result_small.expected_loss_b
        total_loss_large = result_large.expected_loss_a + result_large.expected_loss_b
        assert total_loss_large < total_loss_small

    def test_lift_distribution_positive_for_better_treatment(self):
        """Lift distribution should show positive lift when treatment is better."""
        lift = self.bayes.compute_lift_distribution(
            control_conversions=50,
            control_total=1000,
            treatment_conversions=70,
            treatment_total=1000,
        )
        assert lift["mean_lift"] > 0
        assert lift["prob_positive_lift"] > 0.95

    def test_normal_normal_model_basic(self):
        """Normal-Normal model should detect a clear difference."""
        result = self.bayes.normal_normal(
            control_mean=10.0,
            control_std=2.0,
            control_n=1000,
            treatment_mean=10.5,
            treatment_std=2.0,
            treatment_n=1000,
        )
        assert result.prob_b_better > 0.95
        assert result.posterior_mean_b > result.posterior_mean_a


# ---------------------------------------------------------------------------
# CUPED Tests
# ---------------------------------------------------------------------------

class TestCUPED:
    """Tests for CUPED variance reduction."""

    def setup_method(self):
        self.cuped = CUPED()
        np.random.seed(42)

    def test_cuped_reduces_variance_with_correlated_covariate(self):
        """CUPED should reduce variance when pre/post metrics are correlated."""
        n = 5000
        # Simulate correlated pre/post data
        pre = np.random.normal(100, 15, n)
        noise = np.random.normal(0, 5, n)
        post = 0.8 * pre + noise  # Strong correlation

        adjusted = self.cuped.apply_cuped(post, pre)
        assert np.var(adjusted) < np.var(post)
        assert self.cuped.variance_reduction_pct > 30  # Expect substantial reduction

    def test_cuped_no_reduction_with_uncorrelated_covariate(self):
        """CUPED should not reduce variance much when covariate is uncorrelated."""
        n = 5000
        post = np.random.normal(100, 15, n)
        pre = np.random.normal(50, 10, n)  # Independent

        adjusted = self.cuped.apply_cuped(post, pre)
        # Variance reduction should be near zero (or slightly negative due to noise)
        assert abs(self.cuped.variance_reduction_pct) < 5

    def test_cuped_preserves_mean(self):
        """CUPED adjustment should preserve the sample mean (approximately)."""
        n = 5000
        pre = np.random.normal(100, 15, n)
        post = 0.7 * pre + np.random.normal(10, 5, n)

        adjusted = self.cuped.apply_cuped(post, pre)
        # Means should be close (within sampling error)
        assert abs(np.mean(adjusted) - np.mean(post)) < 1.0

    def test_cuped_full_analysis_detects_effect(self):
        """Full CUPED analysis should detect a real treatment effect."""
        n = 2000
        # Control
        control_pre = np.random.normal(100, 15, n)
        control_post = 0.7 * control_pre + np.random.normal(0, 5, n)
        # Treatment with +2 unit effect
        treatment_pre = np.random.normal(100, 15, n)
        treatment_post = 0.7 * treatment_pre + np.random.normal(2, 5, n)

        results = self.cuped.analyze_with_cuped(
            control_post=control_post,
            control_pre=control_pre,
            treatment_post=treatment_post,
            treatment_pre=treatment_pre,
        )

        # CUPED should have smaller standard error
        assert results["with_cuped"]["standard_error"] < results["without_cuped"]["standard_error"]
        # CUPED p-value should be smaller (more significant)
        assert results["with_cuped"]["p_value"] < results["without_cuped"]["p_value"]
        # Should detect the effect
        assert results["with_cuped"]["is_significant"] == True

    def test_cuped_variance_reduction_percentage(self):
        """Variance reduction should match theoretical expectation (rho^2)."""
        n = 10000
        rho = 0.7  # target correlation
        pre = np.random.normal(0, 1, n)
        post = rho * pre + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)

        self.cuped.apply_cuped(post, pre)
        # Theoretical reduction = rho^2 * 100% = 49%
        expected_reduction = rho**2 * 100
        assert abs(self.cuped.variance_reduction_pct - expected_reduction) < 5

    def test_cuped_se_reduction_reported(self):
        """The analyze method should report SE reduction stats."""
        n = 3000
        control_pre = np.random.normal(50, 10, n)
        control_post = 0.6 * control_pre + np.random.normal(5, 4, n)
        treatment_pre = np.random.normal(50, 10, n)
        treatment_post = 0.6 * treatment_pre + np.random.normal(6, 4, n)

        results = self.cuped.analyze_with_cuped(
            control_post=control_post,
            control_pre=control_pre,
            treatment_post=treatment_post,
            treatment_pre=treatment_pre,
        )

        assert "improvement" in results
        assert results["improvement"]["variance_reduction_pct"] > 0
        assert results["improvement"]["se_reduction_pct"] > 0
        assert results["improvement"]["correlation_xy"] > 0.3
