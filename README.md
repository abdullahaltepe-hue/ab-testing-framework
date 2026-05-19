# A/B Testing Statistical Framework

A rigorous statistical framework for designing, analyzing, and reporting A/B experiments. Implements hypothesis testing, power analysis, sequential testing, and Bayesian inference for product experimentation.

## Motivation

Most A/B testing tools provide p-values without context. This framework provides:
- Pre-experiment power analysis & sample size calculation
- Multiple comparison corrections (Bonferroni, Holm, BH-FDR)
- Sequential testing with spending functions (avoid peeking problems)
- Bayesian posterior estimation for business decision-making
- Effect size estimation with confidence intervals
- Automated experiment reports

## Key Features

### Pre-Experiment
- **Power Analysis**: Calculate required sample size for desired MDE
- **Duration Estimation**: Traffic-based experiment duration planning
- **Metric Selection**: Guardrail vs. success metric framework

### During Experiment
- **Sequential Testing**: O'Brien-Fleming & Pocock boundaries
- **Sample Ratio Mismatch (SRM)**: Detect randomization bugs early
- **Peeking Protection**: Alpha spending functions

### Post-Experiment
- **Frequentist Tests**: Z-test, t-test, chi-squared, Mann-Whitney U
- **Bayesian Analysis**: Beta-Binomial & Normal-Normal conjugate models
- **CUPED**: Variance reduction using pre-experiment covariates
- **Segmentation**: Heterogeneous treatment effect detection

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core framework |
| SciPy | Statistical tests |
| NumPy | Numerical computation |
| Statsmodels | Regression & power analysis |
| Matplotlib / Seaborn | Visualization |
| Jupyter | Interactive analysis |

## Project Structure

```
├── src/
│   ├── experiment.py          # Core experiment class
│   ├── power_analysis.py      # Sample size & power calculations
│   ├── frequentist.py         # Frequentist hypothesis tests
│   ├── bayesian.py            # Bayesian inference
│   ├── sequential.py          # Sequential testing boundaries
│   └── variance_reduction.py  # CUPED implementation
├── notebooks/
│   ├── 01_power_analysis.ipynb
│   ├── 02_frequentist_tests.ipynb
│   ├── 03_bayesian_analysis.ipynb
│   └── 04_sequential_testing.ipynb
├── tests/
│   └── test_statistical_correctness.py
├── examples/
│   └── conversion_rate_test.py
└── requirements.txt
```

## Quick Start

```python
from src.experiment import ABExperiment
from src.power_analysis import PowerAnalyzer

# Pre-experiment: calculate sample size
pa = PowerAnalyzer()
sample_size = pa.required_sample_size(
    baseline_rate=0.12,
    minimum_detectable_effect=0.02,
    alpha=0.05,
    power=0.80
)
print(f"Required: {sample_size:,} per variant")

# Post-experiment: analyze results
exp = ABExperiment(
    control_conversions=1200, control_total=10000,
    treatment_conversions=1350, treatment_total=10000
)
result = exp.analyze()
print(f"Lift: {result.relative_lift:.2%}")
print(f"P-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
```

## Results Validation

All statistical methods are validated against:
- Known analytical solutions
- R `pwr` package outputs
- Monte Carlo simulation (10K+ iterations)
- Published case studies

## License

MIT
