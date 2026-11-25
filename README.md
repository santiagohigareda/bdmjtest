# bdmjtest: Bootstrapped Davidson-MacKinnon J-test in Python

[![PyPI version](https://img.shields.io/pypi/v/bdmjtest?cacheSeconds=3600)](https://pypi.org/project/bdmjtest/)
[![Development Status](https://img.shields.io/badge/Development%20Status-4%20--%20Beta-blue.svg)](https://pypi.org/project/bdmjtest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23786071.svg)](https://doi.org/10.5281/zenodo.23786071)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/santiagohigareda/bdmjtest/actions/workflows/ci.yml/badge.svg)](https://github.com/santiagohigareda/bdmjtest/actions/workflows/ci.yml)

`bdmjtest` is an open-source Python package that provides a high-performance Bootstrapped J-test implementation for comparing non-nested linear and non-linear models.
## What's New in v3.0.0
Standardized p-value evaluation across all backends.

### Breaking Changes
- Symmetric two-tailed test is now the deault (`tails='two'`)
- Replaced `kwargs` argument `boot_tails` with explicit `tails` parameter across `jtest_np`, `jtest_sm` and `jtest_scipy`.
- Unified asymptotic p-value tail handling across all backends.

### New Features and Improvements
- Added support for equal-tail two-sided p-value (`tails='equal'`
- Upfront `tails` input validation.

### Migration Guide (`v2.0.3` -> `v3.0.0`)
```
# v2.0.3
res = jtest_np(y, X, Z, boot_tails='two')
res_one = jtest_np(y, X, Z) # previously defaulted to one-tailed

# v.3.0.0
res = jtest_np(y, X, Z) # now defaults to tails='two'
```
      
## Features
- **Multiple Bootstrap DGPs:** Support residuals (semiparametric), parametric, wild and pairs (nonparametric) bootstrap Data Generation Processes (DGPs) to handle different error structures (e.g. non-normal, skewed or heavy-tailed).
- **Three Backend Engines:** Choose between lightweight NumPy (`jtest_np`), Statsmodels (`jtest_sm`), or SciPy (`jtest_scipy`), based on the characteristics and requirements of the models and data being tested.
- **Standardized Output:** Returns an immutable 'JtestsResults' object containing the asymptotic J-test t-statistics, asymptotic p-values, bootstrap p-values, valid iteration countss and optional series (residuals and fitted values).

## Installation
```
pip install bdmjtest
```

## Usage
The following example is for testing one linear model (H1) against another (H2) using the NumPy backend.
```
import numpy as np
import bdmjtest as jtest

# Define your dependent variable (y) and explanatory variables for H1 (X) and H2 (Z)
y = np.random.rand(100)
X = np.random.rand(100, 1)
Z = np.random.rand(100, 1)

# Perform the bootstrapped J-test
results = jtest_np(y, X, Z)

# Results
  # View a summary table
print(results.summary())
  # Access specific metrics
print(f"H1 vs H2 Bootstrap p-value: {results.p_boot_X}")
print(f"H2 vs H1 Bootstrap p-value: {results.p_boot_Z}")
```

## Citation
If you use `bdmjtest` in a published work, please cite as:

APA 7
```
Guerrero-Higareda, S. (2026). bdmjtest: Bootstrapped Davidson-MacKinnon J-test in Python (Version v3.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23786072
```


## Support and contributing
Contributions are welcome! Questions, bugs reports and feature requests go to [GitHub Issues](https://github.com/santiagohigareda/bdmjtest/issues). If you don´t have a GitHub account or have a particular inquiry, [email me](mailto:s.g.higareda@proton.me).

## License

Copyright (c) 2026 Santiago Guerrero-Higareda

This project is licensed under the MIT License. See [LICENSE](https://github.com/santiagohigareda/bdmjtest/blob/main/LICENSE) for details.
