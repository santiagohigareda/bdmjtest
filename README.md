# bdmjtest: Bootstrapped Davidson-MacKinnon J-test

[![Development Status](https://img.shields.io/badge/Development%20Status-3%20--%20Alpha-orange.svg)](https://pypi.org/project/bdmjtest/)
[![PyPI version](https://img.shields.io/pypi/v/bdmjtest?cacheSeconds=3600)](https://pypi.org/project/bdmjtest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`bdmjtest` is an open-open source Python package that provides a high-performance Bootstrapped J-test implementation for comparing non-nested linear and non-linear models.

## Features
- **Multiple Bootstrap DGPs**: Support residuals (semiparametric), parametric, wild and pairs (nonparametric) bootstrap Data Generation Processes (DGPs) to handle different error structures (e.g. non-normal, skewed or heavy-tailed).
- **Three Backend Engines**: Choose between lightweight NumPy (`jtest_np`), Statsmodels (`jtest_sm`), or SciPy (`jtest_scipy`), based on the characteristics and requirements of the models and data being tested.
- **Standardized Output**: Returns an immutable 'JtestsResults' object containing the asymptotic J-test t-statistic, asymptotic p-value, bootstrap p-value, valid iteration counts and optional series (H<sub>1</sub> residuals and fitted values).

## Installation
```
pip install bdmjtest
```

## Usage
The test must be run pairwise (H<sub>1</sub> vs H<sub>2</sub>, and H<sub>2</sub> vs H<sub>1</sub>).
The following example is for testing one linear model (H1) against another (H2) using the NumPy backend.
```
import numpy as np
import bdmjtest as jtest

# Define your dependent variable (y) and explanatory variables for H1 (X) and H2 (Z)
y = np.random.rand(100)
X = np.random.rand(100, 1)
Z = np.random.rand(100, 1)

# Perform the bootstrapped J-test
# H1 vs H2
res_1 = jtest.jtest_np(y, X, Z, boot_method='residuals', B=999)
# H2 vs H1
res_2 = jtest.jtest_np(y, Z, X, boot_method='residuals', B=999)

# Results
print(f"H1 vs H2 - Bootstrapped p-value: {res_1.p_boot}")
print(f"H2 vs H1 - Bootstrapped p-value: {res_2.p_boot}")
```

## Citation
If you use `bdmjtest` in a published work, please cite as:

APA 7
```
Guerrero-Higareda, S. (2026). bdmjtest: A Python Package for Bootstrapped Davidson-MacKinnon J-test in Linear and Non-Linear Non-Nested Models (Version 1.0.2) [Computer Software]. https://github.com/santiagohigareda/bdmjtest
```

BibTex
```
@software{bdmjtest2026,
  author = {Guerrero-Higareda, Santiago},
  title = {bdmjtest: A Python Package for Bootstrapped Davidson-MacKinnon J-test in Linear and Non-Linear Non-Nested Models},
  year = {2026},
  version = {1.0.2},
  url = {https://github.com/santiagohigareda/bdmjtest}
```

## Support and contributing
Contributions are welcome! Questions, bugs reports and feature requests go to [GitHub Issues](https://github.com/santiagohigareda/bdmjtest/issues). If you don´t have a GitHub account or have a particular inquiry, [email me](mailto:s.g.higareda@proton.me).

## License

Copyright (c) 2026 Santiago Guerrero-Higareda

This project is licensed under the MIT License. See [LICENSE](https://github.com/santiagohigareda/bdmjtest/blob/main/LICENSE) for details.
