# Changelog

## [3.0.0] - 2026-11-25

### Added
- **Equal-tail two-sided p-values (`tails='equal'`):** Added support for `2 * min(p_left, p_right)` bootstrap inference across all backends.
- **Upfront `tails` input validation:** Added `_validate_tails` to validate tail specifications before running model estimation or bootstrap loops.

### Changed
- **Breaking — Default two-tailed inference (`tails='two'`):** Symmetric two-tailed evaluation is now the default across asymptotic (`p_asy`) and bootstrap (`p_boot`) p-values. Pass `tails='one'` to reproduce `v2.0.3` one-tailed behavior.
- **Breaking — Replaced `boot_tails` with `tails`:** Removed the `boot_tails` keyword argument in favor of an explicit `tails` parameter in `jtest_np`, `jtest_sm`, and `jtest_scipy`.
- **Unified asymptotic p-value tail handling:** Standardized asymptotic Student-t p-value calculation across `jtest_np`, `jtest_sm`, and `jtest_scipy`.

## [2.0.3] - 2026-10

### Added
- Exposed `p0_X`, `p0_Z`, `jtest_solver`, and `curvefitmethod` directly in the function signature of `jtest_scipy`.
- Added `**kwargs` fallbacks for `p0_X`, `p0_Z`, and `curvefitmethod`.

### Changed
- Set `curvefitmethod=None` by default in `jtest_scipy` so SciPy automatically selects between `'lm'` and `'trf'`.

### Fixed
- Minor bug fixes.

## [2.0.2] - 2026-10

### Fixed
- General bug fixes.

## [2.0.1] - 2026-10

### Added
- Added Zenodo DOI metadata and citation reference.

## [2.0.0] - 2026-10

### Added
- **Formatted test summaries:** Added `JtestResults.summary()` for table-formatted console output of test statistics.
- **Cross-platform CI matrix:** Configured GitHub Actions testing across Ubuntu, macOS, and Windows for Python 3.13 and 3.14.

### Changed
- **Breaking — Bidirectional testing by default (`direction='pairwise'`):** `jtest_np`, `jtest_sm`, and `jtest_scipy` now test both directions (H1 vs. H2 and H2 vs. H1) by default. Pass `direction='one-way'` to replicate `v1.x` behavior.
- **Breaking — Direction-specific `JtestResults` attributes:** Renamed `t_asy`, `p_boot`, and `residuals` to `t_asy_X` / `t_asy_Z`, `p_boot_X` / `p_boot_Z`, and `residuals_X` / `residuals_Z`.
- **NumPy QR solver default:** Updated `jtest_np` to default to Householder QR decomposition (`solver='qr'`) with `lstsq` fallback.
- **Statsmodels rank-deficiency fallback:** Updated `jtest_sm` to catch `LinAlgError` on singular or ill-conditioned matrices and fall back to pseudo-inverse (`solve_method='pinv'`).
- **Code modularization:** Extracted shared logic into helper functions (`_calc_aug_reg`, `_setup_residual_scaling`, `_draw_y_star`).

### Fixed
- Improved statistical fallbacks for parameter covariance (`pcov`) calculations in `jtest_scipy`.

## [1.0.2] - 2026-10

### Changed
- Updated `README.md`.

## [1.0.1] - 2026-10

### Changed
- Minor revisions and `README.md` updates.

## [1.0.0] - 2026-10-09

### Added
- Initial release of `bdmjtest` implementing the bootstrapped Davidson–MacKinnon J-test.
- Adopted NumPy's modern random number generator (`numpy.random.default_rng()` / `numpy.random.Generator`).
- Standardized function structure and variable names across backends.
