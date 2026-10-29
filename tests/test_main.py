import pytest
import warnings
import numpy as np
import itertools

# Import functions from bdmjtest
from bdmjtest import jtest_np, jtest_sm, jtest_scipy, JtestResults

# 1. Synthetic Data Fixture
@pytest.fixture(scope="module")
def synthetic_data():
    np.random.seed(101)
    N = 100
    X = np.column_stack([np.ones(N), np.random.normal(0, 1, N), np.random.uniform(1, 5, N)])
    Z = np.column_stack([np.ones(N), np.random.poisson(3, N), np.random.normal(2, 0.5, N)])
    true_beta = np.array([1.5, -2.0, 0.5])
    y = X @ true_beta + np.random.normal(0, 1, N)
    return y, X, Z

# 2. Reusable Dummy Functions for SciPy
# Standard SciPy dummy functions
def lin_func_X(X, b0, b1, b2): return b0 * X[:, 0] + b1 * X[:, 1] + b2 * X[:, 2]


def lin_func_Z(Z, g0, g1, g2): return g0 * Z[:, 0] + g1 * Z[:, 1] + g2 * Z[:, 2]


# LMFIT dummy functions (lmfit explicitly requires the kwarg to match the data 'x')
def lmfit_func_X(x, b0, b1, b2): return b0 * x[:, 0] + b1 * x[:, 1] + b2 * x[:, 2]


def lmfit_func_Z(x, g0, g1, g2): return g0 * x[:, 0] + g1 * x[:, 1] + g2 * x[:, 2]

# 3. Exhaustive Grid Setup
BOOT_METHODS = ["wild", "residuals", "pairs", "parametric", "semiparametric"]
DIRECTIONS = ["one-way", "pairwise"]
WILD_DISTS = ["rademacher", "mammen", "webb", "normal"]
NP_SOLVERS = ["qr", "lstsq"]
SM_SOLVERS = ["auto", "qr", "pinv"]
SCIPY_SOLVERS = ["hybrid-ols", "scipy"]

# 4. Test Cases
@pytest.mark.parametrize("boot_method", BOOT_METHODS)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("solver", NP_SOLVERS)
def test_jtest_np_exhaustive_configs(synthetic_data, boot_method, direction, solver):
    y, X, Z = synthetic_data
    dist = "rademacher" if boot_method == "wild" else None

    with pytest.warns(UserWarning, match="yields a minimum p-value"):
        res = jtest_np(y=y, X=X, Z=Z, B=19, boot_method=boot_method, res_distribution=dist, direction=direction,
                       solver=solver, seed=42)

    assert isinstance(res, JtestResults)
    assert not np.isnan(res.t_asy_X)


@pytest.mark.parametrize("dist", WILD_DISTS)
def test_wild_distributions(synthetic_data, dist):
    y, X, Z = synthetic_data

    with pytest.warns(UserWarning, match="yields a minimum p-value"):
        res = jtest_np(y, X, Z, B=19, boot_method="wild", res_distribution=dist, seed=42)

    assert 0.0 <= res.p_boot_X <= 1.0


@pytest.mark.parametrize("boot_method", BOOT_METHODS)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("solver", SM_SOLVERS)
def test_jtest_sm_exhaustive_configs(synthetic_data, boot_method, direction, solver):
    y, X, Z = synthetic_data

    with pytest.warns(UserWarning, match="yields a minimum p-value"):
        res = jtest_sm(y=y, X=X, Z=Z, B=19, boot_method=boot_method, direction=direction, solver=solver, seed=42)

    assert isinstance(res, JtestResults)


@pytest.mark.parametrize("solver", SCIPY_SOLVERS)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_jtest_scipy_exhaustive_configs(synthetic_data, solver, direction):
    y, X, Z = synthetic_data

    with pytest.warns(UserWarning, match="yields a minimum p-value"):
        res = jtest_scipy(y=y, X=X, Z=Z, func_X=lin_func_X, func_Z=lin_func_Z, B=19, boot_method="wild",
                          jtest_solver=solver, direction=direction, p0_X=[1, 1, 1], p0_Z=[1, 1, 1], seed=42)

    assert isinstance(res, JtestResults)


def test_jtest_scipy_lmfit_solver(synthetic_data):
    y, X, Z = synthetic_data
    # Use the lmfit-specific functions and dictionary-based initializers
    with pytest.warns(UserWarning, match="yields a minimum p-value"):
        res = jtest_scipy(
            y=y, X=X, Z=Z,
            func_X=lmfit_func_X, func_Z=lmfit_func_Z,
            B=19,
            boot_method="wild",
            jtest_solver="lmfit",
            p0_X={'b0': 1.0, 'b1': 1.0, 'b2': 1.0},
            p0_Z={'g0': 1.0, 'g1': 1.0, 'g2': 1.0},
            seed=42
        )
    assert isinstance(res, JtestResults)


def test_jtest_sm_rank_deficient_fallback(synthetic_data):
    y, X, _ = synthetic_data
    # Induce rank deficiency / perfect collinearity
    Z_collinear = np.column_stack([X[:, 0], X[:, 0] * 2, X[:, 1]])

    # Catch and ignore the flood of Statsmodels collinearity warnings during bootstrap iterations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = jtest_sm(
            y=y, X=X, Z=Z_collinear,
            B=19,
            boot_method="wild",
            seed=42
        )

    assert isinstance(res, JtestResults)


@pytest.mark.parametrize("flags", list(itertools.product([True, False], repeat=3)))
def test_optional_returns_matrix(synthetic_data, flags):
    ret_res, ret_fit, ret_t = flags
    y, X, Z = synthetic_data

    with pytest.warns(UserWarning, match="yields a minimum p-value"):
        res = jtest_np(y, X, Z, B=9, return_residuals=ret_res, return_fitted=ret_fit, return_t_boot=ret_t,
                       direction="pairwise", seed=42)

    if ret_res: assert res.residuals_X is not None
    if ret_fit: assert res.fitted_X is not None
    if ret_t: assert res.t_boot_X is not None


@pytest.mark.parametrize("invalid_B", [0, -5])
def test_invalid_B_raises_value_error(synthetic_data, invalid_B):
    y, X, Z = synthetic_data
    with pytest.raises(ValueError, match="must be a strictly positive integer"):
        jtest_np(y, X, Z, B=invalid_B, boot_method="wild")


def test_mismatched_dimensions(synthetic_data):
    y, X, Z = synthetic_data
    # Suppress the B warning before catching the ValueError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError):
            jtest_np(y[:-1], X, Z, B=10, boot_method="wild")


@pytest.mark.parametrize("invalid_method", ["magic_bootstrap", ""])
def test_invalid_boot_method(synthetic_data, invalid_method):
    y, X, Z = synthetic_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="Invalid boot method"):
            jtest_np(y, X, Z, B=10, boot_method=invalid_method)


def test_invalid_solver_np(synthetic_data):
    y, X, Z = synthetic_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="Unknown solver"):
            jtest_np(y, X, Z, B=10, boot_method="wild", solver="cholesky")
          
