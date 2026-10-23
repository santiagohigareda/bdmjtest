from dataclasses import dataclass
from typing import Union, Optional, Literal, Callable, Any, Dict, Tuple
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
try:
    from lmfit import Model
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False

@dataclass
class JtestResults:
    """Dataclass holding the outputs of the Davidson-MacKinnon J-test.

    Attributes
    ----------
    t_asy : float
        Aymptotic J-test t-statistic computed from original sample data.
    p_asy : float
        Asymptotic p-value based on Student's t-distribution.
    p_boot : float
        Bootstrap p-value calculated across successful iterations.
    s_its : float
        Total number of successfully executed bootstraps.
    residuals : np.ndarray, optional
        Residuals from the null model (H0), returned if requested.
    fitted : np.ndarray, optional
        Fitted values from the null model (H0), returned if requested.
    """
    # Primary variables
    t_asy : float
    p_asy : float
    p_boot : float
    s_its : int
    # Optional variables
    t_boot: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    fitted : Optional[np.ndarray] = None

# NumPy backend
def jtest_np(
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        B: int = 999,
        boot_method: Literal["residuals","semiparametric", "wild", "parametric", "pairs"] = "residuals",
        seed: Optional[Union[int, np.random.Generator]] = None,
        **kwargs
) -> JtestResults:
    """Performs the Bootstrap Davidson-MacKinnon J-test with NumPy backend.

    Test a linear null model (H1) (y on X) against a linear non-nested alternative (H2) (y on Z).

    Parameters
    ----------
    y : array-like of shape (N,)
        Dependent variable vector.
    X : array-like of shape (N, k1) or (N,)
        Explanatory variables under the null hypothesis (H1).
    Z : array-like of shape (N, k2) or (N,)
        Explanatory variables under the alternative hypothesis (H2).
    B : int, default=999
        Number of bootstrap replications.
    boot_method : {'residuals', 'semiparametric', 'wild', 'parametric', 'pairs'}, default='residuals'
        Bootstrap Data Generation Process (DGP) method.
    **kwargs : dict
        Optional controls:
        - return_residuals (bool): Return H1 residuals (default: False).
        - return_fitted (bool): Return H1 fitted values (default: False).
        - res_distribution (str): Auxiliary distribution for wild bootstrap ('rademacher' or 'mammen').
        - scaled_residuals (str): Residual transformation ('simple' or 'leverage').
        - boot_tails (str): Tail definition for bootstrap p-value ('one' or 'two').

    Returns
    ----------
    JtestResults
        Dataclass containing test statistics, p-values, and optional series.
    """
    return_residuals: bool = kwargs.get('return_residuals', False)
    return_fitted: bool = kwargs.get('return_fitted', False)
    res_distribution: str = kwargs.get('res_distribution', 'rademacher')
    scaled_residuals: str = kwargs.get('scaled_residuals', 'simple')
    boot_tails: str = kwargs.get('boot_tails', 'one')
    return_t_boot: bool = kwargs.get('return_t_boot', False)
    
    rng = np.random.default_rng(seed)
    # Data preprocessing
    y_f = np.asarray(y, dtype=float).ravel()
    N = len(y_f)
    # Ensure 2D matrices
    X_mat = np.atleast_2d(X).T if np.ndim(X) == 1 else np.asarray(X)
    Z_mat = np.atleast_2d(Z).T if np.ndim(Z) == 1 else np.asarray(Z)

    ones_col = np.ones((N,1)) #Column vector for intercept terms
    # 1. Fit base models
    # Model H1
    A1 = np.column_stack([X_mat, ones_col])
    coeffs1, _, _, _ = np.linalg.lstsq(A1, y_f, rcond=None)
    y1_h =  A1 @ coeffs1
    k1 = A1.shape[1]
    # Model H2
    A2 = np.column_stack([Z_mat, ones_col])
    coeffs2, _, _, _ = np.linalg.lstsq(A2, y_f, rcond=None)
    y2_h =  A2 @ coeffs2
    # 2. Augmented regression
    # Augmented model
    A_aug = np.column_stack([A1, y2_h])
    coeffs_aug, _, _, _ = np.linalg.lstsq(A_aug, y_f, rcond=None)
    res_aug = y_f - (A_aug @ coeffs_aug)
    ssr_aug = np.sum(res_aug ** 2)
    # J-statistic and asymptotic p-value
    n_aug, p_aug = A_aug.shape
    dof_aug = n_aug - p_aug
    dof_aug = N - A_aug.shape[1]
    mse_aug = ssr_aug / dof_aug
    cov_aug = mse_aug * np.linalg.pinv(A_aug.T @ A_aug)
    se_obs = np.sqrt(cov_aug[-1, -1])
    t_stat_obs = float(coeffs_aug[-1] / se_obs)
    p_value_obs = float(stats.t.sf(np.abs(t_stat_obs), dof_aug) * 2)
    
    # 3. Residual scaling and bootstrap setup
    res_y1 = y_f - y1_h
    dof_h1 = N - k1

    if boot_method in ['residuals', 'semiparametric']:
        res_centered = res_y1 - np.mean(res_y1) # Center residuals under H1 assumption
        if scaled_residuals == 'simple':
            res_scaled = res_centered * np.sqrt(N / dof_h1)
        elif scaled_residuals == 'leverage':
            Q, _ = np.linalg.qr(A1)
            leverage = np.sum(Q**2, axis=1)
            res_scaled = res_centered / np.sqrt(1.0 - leverage)
        else:
            raise ValueError('Supported residual scaling are "simple" and "leverage".')

    elif boot_method == 'wild':
        if scaled_residuals == 'simple':
            res_scaled = res_y1 * np.sqrt(N / dof_h1)
        elif scaled_residuals == 'leverage':
            Q, _ = np.linalg.qr(A1)
            leverage = np.sum(Q**2, axis=1)
            res_scaled = res_y1 / np.sqrt(1.0 - leverage)
        else:
            raise ValueError('Supported residual scaling are "simple" and "leverage".')

    elif boot_method == 'parametric':
        res_std = np.sqrt(np.sum(res_y1**2) / dof_h1)

    elif boot_method == 'pairs':
        pass # Pairs bootstrap resamples raw data directly

    else:
        raise ValueError('Supported boot_method: "pairs", "wild", "residuals", "parametric".')

    # 4. Bootstrap Loop
    t_stat_boot = []

    for b in range(B):
        try:
            if boot_method == 'pairs':
                # Resample observations (y, X, Z) simultaneously
                b_idx = rng.choice(N, size=N, replace=True)
                y_b, X_mat_b, Z_mat_b = y_f[b_idx], X_mat[b_idx], Z_mat[b_idx]
                # Fit H2 in bootstrap sample
                A2_b = np.column_stack([Z_mat_b, ones_col])
                coeffs2_b, _, _, _ = np.linalg.lstsq(A2_b, y_b, rcond=None)
                y2_h_b = A2_b @ coeffs2_b
                # Fit augmented model on bootstrap sample
                A_aug_b = np.concatenate([X_mat_b, ones_col, y2_h_b.reshape(-1, 1)], axis=1)
                coeffs_b, res_b, _, _ = np.linalg.lstsq(A_aug_b, y_b, rcond=None)
                # Bootstrap J-statistic
                resid_b = y_b - (A_aug_b @ coeffs_b)
                mse_b = np.sum(resid_b**2) / (A_aug_b.shape[0] - A_aug_b.shape[1])
                cov_b = mse_b * np.linalg.pinv(A_aug_b.T @ A_aug_b)
                se_b = np.sqrt(cov_b[-1, -1])
                t_stat_boot.append((coeffs_b[-1]-coeffs_aug[-1]) / se_b) # Centered for pairs bootstrap

            else:
                # Generate bootstrap samples y_star under H1
                if boot_method == 'wild':
                    if res_distribution == 'rademacher':
                        v = rng.choice([-1.0, 1.0], size=N, replace=True)
                    elif res_distribution == 'mammen':
                        v_vals = [-(np.sqrt(5)-1) / 2, (np.sqrt(5)+1) / 2]
                        v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                        v = rng.choice(v_vals, size=N, replace=True, p=v_probs)
                    else:
                        raise ValueError('Supported distributions are "rademacher" and "mammen"')
                    y_star = y1_h + res_scaled * v

                elif boot_method in ['residuals', 'semiparametric']:
                    b_idx = rng.choice(N, size=N, replace=True)
                    y_star = y1_h + res_scaled[b_idx]

                elif boot_method == 'parametric':
                    y_star = y1_h + rng.normal(loc=0.0, scale=res_std, size=N)
                # Fit H2 on y_star
                coeffs2_b, _, _, _ = np.linalg.lstsq(A2, y_star, rcond=None)
                y2_h_b = A2 @ coeffs2_b
                # Fit augmented model on y_star
                A_aug_b = np.column_stack([A1, y2_h_b])
                coeffs_b, _, _, _ = np.linalg.lstsq(A_aug_b, y_star, rcond=None)
                # Bootstrap J-statistic
                resid_b = y_star - (A_aug_b @ coeffs_b)
                mse_b = np.sum(resid_b**2) / (A_aug_b.shape[0] - A_aug_b.shape[1])
                cov_b = mse_b * np.linalg.pinv(A_aug_b.T @ A_aug_b)
                se_b = np.sqrt(cov_b[-1, -1])
                t_stat_boot.append(coeffs_b[-1] / se_b)

        except (np.linalg.LinAlgError, ValueError):
            continue # Skip collinear or singular iterations
            
    # 5. Bootstrap P-value
    t_boot_arr = np.array(t_stat_boot)
    s_iterations = len(t_boot_arr)

    if boot_tails == 'one':
        p_value_boot = float((1 + np.sum(t_boot_arr >= t_stat_obs)) / (1 + s_iterations))
    elif boot_tails == 'two':
        p_value_boot = float((1 + np.sum(np.abs(t_boot_arr) >= np.abs(t_stat_obs)))/(1 + s_iterations))
    else:
        raise ValueError('boot_tails option must be "one" or "two"')

    # 6. Outputs
    res_arr = res_y1 if return_residuals else None
    fitted_arr = y1_h if return_fitted else None
    t_boot_dist = t_boot_arr if return_t_boot else None
    return JtestResults(
        t_asy = t_stat_obs,
        p_asy = p_value_obs,
        p_boot = p_value_boot,
        s_its = int(s_iterations),
        residuals = res_arr,
        fitted = fitted_arr,
        t_boot=t_boot_dist
    )

# statsmodels.OLS backend
def jtest_sm(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    B: int = 999,
    boot_method: Literal["residuals", "semiparametric", "wild", "parametric", "pairs"] = "residuals",
    seed: Optional[Union[int, np.random.Generator]] = None,
    **kwargs
) -> JtestResults:
    """Execute the Bootstrapped Davidson-MacKinnon J-test using statsmodels.OLS.

    Evaluates primary Model H1 (y on X) against non-nested alternative Model H2 (y on Z).

    Parameters
    ----------
    y : array-like of shape (N,)
        Dependent variable vector.
    X : array-like of shape (N, k1) or (N,)
        Explanatory variables under Model H1.
    Z : array-like of shape (N, k2) or (N,)
        Explanatory variables under Model H2.
    B : int, default=999
        Number of bootstrap replications.
    boot_method : {'residuals', 'semiparametric', 'wild', 'parametric', 'pairs'}, default='residuals'
        Bootstrap Data Generation Process (DGP) method.
    **kwargs : dict
        Optional controls:
        - returnmodelresiduals (bool): Return H1 residuals (default: False).
        - returnfittedvalues (bool): Return H1 fitted values (default: False).
        - res_distribution (str): Auxiliary distribution for wild bootstrap ('rademacher' or 'mammen').
        - scaled_residuals (str): Residual scaling method ('simple' or 'leverage').
        - boot_tails (str): Tail definition for bootstrap p-value ('one' or 'two').

    Returns
    -------
    JtestResults
        Dataclass object containing test statistics, p-values, and optional series.
    """
    return_residuals: bool = kwargs.get('return_residuals', False)
    return_fitted: bool = kwargs.get('return_fitted', False)
    res_distribution: str = kwargs.get('res_distribution', 'rademacher')
    scaled_residuals: str = kwargs.get('scaled_residuals', 'simple')
    boot_tails: str = kwargs.get('boot_tails', 'one')

    rng = np.random.default_rng(seed)
    # Data preprocessing
    y_f = np.asarray(y, dtype=float).ravel()
    N = len(y_f)
    # Ensure 2D matrices
    X_mat = np.atleast_2d(X).T if np.ndim(X) == 1 else np.asarray(X, dtype=float)
    Z_mat = np.atleast_2d(Z).T if np.ndim(Z) == 1 else np.asarray(Z, dtype=float)

    indices = np.arange(N)
    ones_col = np.ones((N, 1))
    # 1. Fit base models
    # Model H1
    A1 = np.concatenate([X_mat, ones_col], axis=1)
    m1 = sm.OLS(y_f, A1).fit()
    y1_h = m1.fittedvalues
    dof_h1 = m1.df_resid
    # Model H2
    A2 = np.concatenate([Z_mat, ones_col], axis=1)
    m2 = sm.OLS(y_f, A2).fit()
    y2_h = m2.fittedvalues
    # Augmented regression
    A_aug = np.concatenate([A1, np.asarray(y2_h).reshape(-1, 1)], axis=1)
    m_aug = sm.OLS(y_f, A_aug).fit()
    # J-statistic and asymptotic p-value
    t_stat_obs = float(m_aug.tvalues[-1])
    p_value_obs = float(m_aug.pvalues[-1])

    # 3. Residual scaling and bootstrap setup
    res_y1 = m1.resid

    if boot_method in ['residuals', 'semiparametric']:
        res_centered = res_y1 - np.mean(res_y1)
        if scaled_residuals == 'simple':
            res_scaled = res_centered * np.sqrt(N / dof_h1)
        elif scaled_residuals == 'leverage':
            leverage = m1.get_influence().hat_matrix_diag # Hat matrix diagonal
            res_scaled = res_centered / np.sqrt(1.0 - leverage)
        else:
            raise ValueError('Supported residual scaling options: "simple" or "leverage".')

    elif boot_method == 'wild':
        if scaled_residuals == 'simple':
            res_scaled = res_y1 * np.sqrt(N / dof_h1)
        elif scaled_residuals == 'leverage':
            leverage = m1.get_influence().hat_matrix_diag
            res_scaled = res_y1 / np.sqrt(1.0 - leverage)
        else:
            raise ValueError('Supported residual scaling options: "simple" or "leverage".')

    elif boot_method == 'parametric':
        res_std = np.sqrt(m1.ssr / dof_h1)

    elif boot_method == 'pairs':
        pass # Pairs bootstrap resamples raw data directly

    else:
        raise ValueError('Invalid boot_method. Supported options: "pairs", "wild", "residuals", "parametric".')
    
    # 4. Bootstrap loop
    t_stat_boot = []

    for b in range(B):
        try:
            if boot_method == 'pairs':
                b_idx = rng.choice(N, size=N, replace=True)
                y_b, X_mat_b, Z_mat_b = y_f[b_idx], X_mat[b_idx], Z_mat[b_idx]

                A2_b = np.concatenate([Z_mat_b, ones_col], axis=1)
                m2_b = sm.OLS(y_b, A2_b).fit()
                y2_h_b = m2_b.fittedvalues

                A_aug_b = np.concatenate([X_mat_b, ones_col, np.asarray(y2_h_b).reshape(-1, 1)], axis=1)
                m_aug_b = sm.OLS(y_b, A_aug_b).fit()

                t_stat_boot.append(m_aug_b.tvalues[-1] - t_stat_obs)

            else:
                if boot_method == 'wild':
                    if res_distribution == 'rademacher':
                        v = rng.choice([-1.0, 1.0], size=N, replace=True)
                    elif res_distribution == 'mammen':
                        v_vals = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
                        v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                        v = rng.choice(v_vals, size=N, replace=True, p=v_probs)
                    else:
                        raise ValueError('Supported wild distributions: "rademacher" or "mammen".')
                    y_star = y1_h + res_scaled * v

                elif boot_method in ['residuals', 'semiparametric']:
                    b_idx = rng.choice(N, size=N, replace=True)
                    y_star = y1_h + res_scaled[b_idx]

                elif boot_method == 'parametric':
                    y_star = y1_h + rng.normal(loc=0.0, scale=res_std, size=N)

                m2_b = sm.OLS(y_star, A2).fit()
                y2_h_b = m2_b.fittedvalues

                A_aug_b = np.concatenate([A1, np.asarray(y2_h_b).reshape(-1, 1)], axis=1)
                m_aug_b = sm.OLS(y_star, A_aug_b).fit()

                t_stat_boot.append(m_aug_b.tvalues[-1])

        except (np.linalg.LinAlgError, ValueError):
            continue

    # 5. Bootstrap p-value
    t_boot_arr = np.array(t_stat_boot)
    s_iterations = len(t_boot_arr)

    if boot_tails == 'one':
        p_value_boot = float((1 + np.sum(t_boot_arr >= t_stat_obs)) / (1 + s_iterations))
    elif boot_tails == 'two':
        p_value_boot = float((1 + np.sum(np.abs(t_boot_arr) >= np.abs(t_stat_obs))) / (1 + s_iterations))
    else:
        raise ValueError('boot_tails option must be "one" or "two".')
    
    # 6. Outputs
    res_arr = np.asarray(res_y1) if return_residuals else None
    fitted_arr = np.asarray(y1_h) if return_fitted else None

    return JtestResults(
        t_asy=t_stat_obs,
        p_asy=p_value_obs,
        p_boot=p_value_boot,
        s_its=int(s_iterations),
        residuals=res_arr,
        fitted=fitted_arr
    )

# SciPy backend (+ statsmodels.OLS and LMFIT)
def jtest_scipy(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    func_X: Callable,
    func_Z: Callable,
    B: int = 999,
    boot_method: Literal["residuals", "semiparametric", "wild", "parametric", "pairs"] = "pairs",
    seed: Optional[Union[int, np.random.Generator]] = None,
    **kwargs: Any
) -> JtestResults:
    """Perform the Bootstrapped Davidson-MacKinnon J-test for linear and non-linear regression models. Uses a SciPy backend.

    Evaluates primary non-linear Model H1 (funx_X) against non-nested alternative
    Model H2 (func_Z) across multiple backend solvers ('Hybrid statsmodels.OLS - SciPy.curve_fit', 'scipy.curve_fit', 'LMFIT').

    Parameters
    ----------
    y : array-like of shape (N,)
        Dependent variable vector.
    X : array-like of shape (N, k1) or (N,)
        Explanatory variables under Model H1.
    Z : array-like of shape (N, k2) or (N,)
        Explanatory variables under Model H2.
    func_X : callable
        Model function for H1:
        - For 'hybrid-ols' & 'scipy': `f(x, *params)` expecting positional arguments.
        - For 'lmfit': `f(x, param1, param2, ...)` expecting named keyword arguments.
    func_Z : callable
        Model function for H2 (same signature requirements as `func_X`).
    B : int, default=999
        Number of bootstrap replications.
    boot_method : {'pairs', 'residuals', 'semiparametric', 'wild', 'parametric'}, default='pairs'
        Bootstrap Data Generation Process (DGP) method.
    jtest_solver : {'hybrid-ols', 'scipy', 'lmfit'}, default='hybrid-ols'
        Computational engine used to estimate the baseline models and test parameter:
        * 'hybrid-ols'   : Hybrid approach. Baseline fits via `scipy.optimize.curve_fit`;
                    auxiliary J-test regression solved via `statsmodels.api.OLS`.
        * 'scipy' : Pure Non-linear Least Squares (NLS) pipeline using
                    `scipy.optimize.curve_fit` for both baseline and augmented models.
        * 'lmfit' : Symbolic, parameter-based optimization pipeline using `lmfit.Model`.

    Other Parameters
    ----------------
    **kwargs : dict
        Solver-specific and general execution controls:

        *Global / Bootstrap Controls:*
        - res_distribution : {'rademacher', 'mammen'}, default='rademacher'
            Auxiliary distribution for wild bootstrap resampling.
        - boot_tails : {'one', 'two'}, default='one'
            Tail definition for calculating the empirical bootstrap p-value.
        - returnmodelresiduals : bool, default=False
            If True, populates `residuals` in the returned `JtestResults`.
        - returnfittedvalues : bool, default=False
            If True, populates `fitted` in the returned `JtestResults`.

        *Backend-Specific Controls:*
        - p0_X : list, np.ndarray, or dict, optional
            Initial parameter values for Model H1:
            - `list`/`np.ndarray` for 'hybrid-ols' and 'scipy' (e.g., `[1.0, 0.5]`)
            - `dict` for 'lmfit' (e.g., `{'alpha': 1.0, 'beta': 0.5}`)
        - p0_Z : list, np.ndarray, or dict, optional
            Initial parameter values for Model H2 (same formatting rules as `p0_X`).
        - curvefitmethod : str, default='lm'
            [*'hybrid-ols' and 'scipy' backends only*] Optimization method passed to
            `scipy.optimize.curve_fit` (e.g., `'lm'`, `'trf'`, `'dogbox'`).

    Returns
    -------
    JtestResults
        Dataclass containing:
        - t_asy (float) : Observed J-test t-statistic.
        - p_asy (float) : Asymptotic p-value based on Student's t-distribution.
        - p_boot (float): Empirical bootstrap p-value across valid iterations.
        - s_its (float) : Number of successful bootstrap iterations completed.
        - residuals (np.ndarray, optional): Null model residuals (if requested).
        - fitted (np.ndarray, optional)   : Null model fitted values (if requested).

    Backend Solver Comparison
    -------------------------
    +--------------+-----------------------+--------------------------+----------------------------+
    | Solver       | Baseline Engine       | J-Test Regression Engine | Parameter Initializer (p0) |
    +==============+=======================+==========================+============================+
    | 'hybrid-ols' | scipy.curve_fit       | statsmodels.OLS          | List / Array               |
    +--------------+-----------------------+--------------------------+----------------------------+
    | 'scipy'      | scipy.curve_fit       | scipy.curve_fit (NLS)    | List / Array               |
    +--------------+-----------------------+--------------------------+----------------------------+
    | 'lmfit'      | lmfit.Model           | lmfit.Model              | Dict ({'param': val})      |
    +--------------+-----------------------+--------------------------+----------------------------+

    Notes
    -----
    1. **'hybrid-ols' Backend**: Fast and robust. It converts the non-linear J-test into a linear augmented regression, avoiding full non-linear convergence issues
       during the bootstrap loop.
    2. **'scipy' Backend**: Preserves the non-linear nature of Model H1 in the augmented regression.
    3. **'lmfit' Backend**: Requires the `lmfit` package. Enables parameter naming,
       bounds, and advanced symbolic constraints, but expects keyword-based callables.
    """
    return_residuals: bool = kwargs.get('return_residuals', False)
    return_fitted: bool = kwargs.get('return_fitted', False)
    jtest_solver: str = kwargs.get('jtest_solver', 'hybrid-ols')
    res_distribution: str = kwargs.get('res_distribution', 'rademacher')
    boot_tails: str = kwargs.get('boot_tails', 'one')

    rng = np.random.default_rng(seed)
    # Data prepocessing
    y_f = np.asarray(y, dtype=np.float64).ravel()
    N = len(y_f)
    X_mat = np.asarray(X, dtype=np.float64)
    Z_mat = np.asarray(Z, dtype=np.float64)

    # Helper functions
    def generate_y_star(
        y_h_null: np.ndarray,
        res_scaled: Optional[np.ndarray], 
        res_std: Optional[float],
        rng_inst: np.random.Generator
    ) -> np.ndarray:
        """Simulate synthetic y* values under the null hypothesis (H1)."""
        if boot_method == 'wild':
            if res_distribution == 'rademacher':
                v = rng_inst.choice([-1.0, 1.0], size=N, replace=True)
            elif res_distribution == 'mammen':
                v_choices = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
                v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                v = rng_inst.choice(v_choices, size=N, replace=True, p=v_probs)
            else:
                raise ValueError('Supported distributions for wild bootstrap: "rademacher" or "mammen".')
            return y_h_null + res_scaled * v

        elif boot_method in ['residuals', 'semiparametric']:
            b_idx = rng.choice(N, size=N, replace=True)
            return y_h_null + res_scaled[b_idx]

        elif boot_method == 'parametric':
            return y_h_null + rng.normal(loc=0.0, scale=res_std, size=N)

        raise ValueError(f"Invalid boot_method: {boot_method}")

    def calc_p_boot(t_stat_boot: list, t_stat_obs: float) -> float:
        """Compute the empirical bootstrap p-value."""
        t_boot_arr = np.array(t_stat_boot)
        s_iterations = len(t_boot_arr)
        if s_iterations == 0:
            return np.nan

        if boot_tails == 'one':
            return float((1 + np.sum(t_boot_arr >= t_stat_obs)) / (1 + s_iterations))
        elif boot_tails == 'two':
            return float((1 + np.sum(np.abs(t_boot_arr) >= np.abs(t_stat_obs))) / (1 + s_iterations))
        else:
            raise ValueError('boot_tails option must be "one" or "two".')

    def format_cf_input(arr: np.ndarray) -> np.ndarray:
        """Format inputs for scipy.optimize.curve_fit."""
        return arr.T if arr.ndim > 1 else arr

    # Hybrid-OLS: Base models fitted via SciPy with statmodels.OLS augmented regression
    if jtest_solver == 'hybrid-ols':
        curvefitmethod = kwargs.get('curvefitmethod', 'lm')
        p0_X = kwargs.get('p0_X', None)
        p0_Z = kwargs.get('p0_Z', None)

        def run_jtest_ols(
            x_null: np.ndarray, x_alt: np.ndarray, f_null: Callable, f_alt: Callable, p0_null: Any, p0_alt: Any
        ) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:

            x_null_cf = format_cf_input(x_null)
            x_alt_cf = format_cf_input(x_alt)
            # Fit base models
            popt_null, _ = curve_fit(f_null, x_null_cf, y_f, p0=p0_null, method=curvefitmethod)
            y_h_null = f_null(x_null_cf, *popt_null)
            res_null = y_f - y_h_null
            dof = N - len(popt_null)

            popt_alt, _ = curve_fit(f_alt, x_alt_cf, y_f, p0=p0_alt, method=curvefitmethod)
            y_h_alt = f_alt(x_alt_cf, *popt_alt)
            # Augmented regression
            X_aug = sm.add_constant(np.column_stack((x_null, y_h_alt)))
            j_model = sm.OLS(y_f, X_aug).fit()
            # J-statistic and asymptotic p-value
            t_stat_obs = float(j_model.tvalues[-1])
            p_obs = float(j_model.pvalues[-1])
            
            # Residual scaling and bootstrap setup
            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null - np.mean(res_null)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null**2) / dof))
                
            # Bootstrap loop
            t_stat_boot = []
            for _ in range(B):
                try:
                    if boot_method == 'pairs':
                        idx = np.random.choice(N, size=N, replace=True)
                        x_null_b = x_null[idx] if x_null.ndim == 1 else x_null[idx, :]
                        x_alt_b = x_alt[idx] if x_alt.ndim == 1 else x_alt[idx, :]
                        y_b = y_f[idx]
                    else:
                        x_null_b, x_alt_b = x_null, x_alt
                        y_b = generate_y_star(y_h_null, res_scaled, res_std)

                    x_null_b_cf = format_cf_input(x_null_b)
                    x_alt_b_cf = format_cf_input(x_alt_b)

                    popt_alt_b, _ = curve_fit(f_alt, x_alt_b_cf, y_b, p0=popt_alt, method=curvefitmethod)
                    y_h_alt_b = f_alt(x_alt_b_cf, *popt_alt_b)

                    X_aug_b = sm.add_constant(np.column_stack((x_null_b, y_h_alt_b)))
                    j_model_b = sm.OLS(y_b, X_aug_b).fit()

                    t_b = float(j_model_b.tvalues[-1])
                    t_stat_boot.append(t_b - t_stat_obs if boot_method == 'pairs' else t_b)
                except (RuntimeError, np.linalg.LinAlgError, ValueError):
                    continue

            p_boot = calc_p_boot(t_stat_boot, t_stat_obs)
            return t_stat_obs, p_obs, p_boot, len(t_stat_boot), res_null, y_h_null

        res_m = run_jtest_ols(X_mat, Z_mat, func_X, func_Z, p0_X, p0_Z)

    # SciPy: Uses SciPy for both base models and augmented regression
    elif jtest_solver == 'scipy':
        curvefitmethod = kwargs.get('curvefitmethod', 'lm')
        p0_X = kwargs.get('p0_X', None)
        p0_Z = kwargs.get('p0_Z', None)

        def make_augmented_model(base_model_func: Callable, competitor_fitted: np.ndarray) -> Callable:
            return lambda x, *terms: base_model_func(x, *terms[:-1]) + terms[-1] * competitor_fitted

        def run_jtest_scipy(
            x_null: np.ndarray, x_alt: np.ndarray, f_null: Callable, f_alt: Callable, p0_null: Any, p0_alt: Any
        ) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:

            x_null_cf = format_cf_input(x_null)
            x_alt_cf = format_cf_input(x_alt)
            # Fit base models
            popt_null, _ = curve_fit(f_null, x_null_cf, y_f, p0=p0_null, method=curvefitmethod)
            y_h_null = f_null(x_null_cf, *popt_null)
            res_null = y_f - y_h_null
            dof = N - len(popt_null)

            popt_alt, _ = curve_fit(f_alt, x_alt_cf, y_f, p0=p0_alt, method=curvefitmethod)
            y_h_alt = f_alt(x_alt_cf, *popt_alt)
            # Fit augmented model
            model_aug = make_augmented_model(f_null, y_h_alt)
            popt_aug, pcov_aug = curve_fit(model_aug, x_null_cf, y_f, p0=[*popt_null, 0.0], method=curvefitmethod)
            # J-statistic and asymptotic p-value
            t_stat_obs = float(popt_aug[-1] / np.sqrt(pcov_aug[-1, -1]))
            p_obs = float(stats.t.sf(np.abs(t_stat_obs), df=dof - 1) * 2)
            
            #Residual scaling and bootstrap setup
            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null - np.mean(res_null)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null**2) / dof))
            
            #Bootstrap loop 
            t_stat_boot = []
            for _ in range(B):
                try:
                    if boot_method == 'pairs':
                        idx = rng.choice(N, size=N, replace=True)
                        x_null_b = x_null[idx] if x_null.ndim == 1 else x_null[idx, :]
                        x_alt_b = x_alt[idx] if x_alt.ndim == 1 else x_alt[idx, :]
                        y_b = y_f[idx]
                    else:
                        x_null_b, x_alt_b = x_null, x_alt
                        y_b = generate_y_star(y_h_null, res_scaled, res_std, rng)

                    x_null_b_cf = format_cf_input(x_null_b)
                    x_alt_b_cf = format_cf_input(x_alt_b)

                    popt_alt_b, _ = curve_fit(f_alt, x_alt_b_cf, y_b, p0=popt_alt, method=curvefitmethod)
                    y_h_alt_b = f_alt(x_alt_b_cf, *popt_alt_b)

                    model_aug_b = make_augmented_model(f_null, y_h_alt_b)
                    popt_aug_b, pcov_aug_b = curve_fit(model_aug_b, x_null_b_cf, y_b, p0=[*popt_null, 0.0], method=curvefitmethod)

                    t_b = float(popt_aug_b[-1] / np.sqrt(pcov_aug_b[-1, -1]))
                    t_stat_boot.append(t_b - t_stat_obs if boot_method == 'pairs' else t_b)
                except (RuntimeError, ValueError):
                    continue

            p_boot = calc_p_boot(t_stat_boot, t_stat_obs)
            return t_stat_obs, p_obs, p_boot, len(t_stat_boot), res_null, y_h_null

        res_m = run_jtest_scipy(X_mat, Z_mat, func_X, func_Z, p0_X, p0_Z)

    # LMFIT: Base models and augmented regressions via lmfit.Model
    elif jtest_solver == 'lmfit':
        if not HAS_LMFIT:
            raise ImportError("The 'lmfit' package is required for jtest_solver='lmfit'. Install via 'pip install lmfit'.")

        p0_X: Dict[str, float] = kwargs.get('p0_X', {})
        p0_Z: Dict[str, float] = kwargs.get('p0_Z', {})

        def run_jtest_lmfit(
            x_null: np.ndarray, x_alt: np.ndarray, f_null: Callable, f_alt: Callable, p0_null: dict, p0_alt: dict
        ) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:

            x_null_cf = format_cf_input(x_null)
            x_alt_cf = format_cf_input(x_alt)
            
            # Define lmfit models
            model_null, model_alt = Model(f_null), Model(f_alt)
            params_null, params_alt = model_null.make_params(), model_alt.make_params()

            for k, v in p0_null.items():
                if k in params_null: params_null[k].set(value=v)
            for k, v in p0_alt.items():
                if k in params_alt: params_alt[k].set(value=v)
            
            #Fit base models
            res_null = model_null.fit(y_f, x=x_null_cf, params=params_null)
            res_alt = model_alt.fit(y_f, x=x_alt_cf, params=params_alt)

            y_h_null = res_null.best_fit
            res_null_arr = res_null.residual
            dof = N - res_null.nvarys
            comp_fit_vals = res_alt.best_fit
            # Augmented model
            def make_lmfit_augmented(base_func: Callable, fitted_alt: np.ndarray) -> Callable:
                return lambda x, lam=0.0, **params: base_func(x, **params) + lam * fitted_alt
            # Fit augmented model
            aug_func = make_lmfit_augmented(f_null, comp_fit_vals)
            model_aug = Model(aug_func)
            aug_params = res_null.params.copy()
            aug_params.add('lam', value=0.0)
            # J-statistic and asymptotic p-value
            res_aug = model_aug.fit(y_f, x=x_null_cf, params=aug_params)
            lam_se = res_aug.params['lam'].stderr
            t_stat_obs = float(res_aug.params['lam'].value / lam_se) if lam_se else np.nan
            p_obs = float(stats.t.sf(np.abs(t_stat_obs), df=res_aug.df_intrinsic) * 2) if not np.isnan(t_stat_obs) else 1.0

            # Residual scaling and bootstrap setup
            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null_arr - np.mean(res_null_arr)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null_arr * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null_arr**2) / dof))

            # Bootstrap loop
            t_stat_boot = []
            for _ in range(B):
                try:
                    if boot_method == 'pairs':
                        idx = rng.choice(N, size=N, replace=True)
                        x_null_b = x_null[idx] if x_null.ndim == 1 else x_null[idx, :]
                        x_alt_b = x_alt[idx] if x_alt.ndim == 1 else x_alt[idx, :]
                        y_b = y_f[idx]
                    else:
                        x_null_b, x_alt_b = x_null, x_alt
                        y_b = generate_y_star(y_h_null, res_scaled, res_std)

                    x_null_b_cf = format_cf_input(x_null_b)
                    x_alt_b_cf = format_cf_input(x_alt_b)

                    res_alt_b = model_alt.fit(y_b, x=x_alt_b_cf, params=res_alt.params.copy())
                    aug_func_b = make_lmfit_augmented(f_null, res_alt_b.best_fit)

                    model_aug_b = Model(aug_func_b)
                    aug_params_b = res_null.params.copy()
                    aug_params_b.add('lam', value=0.0)

                    res_aug_b = model_aug_b.fit(y_b, x=x_null_b_cf, params=aug_params_b)
                    lam_se_b = res_aug_b.params['lam'].stderr
                    t_b = float(res_aug_b.params['lam'].value / lam_se_b) if lam_se_b else np.nan

                    if not np.isnan(t_b):
                        t_stat_boot.append(t_b - t_stat_obs if boot_method == 'pairs' else t_b)
                except Exception:
                    continue

            p_boot = calc_p_boot(t_stat_boot, t_stat_obs)
            return t_stat_obs, p_obs, p_boot, len(t_stat_boot), res_null_arr, y_h_null

        res_m = run_jtest_lmfit(X_mat, Z_mat, func_X, func_Z, p0_X, p0_Z)

    else:
        raise ValueError(f"Unknown solver '{jtest_solver}'. Supported options: 'hybrid-ols', 'scipy', 'lmfit'.")

    # Outputs
    t_stat_obs, p_obs, p_boot, s_its, res_null, y_h_null = res_m

    return JtestResults(
        t_asy=float(t_stat_obs),
        p_asy=float(p_obs),
        p_boot=float(p_boot),
        s_its=int(s_its),
        residuals=np.asarray(res_null) if return_residuals else None,
        fitted=np.asarray(y_h_null) if return_fitted else None
    )

