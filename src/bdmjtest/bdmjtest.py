from dataclasses import dataclass
from typing import Optional, Literal, Callable, Any, Dict, Tuple
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
        boot_method: Literal["residuals", "semiparametric", "wild", "parametric", "pairs"] = "residuals",
        **kwargs
) -> JtestResults:

    return_residuals: bool = kwargs.get('return_residuals', False)
    return_fitted: bool = kwargs.get('return_fitted', False)
    res_distribution: str = kwargs.get('res_distribution', 'rademacher')
    scaled_residuals: str = kwargs.get('scaled_residuals', 'simple')
    boot_tails: str = kwargs.get('boot_tails', 'one')
    return_t_boot: bool = kwargs.get('return_t_boot', False)

    y_f = np.asarray(y).ravel()
    N = len(y_f)
    
    X_mat = np.atleast_2d(X).T if np.ndim(X) == 1 else np.asarray(X)
    Z_mat = np.atleast_2d(Z).T if np.ndim(Z) == 1 else np.asarray(Z)

    indices = np.arange(len(y_f))
    ones_col = np.ones((N,1))

    A1 = np.column_stack([X_mat, ones_col])
    coeffs1, _, _, _ = np.linalg.lstsq(A1, y_f, rcond=None)
    y1_h =  A1 @ coeffs1
    k1 = A1.shape[1]

    A2 = np.column_stack([Z_mat, ones_col])
    coeffs2, _, _, _ = np.linalg.lstsq(A2, y_f, rcond=None)
    y2_h =  A2 @ coeffs2

    A_aug = np.column_stack([A1, y2_h])
    coeffs_aug, _, _, _ = np.linalg.lstsq(A_aug, y_f, rcond=None)
    res_aug = y_f - (A_aug @ coeffs_aug)
    ssr_aug = np.sum(res_aug ** 2)

    n_aug, p_aug = A_aug.shape
    dof_aug = n_aug - p_aug

    dof_aug = len(y_f) - A_aug.shape[1]
    mse_aug = ssr_aug / dof_aug
    cov_aug = mse_aug * np.linalg.pinv(A_aug.T @ A_aug)
    se_obs = np.sqrt(cov_aug[-1, -1])

    t_stat_obs = float(coeffs_aug[-1] / se_obs)
    p_value_obs = float(stats.t.sf(np.abs(t_stat_obs), dof_aug) * 2)

    res_y1 = y_f - y1_h
    dof_h1 = N - k1

    if boot_method in ['residuals', 'semiparametric']:
        res_centered = res_y1 - np.mean(res_y1)
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
        pass

    else:
        raise ValueError('Supported boot_method: "pairs", "wild", "residuals", "parametric".')

    t_stat_boot = []

    for b in range(B):
        try:
            if boot_method == 'pairs':
                
                b_idx = np.random.choice(indices, size=N, replace=True)
                y_b, X_mat_b, Z_mat_b = y_f[b_idx], X_mat[b_idx], Z_mat[b_idx]
                
                A2_b = np.column_stack([Z_mat_b, ones_col])
                coeffs2_b, _, _, _ = np.linalg.lstsq(A2_b, y_b, rcond=None)
                y2_h_b = A2_b @ coeffs2_b
                
                A_aug_b = np.concatenate([X_mat_b, ones_col, y2_h_b.reshape(-1, 1)], axis=1)
                coeffs_b, res_b, _, _ = np.linalg.lstsq(A_aug_b, y_b, rcond=None)
                
                resid_b = y_b - (A_aug_b @ coeffs_b)
                mse_b = np.sum(resid_b**2) / (A_aug_b.shape[0] - A_aug_b.shape[1])
                cov_b = mse_b * np.linalg.pinv(A_aug_b.T @ A_aug_b)
                se_b = np.sqrt(cov_b[-1, -1])
                
                t_stat_boot.append((coeffs_b[-1]-coeffs_aug[-1]) / se_b)

            else:
                if boot_method == 'wild':
                    if res_distribution == 'rademacher':
                        v = np.random.choice([-1.0, 1.0], size=N, replace=True)
                    elif res_distribution == 'mammen':
                        v_vals = [-(np.sqrt(5)-1) / 2, (np.sqrt(5)+1) / 2]
                        v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                        v = np.random.choice(v_vals, size=N, replace=True, p=v_probs)
                    else:
                        raise ValueError('Supported distributions are "rademacher" and "mammen"')
                    y_star = y1_h + res_scaled * v

                elif boot_method in ['residuals', 'semiparametric']:
                    b_idx = np.random.choice(indices, size=N, replace=True)
                    y_star = y1_h + res_scaled[b_idx]

                elif boot_method == 'parametric':
                    y_star = y1_h + np.random.normal(0, res_std, size=N)

                coeffs2_b, _, _, _ = np.linalg.lstsq(A2, y_star, rcond=None)
                y2_h_b = A2 @ coeffs2_b
              
                A_aug_b = np.column_stack([A1, y2_h_b])
                coeffs_b, _, _, _ = np.linalg.lstsq(A_aug_b, y_star, rcond=None)

                resid_b = y_star - (A_aug_b @ coeffs_b)
                mse_b = np.sum(resid_b**2) / (A_aug_b.shape[0] - A_aug_b.shape[1])
                cov_b = mse_b * np.linalg.pinv(A_aug_b.T @ A_aug_b)
                se_b = np.sqrt(cov_b[-1, -1])
                t_stat_boot.append(coeffs_b[-1] / se_b)

        except (np.linalg.LinAlgError, ValueError):
            continue

    t_boot_arr = np.array(t_stat_boot)
    s_iterations = len(t_boot_arr)

    if boot_tails == 'one':
        p_value_boot = float((1 + np.sum(t_boot_arr >= t_stat_obs)) / (1 + s_iterations))
    elif boot_tails == 'two':
        p_value_boot = float((1 + np.sum(np.abs(t_boot_arr) >= np.abs(t_stat_obs)))/(1 + s_iterations))
    else:
        raise ValueError('boot_tails option must be "one" or "two"')

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
    **kwargs
) -> JtestResults:
    
    return_residuals: bool = kwargs.get('return_residuals', False)
    return_fitted: bool = kwargs.get('return_fitted', False)
    res_distribution: str = kwargs.get('res_distribution', 'rademacher')
    scaled_residuals: str = kwargs.get('scaled_residuals', 'simple')
    boot_tails: str = kwargs.get('boot_tails', 'one')

    y_f = np.asarray(y, dtype=float).ravel()
    N = len(y_f)

    X_mat = np.atleast_2d(X).T if np.ndim(X) == 1 else np.asarray(X, dtype=float)
    Z_mat = np.atleast_2d(Z).T if np.ndim(Z) == 1 else np.asarray(Z, dtype=float)

    indices = np.arange(N)
    ones_col = np.ones((N, 1))

    A1 = np.concatenate([X_mat, ones_col], axis=1)
    m1 = sm.OLS(y_f, A1).fit()
    y1_h = m1.fittedvalues
    dof_h1 = m1.df_resid

    A2 = np.concatenate([Z_mat, ones_col], axis=1)
    m2 = sm.OLS(y_f, A2).fit()
    y2_h = m2.fittedvalues

    A_aug = np.concatenate([A1, np.asarray(y2_h).reshape(-1, 1)], axis=1)
    m_aug = sm.OLS(y_f, A_aug).fit()

    t_stat_obs = float(m_aug.tvalues[-1])
    p_value_obs = float(m_aug.pvalues[-1])

    res_y1 = m1.resid

    if boot_method in ['residuals', 'semiparametric']:
        res_centered = res_y1 - np.mean(res_y1)
        if scaled_residuals == 'simple':
            res_scaled = res_centered * np.sqrt(N / dof_h1)
        elif scaled_residuals == 'leverage':
            leverage = m1.get_influence().hat_matrix_diag
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
        pass

    else:
        raise ValueError('Invalid boot_method. Supported options: "pairs", "wild", "residuals", "parametric".')

    t_stat_boot = []

    for b in range(B):
        try:
            if boot_method == 'pairs':
                b_idx = np.random.choice(indices, size=N, replace=True)
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
                        v = np.random.choice([-1.0, 1.0], size=N, replace=True)
                    elif res_distribution == 'mammen':
                        v_vals = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
                        v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                        v = np.random.choice(v_vals, size=N, replace=True, p=v_probs)
                    else:
                        raise ValueError('Supported wild distributions: "rademacher" or "mammen".')
                    y_star = y1_h + res_scaled * v

                elif boot_method in ['residuals', 'semiparametric']:
                    b_idx = np.random.choice(indices, size=N, replace=True)
                    y_star = y1_h + res_scaled[b_idx]

                elif boot_method == 'parametric':
                    y_star = y1_h + np.random.normal(0, res_std, size=N)

                m2_b = sm.OLS(y_star, A2).fit()
                y2_h_b = m2_b.fittedvalues

                A_aug_b = np.concatenate([A1, np.asarray(y2_h_b).reshape(-1, 1)], axis=1)
                m_aug_b = sm.OLS(y_star, A_aug_b).fit()

                t_stat_boot.append(m_aug_b.tvalues[-1])

        except (np.linalg.LinAlgError, ValueError):
            continue

    t_boot_arr = np.array(t_stat_boot)
    s_iterations = len(t_boot_arr)

    if boot_tails == 'one':
        p_value_boot = float((1 + np.sum(t_boot_arr >= t_stat_obs)) / (1 + s_iterations))
    elif boot_tails == 'two':
        p_value_boot = float((1 + np.sum(np.abs(t_boot_arr) >= np.abs(t_stat_obs))) / (1 + s_iterations))
    else:
        raise ValueError('boot_tails option must be "one" or "two".')

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

def jtest_scipy(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    func_X: Callable,
    func_Z: Callable,
    B: int = 999,
    boot_method: Literal["residuals", "semiparametric", "wild", "parametric", "pairs"] = "pairs",
    **kwargs: Any
) -> JtestResults:

    return_residuals: bool = kwargs.get('return_residuals', False)
    return_fitted: bool = kwargs.get('return_fitted', False)
    jtest_solver: str = kwargs.get('jtest_solver', 'hybrid-ols')
    res_distribution: str = kwargs.get('res_distribution', 'rademacher')
    boot_tails: str = kwargs.get('boot_tails', 'one')

    y_f = np.asarray(y, dtype=np.float64).ravel()
    N = len(y_f)
    X_mat = np.asarray(X, dtype=np.float64)
    Z_mat = np.asarray(Z, dtype=np.float64)

    def generate_y_star(y_h_null: np.ndarray, res_scaled: Optional[np.ndarray], res_std: Optional[float]) -> np.ndarray:
        """Simulate synthetic y* values under the null hypothesis (H1)."""
        if boot_method == 'wild':
            if res_distribution == 'rademacher':
                v = np.random.choice([-1.0, 1.0], size=N, replace=True)
            elif res_distribution == 'mammen':
                v_choices = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
                v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                v = np.random.choice(v_choices, size=N, replace=True, p=v_probs)
            else:
                raise ValueError('Supported distributions for wild bootstrap: "rademacher" or "mammen".')
            return y_h_null + res_scaled * v

        elif boot_method in ['residuals', 'semiparametric']:
            b_indices = np.random.choice(N, size=N, replace=True)
            return y_h_null + res_scaled[b_indices]

        elif boot_method == 'parametric':
            return y_h_null + np.random.normal(0, res_std, size=N)

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

    # Hybrid-OLS
    if jtest_solver == 'hybrid-ols':
        curvefitmethod = kwargs.get('curvefitmethod', 'lm')
        p0_X = kwargs.get('p0_X', None)
        p0_Z = kwargs.get('p0_Z', None)

        def run_jtest_ols(
            x_null: np.ndarray, x_alt: np.ndarray, f_null: Callable, f_alt: Callable, p0_null: Any, p0_alt: Any
        ) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:

            x_null_cf = format_cf_input(x_null)
            x_alt_cf = format_cf_input(x_alt)

            popt_null, _ = curve_fit(f_null, x_null_cf, y_f, p0=p0_null, method=curvefitmethod)
            y_h_null = f_null(x_null_cf, *popt_null)
            res_null = y_f - y_h_null
            dof = N - len(popt_null)

            popt_alt, _ = curve_fit(f_alt, x_alt_cf, y_f, p0=p0_alt, method=curvefitmethod)
            y_h_alt = f_alt(x_alt_cf, *popt_alt)

            X_aug = sm.add_constant(np.column_stack((x_null, y_h_alt)))
            j_model = sm.OLS(y_f, X_aug).fit()
            t_stat_obs = float(j_model.tvalues[-1])
            p_obs = float(j_model.pvalues[-1])

            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null - np.mean(res_null)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null**2) / dof))

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

    # SciPy
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

            popt_null, _ = curve_fit(f_null, x_null_cf, y_f, p0=p0_null, method=curvefitmethod)
            y_h_null = f_null(x_null_cf, *popt_null)
            res_null = y_f - y_h_null
            dof = N - len(popt_null)

            popt_alt, _ = curve_fit(f_alt, x_alt_cf, y_f, p0=p0_alt, method=curvefitmethod)
            y_h_alt = f_alt(x_alt_cf, *popt_alt)

            model_aug = make_augmented_model(f_null, y_h_alt)
            popt_aug, pcov_aug = curve_fit(model_aug, x_null_cf, y_f, p0=[*popt_null, 0.0], method=curvefitmethod)

            t_stat_obs = float(popt_aug[-1] / np.sqrt(pcov_aug[-1, -1]))
            p_obs = float(stats.t.sf(np.abs(t_stat_obs), df=dof - 1) * 2)

            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null - np.mean(res_null)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null**2) / dof))

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

                    model_aug_b = make_augmented_model(f_null, y_h_alt_b)
                    popt_aug_b, pcov_aug_b = curve_fit(model_aug_b, x_null_b_cf, y_b, p0=[*popt_null, 0.0], method=curvefitmethod)

                    t_b = float(popt_aug_b[-1] / np.sqrt(pcov_aug_b[-1, -1]))
                    t_stat_boot.append(t_b - t_stat_obs if boot_method == 'pairs' else t_b)
                except (RuntimeError, ValueError):
                    continue

            p_boot = calc_p_boot(t_stat_boot, t_stat_obs)
            return t_stat_obs, p_obs, p_boot, len(t_stat_boot), res_null, y_h_null

        res_m = run_jtest_scipy(X_mat, Z_mat, func_X, func_Z, p0_X, p0_Z)

    # LMFIT
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

            model_null, model_alt = Model(f_null), Model(f_alt)
            params_null, params_alt = model_null.make_params(), model_alt.make_params()

            for k, v in p0_null.items():
                if k in params_null: params_null[k].set(value=v)
            for k, v in p0_alt.items():
                if k in params_alt: params_alt[k].set(value=v)

            res_null = model_null.fit(y_f, x=x_null_cf, params=params_null)
            res_alt = model_alt.fit(y_f, x=x_alt_cf, params=params_alt)

            y_h_null = res_null.best_fit
            res_null_arr = res_null.residual
            dof = N - res_null.nvarys
            comp_fit_vals = res_alt.best_fit

            def make_lmfit_augmented(base_func: Callable, fitted_alt: np.ndarray) -> Callable:
                return lambda x, lam=0.0, **params: base_func(x, **params) + lam * fitted_alt

            aug_func = make_lmfit_augmented(f_null, comp_fit_vals)
            model_aug = Model(aug_func)
            aug_params = res_null.params.copy()
            aug_params.add('lam', value=0.0)

            res_aug = model_aug.fit(y_f, x=x_null_cf, params=aug_params)
            lam_se = res_aug.params['lam'].stderr

            t_stat_obs = float(res_aug.params['lam'].value / lam_se) if lam_se else np.nan
            p_obs = float(stats.t.sf(np.abs(t_stat_obs), df=res_aug.df_intrinsic) * 2) if not np.isnan(t_stat_obs) else 1.0

            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null_arr - np.mean(res_null_arr)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null_arr * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null_arr**2) / dof))

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

    t_stat_obs, p_obs, p_boot, s_its, res_null, y_h_null = res_m

    return JtestResults(
        t_asy=float(t_stat_obs),
        p_asy=float(p_obs),
        p_boot=float(p_boot),
        s_its=int(s_its),
        residuals=np.asarray(res_null) if return_residuals else None,
        fitted=np.asarray(y_h_null) if return_fitted else None
    )

