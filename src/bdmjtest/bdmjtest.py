from dataclasses import dataclass
import warnings
from typing import Union, Optional, Literal, Callable, Any, Dict, Tuple
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
# Optional import for lmfit solver
try:
    from lmfit import Model
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False

@dataclass
class JtestResults:
    """
    Container class for the outputs of the pairwise Bootstrapped Davidson-MacKinnon J-test: H1 (X)  vs. H2 (Z), and vice versa.

    Attributes
    ----------
    t_asy_X, t_asy_Z : float
        Asymptotic J-test t-statistics computed from original sample data for each pairwise test.
    p_asy_X, p_asy_Z : float
        Asymptotic p-values based on Student's t-distribution for each pairwise test.
    p_boot_X, p_boot_Z : float
        Bootstrap p-values calculated across successful iterations for each pairwise test.
    s_its_X, s_its_Z : float
        Total number of successfully executed bootstraps for each pairwise test.
    residuals_X, residuals_Z : np.ndarray, optional
        Residuals from H1 and H2, respectively, returned if requested.
    fitted_X, fitted_Z : np.ndarray, optional
        Fitted values from H1 and H2, respectively, returned if requested.
    t_boot_X, t_boot_Z : np.ndarray, optional
        Bootstrap t-statistics distributions, returned if requested.
    """
    # Primary variables

    # Model H1 as null Hypothesis (H1 vs H2)
    t_asy_X : float = np.nan
    p_asy_X : float = np.nan
    p_boot_X : float = np.nan
    s_its_X : int = 0

    # Model H2 as null Hypothesis (H2 vs H1)
    t_asy_Z : float = np.nan
    p_asy_Z : float = np.nan
    p_boot_Z : float = np.nan
    s_its_Z : int = 0

    # Optional variables
    residuals_X: Optional[np.ndarray] = None
    residuals_Z: Optional[np.ndarray] = None
    fitted_X: Optional[np.ndarray] = None
    fitted_Z: Optional[np.ndarray] = None
    t_boot_X: Optional[np.ndarray] = None
    t_boot_Z: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        if np.isnan(self.t_asy_Z):
            return (
                f"JtestResults(\n"
                f"  H1 vs. H2: | t-stat: {self.t_asy_X:.4f} | p-asy: {self.p_asy_X:.4f} | p-boot: {self.p_boot_X:.4f}\n"
                f"  Valid Iterations: {self.s_its_X}\n"
                f")"
            )
        return (
                f"JtestResults(\n"
                f"  H1 vs. H2: | t-stat: {self.t_asy_X:.4f} | p-asy: {self.p_asy_X:.4f} | p-boot: {self.p_boot_X:.4f}\n"
                f"  H2 vs. H1: | t-stat: {self.t_asy_Z:.4f} | p-asy: {self.p_asy_Z:.4f} | p-boot: {self.p_boot_Z:.4f}\n"
                f"  Valid Iterations (X / Z): {self.s_its_X} / {self.s_its_Z}\n"
                f")"
            )

    def summary(self, digits: int = 4) -> str:
        out = (
            f"===============================================\n"
            f"J-Test Results \n"
            f"===============================================\n"
            f"  H1 vs H2\n"
            f"      Asymptotic t-statistic : {self.t_asy_X:.{digits}f}\n"
            f"      Asymptotic p-value     : {self.p_asy_X:.{digits}f}\n"
            f"      Bootstrap p-value      : {self.p_boot_X:.{digits}f} (Valid draws: {self.s_its_X})\n"
        )
        if not np.isnan(self.t_asy_Z):
            out += (
                f"------------------------------------------------\n"
                f"  H2 vs. H1\n"
                f"      Asymptotic t-statistic : {self.t_asy_Z:.{digits}f}\n"
                f"      Asymptotic p-value     : {self.p_asy_Z:.{digits}f}\n"
                f"      Bootstrap p-value      : {self.p_boot_Z:.{digits}f} (Valid draws: {self.s_its_Z})\n"
            )
        out += "================================================"
        return out

def _calc_aug_reg(
        X_mat: np.ndarray,
        y_vec: np.ndarray,
        y_h_alt: np.ndarray,
        solver: str = 'qr'
) -> float:
    """
    Helper function to fit the augmented regression model and return the t-statistic.
    """
    A_aug = np.column_stack((X_mat, y_h_alt))
    N, K = A_aug.shape

    if solver == 'qr':
        Q, R = np.linalg.qr(A_aug)
        r_kk = R[-1, -1]

        if np.min(np.abs(np.diag(R))) < 1e-12 * np.max(np.abs(np.diag(R))):
            solver = 'lstsq'
        else:
            beta_aug = np.linalg.solve(R, Q.T @ y_vec)
            res_aug = y_vec - A_aug @ beta_aug
            ssr = float(res_aug @ res_aug)
            sigma2 = ssr / (N - K)
            se = np.sqrt(sigma2) / abs(r_kk)

            if np.isnan(se) or se == 0:
                return np.nan

            return float(beta_aug[-1] / se)

    if solver == 'lstsq':
        coeffs, _, rank_a, _ = np.linalg.lstsq(A_aug, y_vec, rcond=None)
        res_aug = y_vec - (A_aug @ coeffs)
        ssr = float(res_aug @ res_aug)
        dof = N - rank_a
        sigma2 = ssr / dof if dof > 0 else np.inf
        cov_aug = sigma2 * np.linalg.pinv(A_aug.T @ A_aug)
        se = np.sqrt(np.abs(cov_aug[-1, -1]))

        if np.isnan(se) or se == 0:
            return np.nan
        return float(coeffs[-1] / se)

    else:
        raise ValueError(f"Unknown solver: {solver}. Options are 'qr' and 'lstsq'.")

def _setup_residual_scaling(
        res: np.ndarray,
        N: int,
        dof: int,
        Q_mat: Optional[np.ndarray],
        A_mat: np.ndarray,
        boot_method: str,
        scaled_residuals: str,
        solver: str
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Helper function to perform bootstrap scaling
    """
    if boot_method in ['residuals', 'semiparametric']:
        res_centered = res - np.mean(res)
        if scaled_residuals == 'simple':
            return res_centered * np.sqrt(N / dof), None
        elif scaled_residuals == 'leverage':
            if solver == "qr" and Q_mat is not None:
                leverage = np.sum(Q_mat ** 2, axis=1)
            else:
                H = A_mat @ np.linalg.pinv(A_mat.T @ A_mat) @ A_mat.T
                leverage = np.diag(H)
            return res_centered / np.sqrt(1.0 - leverage), None
        else:
            raise ValueError("Supported residual scaling options are 'simple' and 'leverage'.")

    elif boot_method == 'wild':
        if scaled_residuals == 'simple':
            return res * np.sqrt(N / dof), None
        elif scaled_residuals == 'leverage':
            if solver == "qr" and Q_mat is not None:
                leverage = np.sum(Q_mat ** 2, axis=1)
            else:
                H = A_mat @ np.linalg.pinv(A_mat.T @ A_mat) @ A_mat.T
                leverage = np.diag(H)
            return res / np.sqrt(1.0 - leverage), None
        else:
            raise ValueError("Supported residual scaling options are 'simple' and 'leverage'.")

    elif boot_method == 'parametric':
        return None, float(np.sqrt(np.sum(res ** 2) / dof))

    return None, None

# NumPy backend
def jtest_np(
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        B: int = 999,
        boot_method: Literal['residuals', 'semiparametric', 'wild', 'parametric', 'pairs'] = 'residuals',
        direction: Literal['pairwise', 'one-way'] = 'pairwise',
        solver: Literal['qr', 'lstsq'] = 'qr',
        seed: Optional[Union[int, np.random.Generator]] = None,
        **kwargs
) -> JtestResults:
    """Performs the Bootstrap Davidson-MacKinnon J-test with NumPy backend.

    Test a linear null model (H1) (y on X) against a linear non-nested alternative (H2) (y on Z), and vice versa.

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
    boot_method : {'residuals' ('semiparametric') 'wild', 'parametric', 'pairs'}, default='residuals'
        Bootstrap Data Generation Process (DGP) method.
    direction : {'pairwise', 'one-way'}, default='pairwise'
        If 'pairwise', evaluates both directions (H1 vs H2 and H2 vs H1).
        If 'one-way', evaluates only H1 vs H2.
    solver: {'qr', 'lstsq'}, default='qr'
        Numerical solver engine:
        - 'qr' : Householder QR decomposition for full-rank matrices
        - 'lstsq' : Singular Value Decomposition (SVD) based least squares for rank-deficient matrices
    seed : int, default=None
        Random seed for reproducible bootstrap sampling.
    **kwargs : dict
        Optional controls:
        - return_residuals (bool): Return H1 residuals (default: False).
        - return_fitted (bool): Return H1 fitted values (default: False).
        - res_distribution (str): Auxiliary distribution for wild bootstrap ('rademacher', 'mammen', 'webb', 'normal').
        - scaled_residuals (str): Residual transformation ('simple' or 'leverage').
        - boot_tails (str): Tail definition for bootstrap p-value ('one' or 'two').
        - return_t_boot (bool): Return bootstrap t-statistics distributions (default: False).

    Returns
    ----------
    JtestResults
        Dataclass containing test statistics, p-values, and optional series.
    """

    if B <= 0:
        raise ValueError("B must be a strictly positive integer")
    elif B < 399:
        min_pval = 1.0 / (B + 1)
        warnings.warn(
            f"B={B} yields a minimum p-value of {min_pval:.4f} and may produce unstable results. B >= 999 is recommended for statistical stability.",
            UserWarning
        )
    if boot_method not in ['residuals', 'semiparametric', 'wild', 'parametric', 'pairs']:
        raise ValueError(f"Invalid boot method: {boot_method}")

    if solver not in ('qr', 'lstsq'):
        raise ValueError(f"Unknown solver: {solver}. Options are 'qr' and 'lstsq'.")

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

    ones_col = np.ones((N, 1))  # Column vector for intercept terms
    A1 = np.column_stack([X_mat, ones_col])
    A2 = np.column_stack([Z_mat, ones_col])

    # 1. Fit base models
    if solver == 'qr':
        # Model H1
        Q1, _ = np.linalg.qr(A1, mode='reduced')
        y1_h = Q1 @ (Q1.T @ y_f)
        # Model H2
        Q2, _ = np.linalg.qr(A2, mode='reduced')
        y2_h = Q2 @ (Q2.T @ y_f)
    else:
        Q1, Q2 = None, None
        # Model 1
        coeffs1, _, _, _ = np.linalg.lstsq(A1, y_f, rcond=None)
        y1_h = A1 @ coeffs1
        # Model H2
        coeffs2, _, _, _ = np.linalg.lstsq(A2, y_f, rcond=None)
        y2_h = A2 @ coeffs2

    res_y1 = y_f - y1_h
    dof_h1 = N - A1.shape[1]

    res_y2 = y_f - y2_h
    dof_h2 = N - A2.shape[1]

    # 2. Augmented regression
    # Augmented model: H1 vs H2
    t_asy_X = _calc_aug_reg(A1, y_f, y2_h, solver=solver)
    dof_aug_X = N - A1.shape[1] - 1
    p_asy_X = float(stats.t.sf(np.abs(t_asy_X), dof_aug_X) * 2) if not np.isnan(t_asy_X) else np.nan
    # Augmented model: H2 vs H1
    t_asy_Z, p_asy_Z = np.nan, np.nan
    if direction == 'pairwise':
        t_asy_Z = _calc_aug_reg(A2, y_f, y1_h, solver=solver)
        dof_aug_Z = N - A2.shape[1] - 1
        p_asy_Z = float(stats.t.sf(np.abs(t_asy_Z), dof_aug_Z) * 2) if not np.isnan(t_asy_Z) else np.nan

    # 3. Residual scaling and bootstrap setup
    res_scaled_X, res_std_X = _setup_residual_scaling(res_y1, N, dof_h1, Q1, A1, boot_method, scaled_residuals,
                                                      solver)
    res_scaled_Z, res_std_Z = None, None
    if direction == 'pairwise':
        res_scaled_Z, res_std_Z = _setup_residual_scaling(res_y2, N, dof_h2, Q2, A2, boot_method, scaled_residuals,
                                                          solver)

    # Helper function to generate bootstrap samples y_star
    def _draw_y_star(
            y_null_fit: np.ndarray,
            res_scaled: np.ndarray,
            res_std: float
    ) -> np.ndarray:
        if boot_method == 'wild':
            if res_distribution == 'rademacher':
                v = rng.choice([-1.0, 1.0], size=N, replace=True)
            elif res_distribution == 'mammen':
                v_vals = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
                v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                v = rng.choice(v_vals, size=N, replace=True, p=v_probs)
            elif res_distribution == 'webb':
                v_vals = [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)]
                v = rng.choice(v_vals, size=N, replace=True)
            elif res_distribution == 'normal':
                v = rng.normal(loc=0.0, scale=1.0, size=N)
            else:
                raise ValueError("Supported distributions are 'rademacher', 'mammen', 'webb', and 'normal'")
            return y_null_fit + res_scaled * v

        elif boot_method in ['residuals', 'semiparametric']:
            b_idx = rng.choice(N, size=N, replace=True)
            return y_null_fit + res_scaled[b_idx]

        elif boot_method == 'parametric':
            return y_null_fit + rng.normal(loc=0.0, scale=res_std, size=N)

        raise ValueError(f"Invalid boot method: {boot_method}.")

    # 4. Bootstrap loop
    t_boot_X_list, t_boot_Z_list = [], []

    for _ in range(B):
        if boot_method == 'pairs':
            b_idx = rng.choice(N, size=N, replace=True)
            y_b, X_b, Z_b = y_f[b_idx], X_mat[b_idx], Z_mat[b_idx]
            A1_b = np.column_stack([X_b, ones_col])
            A2_b = np.column_stack([Z_b, ones_col])

            # H1 vs H2
            try:
                if solver == 'qr':
                    Q2_b, _ = np.linalg.qr(A2_b, mode='reduced')
                    y2_h_b = Q2_b @ (Q2_b.T @ y_b)
                else:
                    coeffs2_b, _, _, _ = np.linalg.lstsq(A2_b, y_b, rcond=None)
                    y2_h_b = A2_b @ coeffs2_b

                t_b_X = _calc_aug_reg(A1_b, y_b, y2_h_b, solver=solver)
                if not np.isnan(t_b_X):
                    t_boot_X_list.append(t_b_X - t_asy_X)
            except (np.linalg.LinAlgError, ValueError):
                pass

            # H2 vs H1
            if direction == 'pairwise':
                try:
                    if solver == 'qr':
                        Q1_b, _ = np.linalg.qr(A1_b, mode='reduced')
                        y1_h_b = Q1_b @ (Q1_b.T @ y_b)
                    else:
                        coeffs1_b, _, _, _ = np.linalg.lstsq(A1_b, y_b, rcond=None)
                        y1_h_b = A1_b @ coeffs1_b

                    t_b_Z = _calc_aug_reg(A2_b, y_b, y1_h_b, solver=solver)
                    if not np.isnan(t_b_Z):
                        t_boot_Z_list.append(t_b_Z - t_asy_Z)
                except (np.linalg.LinAlgError, ValueError):
                    pass

        else:
            # H1 vs H2
            try:
                y_star_X = _draw_y_star(y1_h, res_scaled_X, res_std_X)
                if solver == 'qr':
                    y2_h_star = Q2 @ (Q2.T @ y_star_X)
                else:
                    coeffs2_star, _, _, _ = np.linalg.lstsq(A2, y_star_X, rcond=None)
                    y2_h_star = A2 @ coeffs2_star

                t_b_X = _calc_aug_reg(A1, y_star_X, y2_h_star, solver=solver)
                if not np.isnan(t_b_X):
                    t_boot_X_list.append(t_b_X)
            except (np.linalg.LinAlgError, ValueError):
                pass

            # H2 vs H1
            if direction == 'pairwise':
                try:
                    y_star_Z = _draw_y_star(y2_h, res_scaled_Z, res_std_Z)
                    if solver == 'qr':
                        y1_h_star = Q1 @ (Q1.T @ y_star_Z)
                    else:
                        coeffs1_star, _, _, _ = np.linalg.lstsq(A1, y_star_Z, rcond=None)
                        y1_h_star = A1 @ coeffs1_star

                    t_b_Z = _calc_aug_reg(A2, y_star_Z, y1_h_star, solver=solver)
                    if not np.isnan(t_b_Z):
                        t_boot_Z_list.append(t_b_Z)
                except (np.linalg.LinAlgError, ValueError):
                    pass

    # 5. Bootstrap P-value
    def _compute_p_boot(
            t_boot_list: list,
            t_obs: float
    ) -> Tuple[float, int]:
        arr = np.array(t_boot_list)
        s_its = len(arr)
        if s_its == 0 or np.isnan(t_obs):
            return np.nan, 0
        if boot_tails == 'one':
            p_val_boot = float((1 + np.sum(arr >= t_obs)) / (1 + s_its))
        elif boot_tails == 'two':
            p_val_boot = float((1 + np.sum(np.abs(arr) >= np.abs(t_obs))) / (1 + s_its))
        else:
            raise ValueError("boot_tails must be 'one' or 'two'.")
        return p_val_boot, s_its

    p_boot_X, s_its_X = _compute_p_boot(t_boot_X_list, t_asy_X)
    p_boot_Z, s_its_Z = np.nan, 0
    if direction == 'pairwise':
        p_boot_Z, s_its_Z = _compute_p_boot(t_boot_Z_list, t_asy_Z)

    # 6. Outputs
    return JtestResults(
        t_asy_X=t_asy_X,
        p_asy_X=p_asy_X,
        p_boot_X=p_boot_X,
        s_its_X=s_its_X,
        t_asy_Z=t_asy_Z,
        p_asy_Z=p_asy_Z,
        p_boot_Z=p_boot_Z,
        s_its_Z=s_its_Z,
        t_boot_X=np.array(t_boot_X_list) if return_t_boot else None,
        t_boot_Z=np.array(t_boot_Z_list) if (return_t_boot and direction == 'pairwise') else None,
        residuals_X=res_y1 if return_residuals else None,
        residuals_Z=res_y2 if (return_residuals and direction == 'pairwise') else None,
        fitted_X=y1_h if return_fitted else None,
        fitted_Z=y2_h if (return_fitted and direction == 'pairwise') else None
    )

def _fit_ols_sm(
        y: np.ndarray,
        X: np.ndarray,
        solver: str = 'pinv'
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit Statsmodels OLS with explicit solver controls ('qr', 'pinv') and rank-deficient  matrices fallback support.
    """
    if solver == 'pinv':
        return sm.OLS(y, X).fit(solve_method='pinv')
    elif solver == 'qr':
        return sm.OLS(y, X).fit(solve_method='qr')

    try:
        res = sm.OLS(y, X).fit()
        if np.isnan(res.params).any() or np.isnan(res.bse).any():
            raise np.linalg.LinAlgError("Rank deficient or ill-conditioned matrix encountered.")
        return res
    except (np.linalg.LinAlgError, sla.LinAlgError):
        warnings.warn(
            "Design matrix is rank-deficient or singular. Falling back to 'pinv' pseudoinverse solver.",
            UserWarning,
            stacklevel=3,
        )
        return sm.OLS(y, X).fit(solve_method='pinv')

# statsmodels.OLS backend
def jtest_sm(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    B: int = 999,
    boot_method: Literal['residuals', 'semiparametric', 'wild', 'parametric', 'pairs'] = 'residuals',
    direction: Literal['pairwise', 'one-way'] = 'pairwise',
    solver: Literal['auto', 'qr', 'pinv'] = 'pinv',
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
    boot_method : {'residuals' ('semiparametric'), 'wild', 'parametric', 'pairs'}, default='residuals'
        Bootstrap Data Generation Process (DGP) method.
    direction : {'pairwise', 'one-way'}, default='pairwise'
        If 'pairwise', evaluates both directions (H1 vs H2 and H2 vs H1).
        If 'one-way', evaluates only H1 vs H2.
    **kwargs : dict
        Optional controls:
        - return_residuals (bool): Return H1 residuals (default: False).
        - return_fitted (bool): Return H1 fitted values (default: False).
        - res_distribution (str): Auxiliary distribution for wild bootstrap ('rademacher', 'mammen', 'webb', 'normal').
        - scaled_residuals (str): Residual scaling method ('simple' or 'leverage').
        - boot_tails (str): Tail definition for bootstrap p-value ('one' or 'two').
        - return_t_boot (bool): Return bootstrap t-statistics distributions (default: False).

    Returns
    -------
    JtestResults
        Dataclass object containing test statistics, p-values, and optional series.
    """

    if B <= 0:
        raise ValueError("B must be a strictly positive integer")
    elif B < 399:
        min_pval = 1.0 / (B + 1)
        warnings.warn(
            f"B={B} yields a minimum p-value of {min_pval:.4f} and may produce unstable results. B >= 999 is recommended for statistical stability.",
            UserWarning
        )
    if boot_method not in ['residuals', 'semiparametric', 'wild', 'parametric', 'pairs']:
        raise ValueError(f"Invalid boot method: {boot_method}")

    return_residuals: bool = kwargs.get('return_residuals', False)
    return_fitted: bool = kwargs.get('return_fitted', False)
    res_distribution: str = kwargs.get('res_distribution', 'rademacher')
    scaled_residuals: str = kwargs.get('scaled_residuals', 'simple')
    boot_tails: str = kwargs.get('boot_tails', 'one')
    return_t_boot: bool = kwargs.get('return_t_boot', False)

    rng = np.random.default_rng(seed)

    # Data Preprocessing
    y_f = np.asarray(y, dtype=float).ravel()
    N = len(y_f)

    # Ensure 2D matrices
    X_mat = np.atleast_2d(X).T if np.ndim(X) == 1 else np.asarray(X, dtype=float)
    Z_mat = np.atleast_2d(Z).T if np.ndim(Z) == 1 else np.asarray(Z, dtype=float)

    ones_col = np.ones((N, 1))  # Intercept column vector

    # 1. Fit Base Models
    # Model H1
    A1 = np.concatenate([X_mat, ones_col], axis=1)
    m1 = _fit_ols_sm(y_f, A1, solver=solver)
    y1_h = m1.fittedvalues
    res_y1 = m1.resid
    dof_h1 = m1.df_resid

    # Model H2
    A2 = np.concatenate([Z_mat, ones_col], axis=1)
    m2 = _fit_ols_sm(y_f, A2, solver=solver)
    y2_h = m2.fittedvalues
    res_y2 = m2.resid
    dof_h2 = m2.df_resid

    # 2. Augmented Regression
    # H1 vs H2
    A_aug_X = np.concatenate([A1, np.asarray(y2_h).reshape(-1, 1)], axis=1)
    m_aug_X = _fit_ols_sm(y_f, A_aug_X, solver=solver)
    t_asy_X = float(m_aug_X.tvalues[-1])
    p_asy_X = float(m_aug_X.pvalues[-1])

    # H2 vs H1
    t_asy_Z, p_asy_Z = np.nan, np.nan
    if direction == 'pairwise':
        A_aug_Z = np.concatenate([A2, np.asarray(y1_h).reshape(-1, 1)], axis=1)
        m_aug_Z = _fit_ols_sm(y_f, A_aug_Z, solver=solver)
        t_asy_Z = float(m_aug_Z.tvalues[-1])
        p_asy_Z = float(m_aug_Z.pvalues[-1])


    # 3. Residual scaling and bootstrap setup
    def _setup_sm_residual_scaling(m, res_y, dof_h):
        if boot_method in ['residuals', 'semiparametric']:
            res_centered = res_y - np.mean(res_y)
            if scaled_residuals == 'simple':
                return res_centered * np.sqrt(N / dof_h), None
            elif scaled_residuals == 'leverage':
                leverage = m.get_influence().hat_matrix_diag  # Hat matrix diagonal
                return res_centered / np.sqrt(1.0 - leverage), None
            else:
                raise ValueError("Supported residual scaling options: 'simple' or 'leverage'.")
        elif boot_method == 'wild':
            if scaled_residuals == 'simple':
                return res_y * np.sqrt(N / dof_h), None
            elif scaled_residuals == 'leverage':
                leverage = m.get_influence().hat_matrix_diag
                return res_y / np.sqrt(1.0 - leverage), None
            else:
                raise ValueError("Supported residual scaling options: 'simple' or 'leverage'.")
        elif boot_method == 'parametric':
            return None, np.sqrt(m.ssr / dof_h)
        return None, None

    res_scaled_X, res_std_X = _setup_sm_residual_scaling(m1, res_y1, dof_h1)
    res_scaled_Z, res_std_Z = None, None
    if direction == 'pairwise':
        res_scaled_Z, res_std_Z = _setup_sm_residual_scaling(m2, res_y2, dof_h2)

    # Helper function to generate bootstrap samples y_star
    def _draw_y_star(
            y_null_fit: np.ndarray,
            res_scaled: np.ndarray,
            res_std: float
    ) -> np.ndarray:
        if boot_method == 'wild':
            if res_distribution == 'rademacher':
                v = rng.choice([-1.0, 1.0], size=N, replace=True)
            elif res_distribution == 'mammen':
                v_vals = [-(np.sqrt(5)-1) / 2, (np.sqrt(5)+1) / 2]
                v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                v = rng.choice(v_vals, size=N, replace=True, p=v_probs)
            elif res_distribution == 'webb':
                v_vals = [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)]
                v = rng.choice(v_vals, size=N, replace=True)
            elif res_distribution == 'normal':
                v = rng.normal(loc=0.0, scale=1.0, size=N)
            else:
                raise ValueError("Supported distributions are 'rademacher', 'mammen', 'webb', and 'normal'")
            return y_null_fit + res_scaled * v

        elif boot_method in ['residuals', 'semiparametric']:
            b_idx = rng.choice(N, size=N, replace=True)
            return y_null_fit + res_scaled[b_idx]

        elif boot_method == 'parametric':
            return y_null_fit + rng.normal(loc=0.0, scale=res_std, size=N)

        raise ValueError(f"Invalid boot method: {boot_method}.")


    # 4. Bootstrap loop
    t_boot_X_list, t_boot_Z_list = [], []

    for _ in range(B):
        if boot_method == 'pairs':
            b_idx = rng.choice(N, size=N, replace=True)
            y_b, X_b, Z_b = y_f[b_idx], X_mat[b_idx], Z_mat[b_idx]
            A1_b = np.concatenate([X_b, ones_col], axis=1)
            A2_b = np.concatenate([Z_b, ones_col], axis=1)

            # H1 vs H2
            try:
                m2_b = _fit_ols_sm(y_b, A2_b, solver=solver)
                y2_h_b = m2_b.fittedvalues
                A_aug_b_X = np.concatenate([A1_b, np.asarray(y2_h_b).reshape(-1, 1)], axis=1)
                m_aug_b_X = _fit_ols_sm(y_b, A_aug_b_X, solver=solver)
                t_boot_X_list.append(float(m_aug_b_X.tvalues[-1]) - t_asy_X)
            except (np.linalg.LinAlgError, ValueError):
                pass

            # H2 vs H1
            if direction == 'pairwise':
                try:
                    m1_b = _fit_ols_sm(y_b, A1_b, solver=solver)
                    y1_h_b = m1_b.fittedvalues
                    A_aug_b_Z = np.concatenate([A2_b, np.asarray(y1_h_b).reshape(-1, 1)], axis=1)
                    m_aug_b_Z = _fit_ols_sm(y_b, A_aug_b_Z, solver=solver)
                    t_boot_Z_list.append(float(m_aug_b_Z.tvalues[-1]) - t_asy_Z)
                except (np.linalg.LinAlgError, ValueError):
                    pass

        else:
            # H1 vs H2
            try:
                y_star_X = _draw_y_star(y1_h, res_scaled_X, res_std_X)
                m2_b_X = _fit_ols_sm(y_star_X, A2, solver=solver)
                y2_h_b = m2_b_X.fittedvalues
                A_aug_b_X = np.concatenate([A1, np.asarray(y2_h_b).reshape(-1, 1)], axis=1)
                m_aug_b_X = _fit_ols_sm(y_star_X, A_aug_b_X, solver=solver)
                t_boot_X_list.append(float(m_aug_b_X.tvalues[-1]))
            except (np.linalg.LinAlgError, ValueError):
                pass

            # H2 vs H1
            if direction == 'pairwise':
                try:
                    y_star_Z = _draw_y_star(y2_h, res_scaled_Z, res_std_Z)
                    m1_b_Z = _fit_ols_sm(y_star_Z, A1, solver=solver)
                    y1_h_b_Z = m1_b_Z.fittedvalues
                    A_aug_b_Z = np.concatenate([A2, np.asarray(y1_h_b_Z).reshape(-1, 1)], axis=1)
                    m_aug_b_Z = _fit_ols_sm(y_star_Z, A_aug_b_Z, solver=solver)
                    t_boot_Z_list.append(float(m_aug_b_Z.tvalues[-1]))
                except (np.linalg.LinAlgError, ValueError):
                    pass

    # 5. Bootstrap P-Value
    def _compute_p_boot(
            t_boot_list: list,
            t_obs: float
    ) -> Tuple[float, int]:
        arr = np.array(t_boot_list)
        s_its = len(arr)
        if s_its == 0 or np.isnan(t_obs):
            return np.nan, 0
        if boot_tails == 'one':
            p_val_boot = float((1 + np.sum(arr >= t_obs)) / (1 + s_its))
        elif boot_tails == 'two':
            p_val_boot = float((1 + np.sum(np.abs(arr) >= np.abs(t_obs))) / (1 + s_its))
        else:
            raise ValueError("boot_tails must be 'one' or 'two'.")
        return p_val_boot, s_its

    p_boot_X, s_its_X = _compute_p_boot(t_boot_X_list, t_asy_X)
    p_boot_Z, s_its_Z = np.nan, 0
    if direction == 'pairwise':
        p_boot_Z, s_its_Z = _compute_p_boot(t_boot_Z_list, t_asy_Z)

    # 6. Outputs
    return JtestResults(
        t_asy_X=t_asy_X,
        p_asy_X=p_asy_X,
        p_boot_X=p_boot_X,
        s_its_X=s_its_X,
        t_asy_Z=t_asy_Z,
        p_asy_Z=p_asy_Z,
        p_boot_Z=p_boot_Z,
        s_its_Z=s_its_Z,
        t_boot_X=np.array(t_boot_X_list) if return_t_boot else None,
        t_boot_Z=np.array(t_boot_Z_list) if (return_t_boot and direction == 'pairwise') else None,
        residuals_X=res_y1 if return_residuals else None,
        residuals_Z=res_y2 if (return_residuals and direction == 'pairwise') else None,
        fitted_X=y1_h if return_fitted else None,
        fitted_Z=y2_h if (return_fitted and direction == 'pairwise') else None
    )


# SciPy backend (+ statsmodels.OLS and LMFIT)
def jtest_scipy(
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        func_X: Callable,
        func_Z: Callable,
        B: int = 999,
        boot_method: Literal['residuals', 'semiparametric', 'wild', 'parametric', 'pairs'] = "pairs",
        direction: Literal['pairwise', 'one-way'] = 'pairwise',
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
    boot_method : {'pairs', 'residuals' ('semiparametric'), 'wild', 'parametric'}, default='pairs'
        Bootstrap Data Generation Process (DGP) method.
    direction : {'pairwise', 'one-way'}, default='pairwise'
        If 'pairwise', evaluates both directions (H1 vs H2 and H2 vs H1).
        If 'one-way', evaluates only H1 vs H2.
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
        - res_distribution : {'rademacher', 'mammen', 'webb', 'normal'}, default='rademacher'
            Auxiliary distribution for wild bootstrap resampling.
        - boot_tails : {'one', 'two'}, default='one'
            Tail definition for calculating the empirical bootstrap p-value.
        - return_residuals : bool, default=False
            If True, populates `residuals` in the returned `JtestResults`.
        - return_fitted : bool, default=False
            If True, populates `fitted` in the returned `JtestResults`.
        - return_t_boot (bool): default=False
            Return bootstrap t-statistics distributions.

        *Backend-Specific Controls:*
        - p0_X : list, np.ndarray, or dict, optional
            Initial parameter values for Model H1:
            - `list`/`np.ndarray` for 'hybrid-ols' and 'scipy' (e.g., `[1.0, 0.5]`)
            - `dict` for 'lmfit' (e.g., `{'alpha': 1.0, 'beta': 0.5}`)
        - p0_Z : list, np.ndarray, or dict, optional
            Initial parameter values for Model H2 (same formatting rules as `p0_X`).
        - curvefitmethod : str, default='lm'
            [*'hybrid-ols' and 'scipy' backends only*] Optimization method passed to
            `scipy.optimize.curve_fit` (e.g., 'lm', 'trf', 'dogbox').

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

    if B <= 0:
        raise ValueError('B must be a strictly positive integer')
    elif B < 399:
        min_pval = 1.0 / (B + 1)
        warnings.warn(
            f"B={B} yields a minimum p-value of {min_pval:.4f} and may produce unstable results. B >= 999 is recommended for statistical stability.",
            UserWarning
        )
    if boot_method not in ['residuals', 'semiparametric', 'wild', 'parametric', 'pairs']:
        raise ValueError(f"Invalid boot method: {boot_method}")

    return_residuals: bool = kwargs.get('return_residuals', False)
    return_fitted: bool = kwargs.get('return_fitted', False)
    jtest_solver: str = kwargs.get('jtest_solver', 'hybrid-ols')
    res_distribution: str = kwargs.get('res_distribution', 'rademacher')
    boot_tails: str = kwargs.get('boot_tails', 'one')
    return_t_boot: bool = kwargs.get('return_t_boot', False)

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
    ) -> np.ndarray:
        """Simulate synthetic y* values under the null hypothesis (H1)."""
        if boot_method == 'wild':
            if res_distribution == 'rademacher':
                v = rng.choice([-1.0, 1.0], size=N, replace=True)
            elif res_distribution == 'mammen':
                v_choices = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
                v_probs = [(np.sqrt(5) + 1) / (2 * np.sqrt(5)), (np.sqrt(5) - 1) / (2 * np.sqrt(5))]
                v = rng.choice(v_choices, size=N, replace=True, p=v_probs)
            elif res_distribution == 'webb':
                v_vals = [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)]
                v = rng.choice(v_vals, size=N, replace=True)
            elif res_distribution == 'normal':
                v = rng.normal(loc=0.0, scale=1.0, size=N)
            else:
                raise ValueError("Supported distributions are 'rademacher', 'mammen', 'webb', and 'normal'.")
            return y_h_null + res_scaled * v

        elif boot_method in ['residuals', 'semiparametric']:
            b_idx = rng.choice(N, size=N, replace=True)
            return y_h_null + res_scaled[b_idx]

        elif boot_method == 'parametric':
            return y_h_null + rng.normal(loc=0.0, scale=res_std, size=N)

        raise ValueError(f'Invalid boot_method: {boot_method}')

    def _run_directional_test(
            x_null: np.ndarray,
            x_alt: np.ndarray,
            f_null: Callable,
            f_alt: Callable,
            p0_null: Any,
            p0_alt: Any
    ) -> Tuple[float, float, float, int, np.ndarray, np.ndarray, np.ndarray]:

        # Hybrid-OLS: Base models fitted via SciPy with statmodels.OLS augmented regression
        if jtest_solver == 'hybrid-ols':
            curvefitmethod = kwargs.get('curvefitmethod', 'lm')
            popt_null, _ = curve_fit(f_null, x_null, y_f, p0=p0_null, method=curvefitmethod)
            y_h_null = f_null(x_null, *popt_null)
            res_null = y_f - y_h_null
            dof = N - len(popt_null)

            popt_alt, _ = curve_fit(f_alt, x_alt, y_f, p0=p0_alt, method=curvefitmethod)
            y_h_alt = np.asarray(f_alt(x_alt, *popt_alt)).ravel()

            X_aug = sm.add_constant(np.column_stack([x_null, y_h_alt]))
            j_model = sm.OLS(y_f, X_aug).fit()
            t_stat_obs = float(j_model.tvalues[-1])
            p_obs = float(j_model.pvalues[-1])

            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null - np.mean(res_null)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null ** 2) / dof))

            # Bootstrap loop
            t_stat_boot = []
            for _ in range(B):
                try:
                    if boot_method == 'pairs':
                        b_idx = rng.choice(N, size=N, replace=True)
                        x_null_b = x_null[b_idx] if x_null.ndim == 1 else x_null[b_idx, :]
                        x_alt_b = x_alt[b_idx] if x_alt.ndim == 1 else x_alt[b_idx, :]
                        y_b = y_f[b_idx]

                    else:
                        x_null_b, x_alt_b = x_null, x_alt
                        y_b = generate_y_star(y_h_null, res_scaled, res_std)

                    popt_alt_b, _ = curve_fit(f_alt, x_alt_b, y_b, p0=p0_alt, method=curvefitmethod)
                    y_h_alt_b = np.asarray(f_alt(x_alt_b, *popt_alt_b)).ravel()

                    X_aug_b = sm.add_constant(np.column_stack((x_null_b, y_h_alt_b)))
                    j_model_b = sm.OLS(y_b, X_aug_b).fit()

                    t_b = float(j_model_b.tvalues[-1])
                    t_stat_boot.append(t_b - t_stat_obs if boot_method == 'pairs' else t_b)
                except (RuntimeError, np.linalg.LinAlgError, ValueError):
                    continue

        # SciPy: Uses SciPy for both base models and augmented regression
        elif jtest_solver == 'scipy':
            curvefitmethod = kwargs.get('curvefitmethod', 'lm')
            popt_null, _ = curve_fit(f_null, x_null, y_f, p0=p0_null, method=curvefitmethod)
            y_h_null = f_null(x_null, *popt_null)
            res_null = y_f - y_h_null
            dof = N - len(popt_null)

            popt_alt, _ = curve_fit(f_alt, x_alt, y_f, p0=p0_alt, method=curvefitmethod)
            y_h_alt = np.asarray(f_alt(x_alt, *popt_alt)).ravel()

            def make_augmented_model(
                    base_model_func: Callable,
                    competitor_fitted: np.ndarray,
            ) -> Callable:
                return lambda x, *terms: base_model_func(x, *terms[:-1]) + terms[-1] * competitor_fitted

            model_aug = make_augmented_model(f_null, y_h_alt)
            popt_aug, pcov_aug = curve_fit(model_aug, x_null, y_f, p0=[*popt_null, 0.0], method=curvefitmethod)

            try:
                se_aug = np.sqrt(np.abs(pcov_aug[-1, -1]))
            except (IndexError, TypeError):
                se_aug = np.nan

            t_stat_obs = float(popt_aug[-1] / se_aug) if not np.isnan(se_aug) and se_aug > 0 else np.nan
            p_obs = float(stats.t.sf(np.abs(t_stat_obs), df=dof - 1) * 2) if not np.isnan(t_stat_obs) else np.nan

            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null - np.mean(res_null)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null ** 2) / dof))

            # Bootstrap loop
            t_stat_boot = []
            for _ in range(B):
                try:
                    if boot_method == 'pairs':
                        b_idx = rng.choice(N, size=N, replace=True)
                        x_null_b = x_null[b_idx] if x_null.ndim == 1 else x_null[b_idx, :]
                        x_alt_b = x_alt[b_idx] if x_alt.ndim == 1 else x_alt[b_idx, :]
                        y_b = y_f[b_idx]
                    else:
                        x_null_b, x_alt_b = x_null, x_alt
                        y_b = generate_y_star(y_h_null, res_scaled, res_std)

                    popt_alt_b, _ = curve_fit(f_alt, x_alt_b, y_b, p0=p0_alt, method=curvefitmethod)
                    y_h_alt_b = np.asarray(f_alt(x_alt_b, *popt_alt_b)).ravel()

                    model_aug_b = make_augmented_model(f_null, y_h_alt_b)
                    popt_aug_b, pcov_aug_b = curve_fit(model_aug_b, x_null_b, y_b, p0=[*popt_null, 0.0],
                                                       method=curvefitmethod)

                    try:
                        se_b = np.sqrt(np.abs(pcov_aug_b[-1, -1]))
                    except (IndexError, TypeError):
                        se_b = np.nan

                    if not np.isnan(se_b) and se_b > 0:
                        t_b = float(popt_aug_b[-1] / np.sqrt(pcov_aug_b[-1, -1]))
                        t_stat_boot.append(t_b - t_stat_obs if boot_method == 'pairs' else t_b)
                except (RuntimeError, ValueError):
                    continue

        # LMFIT: Base models and augmented regressions via lmfit.Model
        elif jtest_solver == 'lmfit':
            if not HAS_LMFIT:
                raise ImportError(
                    "The 'lmfit' package is required for jtest_solver='lmfit'. Install via 'pip install lmfit'.")

            model_null, model_alt = Model(f_null), Model(f_alt)
            params_null, params_alt = model_null.make_params(), model_alt.make_params()

            if isinstance(p0_null, dict):
                for k, v in p0_null.items():
                    if k in params_null: params_null[k].set(value=v)
            if isinstance(p0_alt, dict):
                for k, v in p0_alt.items():
                    if k in params_alt: params_alt[k].set(value=v)

            res_null = model_null.fit(y_f, x=x_null, params=params_null)
            res_alt = model_alt.fit(y_f, x=x_alt, params=params_alt)

            y_h_null = res_null.best_fit
            res_null_arr = res_null.residual
            dof = N - res_null.nvarys

            def make_lmfit_augmented(
                    base_func: Callable,
                    fitted_alt: np.ndarray
            ) -> Callable:
                return lambda x, lam=0.0, **params: base_func(x, **params) + lam * fitted_alt

            aug_func = make_lmfit_augmented(f_null, res_alt.best_fit)
            model_aug = Model(aug_func)
            aug_params = res_null.params.copy()
            aug_params.add('lam', value=0.0)

            res_aug = model_aug.fit(y_f, x=x_null, params=aug_params)
            lam_se = res_aug.params['lam'].stderr

            t_stat_obs = float(res_aug.params['lam'].value / lam_se if lam_se else np.nan)
            p_obs = float(stats.t.sf(np.abs(t_stat_obs), df=res_aug.nfree) * 2) if not np.isnan(
                t_stat_obs) else np.nan

            res_scaled, res_std = None, None
            if boot_method in ['residuals', 'semiparametric']:
                res_scaled = (res_null_arr - np.mean(res_null_arr)) * np.sqrt(N / dof)
            elif boot_method == 'wild':
                res_scaled = res_null_arr * np.sqrt(N / dof)
            elif boot_method == 'parametric':
                res_std = float(np.sqrt(np.sum(res_null_arr ** 2) / dof))

            t_stat_boot = []
            for _ in range(B):
                try:
                    if boot_method == 'pairs':
                        b_idx = rng.choice(N, size=N, replace=True)
                        x_null_b = x_null[b_idx] if x_null.ndim == 1 else x_null[b_idx, :]
                        x_alt_b = x_alt[b_idx] if x_alt.ndim == 1 else x_alt[b_idx, :]
                        y_b = y_f[b_idx]
                    else:
                        x_null_b, x_alt_b = x_null, x_alt
                        y_b = generate_y_star(y_h_null, res_scaled, res_std)

                    res_alt_b = model_alt.fit(y_b, x=x_alt_b, params=res_alt.params.copy())
                    aug_func_b = make_lmfit_augmented(f_null, res_alt_b.best_fit)

                    model_aug_b = Model(aug_func_b)
                    aug_params_b = res_null.params.copy()
                    aug_params_b.add('lam', value=0.0)

                    res_aug_b = model_aug_b.fit(y_b, x=x_null_b, params=aug_params_b)
                    lam_se_b = res_aug_b.params['lam'].stderr
                    t_b = float(res_aug_b.params['lam'].value / lam_se_b) if lam_se_b else np.nan

                    if not np.isnan(t_b):
                        t_stat_boot.append(t_b - t_stat_obs if boot_method == 'pairs' else t_b)
                except Exception:
                    continue
            res_null = res_null_arr
        else:
            raise ValueError(
                f"Unknown jtest_solver: {jtest_solver}. Supported options are hybrid-ols, scipy, 'lmfit'.")

        # 5. Bootstrap P-value
        t_arr = np.array(t_stat_boot)
        s_its = len(t_arr)
        if s_its == 0 or np.isnan(t_stat_obs):
            p_boot = np.nan
        elif boot_tails == 'one':
            p_boot = float((1 + np.sum(t_arr >= t_stat_obs)) / (1 + s_its))
        elif boot_tails == 'two':
            p_boot = float((1 + np.sum(np.abs(t_arr) >= np.abs(t_stat_obs))) / (1 + s_its))
        else:
            raise ValueError("boot_tails option must be 'one' or 'two'.")

        return t_stat_obs, p_obs, p_boot, s_its, t_arr, res_null, y_h_null

    p0_X = kwargs.get('p0_X', None)
    p0_Z = kwargs.get('p0_Z', None)

    # H1 vs H2
    t_asy_X, p_asy_X, p_boot_X, s_its_X, t_boot_X, res_X, fit_X = _run_directional_test(
        X_mat, Z_mat, func_X, func_Z, p0_X, p0_Z)

    # H2 vs H1
    t_asy_Z, p_asy_Z, p_boot_Z, s_its_Z = np.nan, np.nan, np.nan, 0
    t_boot_Z, res_Z, fit_Z = None, None, None

    if direction == "pairwise":
        t_asy_Z, p_asy_Z, p_boot_Z, s_its_Z, t_boot_Z, res_Z, fit_Z = _run_directional_test(
            Z_mat, X_mat, func_Z, func_X, p0_Z, p0_X)

    # 6.Outputs
    return JtestResults(
        t_asy_X=t_asy_X,
        p_asy_X=p_asy_X,
        p_boot_X=p_boot_X,
        s_its_X=s_its_X,
        t_asy_Z=t_asy_Z,
        p_asy_Z=p_asy_Z,
        p_boot_Z=p_boot_Z,
        s_its_Z=s_its_Z,
        t_boot_X=t_boot_X if return_t_boot else None,
        t_boot_Z=t_boot_Z if (return_t_boot and direction == 'pairwise') else None,
        residuals_X=np.asarray(res_X) if return_residuals else None,
        residuals_Z=np.asarray(res_Z) if (return_residuals and direction == 'pairwise') else None,
        fitted_X=np.asarray(fit_X) if return_fitted else None,
        fitted_Z=np.asarray(fit_Z) if (return_fitted and direction == 'pairwise') else None
    )
