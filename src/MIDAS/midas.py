"""Restricted MIDAS tools for teaching and applied notebook work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .align import stack_high_freq_lags


def beta_weights(theta1: float, theta2: float, K: int, eps: float = 1e-6) -> np.ndarray:
    """Normalized Beta weights w_k for k = 1..K."""
    x = np.linspace(eps, 1 - eps, K)
    log_w = (theta1 - 1.0) * np.log(x) + (theta2 - 1.0) * np.log1p(-x)
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.repeat(1.0 / K, K)
    return w / total


def almon_weights(coefs: np.ndarray, K: int) -> np.ndarray:
    """Exponential Almon weights w_k = exp(c_1 k + c_2 k^2 + ...) / sum."""
    k = np.arange(1, K + 1, dtype=float)
    poly = np.zeros_like(k)
    for j, c in enumerate(coefs, start=1):
        poly += c * k ** j
    w = np.exp(poly)
    return w / w.sum()


def legendre_dictionary(n_lags: int, degree: int = 2) -> np.ndarray:
    """``(n_lags, degree + 1)`` shifted-Legendre MIDAS dictionary.

    An *unrestricted* alternative to the Beta and Almon weight functions: instead
    of one weight vector, it returns a small orthogonal basis. Column ``l`` is the
    degree-``l`` Legendre polynomial evaluated at ``n_lags`` equally spaced points
    on ``[0, 1]``. Projecting a block of lags onto it, ``lags @ W``, yields a smooth
    low-dimensional summary of the lag shape; the projected columns form the *group*
    penalised together in :class:`~MIDAS.sparse.SparseGroupLasso` MIDAS regressions
    (Babii, Ghysels and Striaukas, 2022).
    """
    x = np.linspace(0.0, 1.0, n_lags)
    return np.polynomial.legendre.legvander(2.0 * x - 1.0, degree)


def _as_2d(arr: np.ndarray | pd.DataFrame | pd.Series, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D")
    return out


def _normalize_high_freq_inputs(
    X_high: Mapping[str, np.ndarray | pd.DataFrame] | Sequence[np.ndarray | pd.DataFrame] | np.ndarray | pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """Normalize monthly lag inputs to (feature_names, list_of_2d_arrays)."""
    if isinstance(X_high, Mapping):
        names = list(X_high.keys())
        arrays = [_as_2d(X_high[name], name=name) for name in names]
        return names, arrays

    if isinstance(X_high, (list, tuple)):
        arrays = [_as_2d(arr, name=f"x{i + 1}") for i, arr in enumerate(X_high)]
        if feature_names is None:
            names = [f"x{i + 1}" for i in range(len(arrays))]
        else:
            names = list(feature_names)
        if len(names) != len(arrays):
            raise ValueError("feature_names length does not match X_high")
        return names, arrays

    name = feature_names[0] if feature_names else "x1"
    return [name], [_as_2d(X_high, name=name)]


@dataclass
class BetaMidasResult:
    alpha: float
    beta: float
    theta1: float
    theta2: float
    K: int
    weights: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    success: bool = True
    message: str = ""
    params: np.ndarray | None = None

    def predict(self, X_high: np.ndarray | pd.DataFrame) -> np.ndarray:
        X = _as_2d(X_high, name="X_high")
        return self.alpha + self.beta * (X @ self.weights)


@dataclass
class MultiBetaMidasResult:
    alpha: float
    feature_names: list[str]
    betas: np.ndarray
    theta1: np.ndarray
    theta2: np.ndarray
    K_list: list[int]
    weights: dict[str, np.ndarray]
    low_freq_coefs: np.ndarray | None
    low_freq_feature_names: list[str]
    fitted: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    params: np.ndarray | None = None

    def predict(
        self,
        X_high: Mapping[str, np.ndarray | pd.DataFrame] | Sequence[np.ndarray | pd.DataFrame] | np.ndarray | pd.DataFrame,
        X_low: np.ndarray | pd.DataFrame | pd.Series | None = None,
    ) -> np.ndarray:
        names, arrays = _normalize_high_freq_inputs(X_high, feature_names=self.feature_names)
        if names != self.feature_names:
            raise ValueError(f"feature mismatch: expected {self.feature_names}, got {names}")

        n_obs = arrays[0].shape[0]
        pred = np.full(n_obs, self.alpha, dtype=float)
        for name, beta in zip(self.feature_names, self.betas):
            pred += beta * (arrays[self.feature_names.index(name)] @ self.weights[name])

        if self.low_freq_coefs is not None and len(self.low_freq_coefs):
            X_low_arr = _as_2d(X_low, name="X_low")
            pred += X_low_arr @ self.low_freq_coefs
        return pred


@dataclass
class BetaMIDASRegressor:
    """Sklearn-style wrapper around the restricted Beta-MIDAS estimator.

    Parameters
    ----------
    monthly_vars:
        Monthly variables that will be stacked as Beta-weighted lag blocks.
    n_lags:
        Number of monthly lags to include for each variable.
    low_freq_features:
        Optional quarterly controls such as lagged GDP growth.
    """

    monthly_vars: Sequence[str]
    n_lags: int = 12
    low_freq_features: Sequence[str] = field(default_factory=list)
    result_: MultiBetaMidasResult | None = None

    def fit(
        self,
        y: np.ndarray | pd.Series,
        X_high: Mapping[str, np.ndarray | pd.DataFrame] | Sequence[np.ndarray | pd.DataFrame] | np.ndarray | pd.DataFrame,
        X_low: np.ndarray | pd.DataFrame | pd.Series | None = None,
        *,
        start_params: np.ndarray | None = None,
    ) -> "BetaMIDASRegressor":
        self.result_ = fit_beta_midas_multi(
            y=y,
            X_high=X_high,
            X_low=X_low,
            feature_names=list(self.monthly_vars),
            low_freq_feature_names=list(self.low_freq_features),
            start_params=start_params,
        )
        return self

    def fit_frame(self, frame: pd.DataFrame, target: str) -> "BetaMIDASRegressor":
        needed = [target] + list(self.low_freq_features)
        for var in self.monthly_vars:
            needed.extend([f"{var}_L{i}" for i in range(self.n_lags)])
        clean = frame.dropna(subset=needed)
        return self.fit(
            y=clean[target].to_numpy(dtype=float),
            X_high=extract_midas_arrays(clean, self.monthly_vars, n_lags=self.n_lags),
            X_low=clean.loc[:, list(self.low_freq_features)].to_numpy(dtype=float) if self.low_freq_features else None,
        )

    def predict(
        self,
        X_high: Mapping[str, np.ndarray | pd.DataFrame] | Sequence[np.ndarray | pd.DataFrame] | np.ndarray | pd.DataFrame,
        X_low: np.ndarray | pd.DataFrame | pd.Series | None = None,
    ) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("fit the model before calling predict")
        return self.result_.predict(X_high, X_low)

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("fit the model before calling predict_frame")
        X_high = extract_midas_arrays(frame, self.monthly_vars, n_lags=self.n_lags)
        X_low = frame.loc[:, list(self.low_freq_features)].to_numpy(dtype=float) if self.low_freq_features else None
        return self.predict(X_high, X_low)

    def weight_frame(self) -> pd.DataFrame:
        if self.result_ is None:
            raise ValueError("fit the model before calling weight_frame")
        return weight_profile_table(self.result_)

    def parameter_table(self) -> pd.DataFrame:
        if self.result_ is None:
            raise ValueError("fit the model before calling parameter_table")
        rows = []
        for var, beta, theta1, theta2 in zip(
            self.result_.feature_names,
            self.result_.betas,
            self.result_.theta1,
            self.result_.theta2,
        ):
            rows.append(
                {
                    "variable": var,
                    "beta": beta,
                    "theta1": theta1,
                    "theta2": theta2,
                }
            )
        if self.result_.low_freq_coefs is not None:
            for name, coef in zip(self.result_.low_freq_feature_names, self.result_.low_freq_coefs):
                rows.append({"variable": name, "beta": coef, "theta1": np.nan, "theta2": np.nan})
        return pd.DataFrame(rows)


def _weighted_regressor_init(
    y: np.ndarray,
    X_use: list[np.ndarray],
    K_list: list[int],
    X_low_use: np.ndarray | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """OLS-based initializer using default recency-loaded Beta weights."""
    default_weights = [beta_weights(1.0, 5.0, K) for K in K_list]
    monthly_agg = np.column_stack([x @ w for x, w in zip(X_use, default_weights)])
    pieces = [np.ones((len(y), 1)), monthly_agg]
    if X_low_use is not None and X_low_use.size:
        pieces.append(X_low_use)
    design = np.column_stack(pieces)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha0 = float(coef[0])
    betas0 = coef[1:1 + len(X_use)]
    gamma0 = coef[1 + len(X_use):] if X_low_use is not None and X_low_use.size else np.array([])
    return alpha0, np.asarray(betas0, dtype=float), np.asarray(gamma0, dtype=float)


def fit_beta_midas(
    y: np.ndarray | pd.Series,
    X_high: np.ndarray | pd.DataFrame,
    K: int | None = None,
    init: tuple[float, float, float, float] | None = None,
    start_params: np.ndarray | None = None,
) -> BetaMidasResult:
    """Fit y_t = alpha + beta * sum_k w_k(theta) X_{t,k} + eps_t by NLS."""
    multi = fit_beta_midas_multi(
        y=y,
        X_high={"x1": X_high},
        X_low=None,
        init=init,
        start_params=start_params,
    )
    weights = multi.weights["x1"]
    return BetaMidasResult(
        alpha=multi.alpha,
        beta=float(multi.betas[0]),
        theta1=float(multi.theta1[0]),
        theta2=float(multi.theta2[0]),
        K=len(weights) if K is None else K,
        weights=weights,
        fitted=multi.fitted,
        residuals=multi.residuals,
        success=multi.success,
        message=multi.message,
        params=multi.params,
    )


def fit_beta_midas_multi(
    y: np.ndarray | pd.Series,
    X_high: Mapping[str, np.ndarray | pd.DataFrame] | Sequence[np.ndarray | pd.DataFrame] | np.ndarray | pd.DataFrame,
    X_low: np.ndarray | pd.DataFrame | pd.Series | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    low_freq_feature_names: Sequence[str] | None = None,
    init: tuple[float, float, float, float] | None = None,
    start_params: np.ndarray | None = None,
) -> MultiBetaMidasResult:
    """Fit a restricted multi-indicator MIDAS / ADL-MIDAS by NLS.

    Model:
        y_t = alpha
              + sum_j beta_j * sum_k w_{j,k}(theta_{j1}, theta_{j2}) x_{j,t-k}
              + z_t' gamma
              + eps_t

    where z_t can contain quarterly lags such as GDP_{t-1}.
    """
    feature_names_, arrays = _normalize_high_freq_inputs(X_high, feature_names=feature_names)
    if not arrays:
        raise ValueError("X_high must contain at least one monthly predictor")

    y_arr = np.asarray(y, dtype=float).ravel()
    n_obs = arrays[0].shape[0]
    if len(y_arr) != n_obs:
        raise ValueError("y and X_high must have the same number of rows")
    if any(arr.shape[0] != n_obs for arr in arrays[1:]):
        raise ValueError("all X_high arrays must have the same number of rows")

    K_list = [arr.shape[1] for arr in arrays]
    X_low_arr = None if X_low is None else _as_2d(X_low, name="X_low")
    if X_low_arr is not None and X_low_arr.shape[0] != n_obs:
        raise ValueError("X_low must have the same number of rows as y")

    mask = np.isfinite(y_arr)
    for arr in arrays:
        mask &= np.isfinite(arr).all(axis=1)
    if X_low_arr is not None:
        mask &= np.isfinite(X_low_arr).all(axis=1)

    y_use = y_arr[mask]
    X_use = [arr[mask] for arr in arrays]
    X_low_use = X_low_arr[mask] if X_low_arr is not None else None
    if len(y_use) < 8:
        raise ValueError("Too few observations after dropping NaNs")

    alpha0, betas0, gamma0 = _weighted_regressor_init(y_use, X_use, K_list, X_low_use)
    theta_defaults = []
    for _ in feature_names_:
        if init is None:
            theta_defaults.extend([1.0, 5.0])
        else:
            theta_defaults.extend([init[2], init[3]])

    if init is not None:
        alpha0 = init[0]
        if len(betas0):
            betas0[0] = init[1]

    default_x0 = [alpha0]
    for beta0, (t1_0, t2_0) in zip(betas0, np.asarray(theta_defaults).reshape(-1, 2)):
        default_x0.extend([beta0, t1_0, t2_0])
    default_x0.extend(gamma0.tolist())

    bounds = [(-np.inf, np.inf)]
    for _ in feature_names_:
        bounds.extend([(-np.inf, np.inf), (1e-3, 20.0), (1e-3, 20.0)])
    if X_low_use is not None and X_low_use.size:
        bounds.extend([(-np.inf, np.inf)] * X_low_use.shape[1])

    def _predict_from_params(
        params: np.ndarray,
        X_list: list[np.ndarray],
        X_low_block: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pred = np.full(X_list[0].shape[0], params[0], dtype=float)
        offset = 1
        weights_map: dict[str, np.ndarray] = {}
        for idx, name in enumerate(feature_names_):
            beta = params[offset]
            theta1 = params[offset + 1]
            theta2 = params[offset + 2]
            offset += 3
            w = beta_weights(theta1, theta2, K_list[idx])
            weights_map[name] = w
            pred += beta * (X_list[idx] @ w)
        if X_low_block is not None and X_low_block.size:
            gamma = params[offset:offset + X_low_block.shape[1]]
            pred += X_low_block @ gamma
        return pred, weights_map

    def neg_ssr(params: np.ndarray) -> float:
        pred, _ = _predict_from_params(params, X_use, X_low_use)
        resid = y_use - pred
        return float(resid @ resid)

    x0 = np.asarray(start_params, dtype=float) if start_params is not None else np.asarray(default_x0, dtype=float)
    if len(x0) != len(bounds):
        x0 = np.asarray(default_x0, dtype=float)

    res = minimize(
        neg_ssr,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000},
    )

    params = np.asarray(res.x, dtype=float)
    fitted_use, weights_map = _predict_from_params(params, X_use, X_low_use)
    fitted_full = np.full_like(y_arr, np.nan, dtype=float)
    fitted_full[mask] = fitted_use

    betas = []
    theta1 = []
    theta2 = []
    offset = 1
    for _ in feature_names_:
        betas.append(params[offset])
        theta1.append(params[offset + 1])
        theta2.append(params[offset + 2])
        offset += 3
    low_freq_coefs = params[offset:] if offset < len(params) else None

    return MultiBetaMidasResult(
        alpha=float(params[0]),
        feature_names=feature_names_,
        betas=np.asarray(betas, dtype=float),
        theta1=np.asarray(theta1, dtype=float),
        theta2=np.asarray(theta2, dtype=float),
        K_list=K_list,
        weights=weights_map,
        low_freq_coefs=None if low_freq_coefs is None or len(low_freq_coefs) == 0 else np.asarray(low_freq_coefs, dtype=float),
        low_freq_feature_names=list(low_freq_feature_names or []),
        fitted=fitted_full,
        residuals=y_arr - fitted_full,
        success=bool(res.success),
        message=str(res.message),
        params=params,
    )


def stack_midas_features(
    monthly: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    variables: Sequence[str],
    *,
    n_lags: int = 12,
    end_month: int | None = 3,
) -> pd.DataFrame:
    """Stack multiple monthly predictors into a wide MIDAS-ready frame."""
    if not isinstance(monthly.index, pd.DatetimeIndex):
        raise TypeError("monthly must be indexed by date")

    blocks = []
    for var in variables:
        lagged = stack_high_freq_lags(monthly[var], target_index, n_lags=n_lags, end_month=end_month)
        lagged.columns = [f"{var}_L{i}" for i in range(n_lags)]
        blocks.append(lagged)
    return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=target_index)


def extract_midas_arrays(
    frame: pd.DataFrame,
    monthly_vars: Sequence[str],
    *,
    n_lags: int = 12,
) -> dict[str, np.ndarray]:
    """Recover the monthly lag matrices expected by fit_beta_midas_multi."""
    arrays: dict[str, np.ndarray] = {}
    for var in monthly_vars:
        cols = [f"{var}_L{i}" for i in range(n_lags)]
        arrays[var] = frame.loc[:, cols].to_numpy(dtype=float)
    return arrays


def rolling_beta_midas_forecast(
    frame: pd.DataFrame,
    target: str,
    monthly_vars: Sequence[str],
    *,
    n_lags: int = 12,
    low_freq_features: Sequence[str] | None = None,
    eval_start: str,
    min_train: int = 24,
    window: str = "expanding",
    country: str | None = None,
) -> pd.DataFrame:
    """Expanding/rolling forecasts for a single-country MIDAS design frame."""
    low_freq_features = list(low_freq_features or [])
    if "date" not in frame.columns:
        raise ValueError("frame must contain a 'date' column")

    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    eval_start_ts = pd.to_datetime(eval_start)
    monthly_cols = [f"{var}_L{i}" for var in monthly_vars for i in range(n_lags)]
    required_cols = [target] + monthly_cols + low_freq_features

    if country is None and "country" in df.columns and df["country"].nunique() == 1:
        country = str(df["country"].iloc[0])

    rows = []
    eval_dates = df.loc[df["date"] >= eval_start_ts, "date"].tolist()
    last_params: np.ndarray | None = None
    for t in eval_dates:
        train = df.loc[df["date"] < t].dropna(subset=required_cols)
        test = df.loc[df["date"] == t].dropna(subset=required_cols)
        if len(train) < min_train or test.empty:
            continue
        if window == "rolling":
            train = train.tail(min_train * 4)

        X_high_tr = extract_midas_arrays(train, monthly_vars, n_lags=n_lags)
        X_high_te = extract_midas_arrays(test, monthly_vars, n_lags=n_lags)
        X_low_tr = train.loc[:, low_freq_features].to_numpy(dtype=float) if low_freq_features else None
        X_low_te = test.loc[:, low_freq_features].to_numpy(dtype=float) if low_freq_features else None

        try:
            model = fit_beta_midas_multi(
                y=train[target].to_numpy(dtype=float),
                X_high=X_high_tr,
                X_low=X_low_tr,
                low_freq_feature_names=low_freq_features,
                start_params=last_params,
            )
            last_params = model.params
            pred = model.predict(X_high_te, X_low_te)
        except Exception:
            pred = np.array([np.nan], dtype=float)

        pred = np.asarray(pred).ravel()
        for i, row in enumerate(test.itertuples()):
            rows.append(
                {
                    "country": country,
                    "date": row.date,
                    "y_true": getattr(row, target),
                    "y_hat": float(pred[i]) if i < len(pred) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def weight_profile_table(result: BetaMidasResult | MultiBetaMidasResult) -> pd.DataFrame:
    """Tidy weight profiles for plotting."""
    if isinstance(result, BetaMidasResult):
        return pd.DataFrame(
            {
                "variable": "x1",
                "lag": np.arange(1, result.K + 1),
                "weight": result.weights,
            }
        )

    rows = []
    for name in result.feature_names:
        weights = result.weights[name]
        rows.extend(
            {
                "variable": name,
                "lag": lag,
                "weight": weight,
            }
            for lag, weight in enumerate(weights, start=1)
        )
    return pd.DataFrame(rows)
