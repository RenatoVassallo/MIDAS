"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _align(y_true, y_pred):
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def rmse(y_true, y_pred) -> float:
    a, b = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def mae(y_true, y_pred) -> float:
    a, b = _align(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def mape(y_true, y_pred) -> float:
    a, b = _align(y_true, y_pred)
    nz = np.abs(a) > 1e-9
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((a[nz] - b[nz]) / a[nz])) * 100.0)


def pinball_loss(y_true, y_pred, q: float) -> float:
    """Quantile (pinball) loss at level q in (0,1)."""
    a, b = _align(y_true, y_pred)
    diff = a - b
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def dm_test(
    e1: np.ndarray | pd.Series,
    e2: np.ndarray | pd.Series,
    h: int = 1,
    loss: str = "se",
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Returns (DM-statistic, two-sided p-value).
    H0: E[loss(e1) - loss(e2)] = 0.
    Positive DM => model 1 has higher loss (worse).
    """
    a = np.asarray(e1, dtype=float).ravel()
    b = np.asarray(e2, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if loss == "se":
        d = a ** 2 - b ** 2
    elif loss == "ae":
        d = np.abs(a) - np.abs(b)
    else:
        raise ValueError("loss must be 'se' or 'ae'")
    T = len(d)
    if T < 8:
        return float("nan"), float("nan")
    d_bar = d.mean()
    # Newey-West-style HAC variance with bandwidth h-1
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        var_d += 2.0 * cov
    var_d = max(var_d, 1e-12)
    dm = d_bar / np.sqrt(var_d / T)
    # Harvey-Leybourne-Newbold small-sample correction
    K = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** 0.5
    dm_corr = dm * K
    pval = 2.0 * (1.0 - stats.t.cdf(abs(dm_corr), df=T - 1))
    return float(dm_corr), float(pval)


def evaluation_table(
    y_true,
    forecasts: dict[str, "np.ndarray | pd.Series"],
) -> pd.DataFrame:
    """Build a tidy RMSE/MAE table for a dict of {model_name: yhat}."""
    rows = []
    for name, yhat in forecasts.items():
        rows.append({"model": name, "RMSE": rmse(y_true, yhat), "MAE": mae(y_true, yhat)})
    return pd.DataFrame(rows).set_index("model")
