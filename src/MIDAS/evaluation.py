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


def cw_test(
    y_true: np.ndarray | pd.Series,
    f_restricted: np.ndarray | pd.Series,
    f_unrestricted: np.ndarray | pd.Series,
    h: int = 1,
) -> tuple[float, float]:
    """Clark-West (2007) MSPE-adjusted test of equal accuracy for *nested* models.

    The Diebold-Mariano test is non-standard when one model nests the other (any model
    against a random walk or the historical mean; a rich model against a restricted
    special case of itself). Under the null the extra parameters are pure estimation
    noise, so they inflate the larger model's MSPE and DM is undersized. Clark-West adds
    back the ``(f_restricted - f_unrestricted)^2`` adjustment that removes exactly that
    noise.

    Pass the realised target and the forecast *levels* of the restricted (smaller,
    benchmark) and unrestricted (larger) models. Returns ``(CW-statistic, one-sided
    p-value)`` for H0: equal MSPE (the extra predictors add nothing) against H1: the
    unrestricted model is more accurate. Reject for large positive statistics; the
    statistic is referred to the standard normal, as in Clark and West (2007).
    """
    y = np.asarray(y_true, dtype=float).ravel()
    fr = np.asarray(f_restricted, dtype=float).ravel()
    fu = np.asarray(f_unrestricted, dtype=float).ravel()
    mask = np.isfinite(y) & np.isfinite(fr) & np.isfinite(fu)
    y, fr, fu = y[mask], fr[mask], fu[mask]
    f_adj = (y - fr) ** 2 - ((y - fu) ** 2 - (fr - fu) ** 2)
    T = len(f_adj)
    if T < 8:
        return float("nan"), float("nan")
    f_bar = f_adj.mean()
    # Newey-West-style HAC variance with bandwidth h-1 (an optimal h-step error is MA(h-1))
    var_f = np.var(f_adj, ddof=0)
    for k in range(1, h):
        cov = np.mean((f_adj[k:] - f_bar) * (f_adj[:-k] - f_bar))
        var_f += 2.0 * cov
    var_f = max(var_f, 1e-12)
    cw = f_bar / np.sqrt(var_f / T)
    pval = float(1.0 - stats.norm.cdf(cw))  # one-sided: large positive => reject
    return float(cw), pval


def evaluation_table(
    y_true,
    forecasts: dict[str, "np.ndarray | pd.Series"],
) -> pd.DataFrame:
    """Build a tidy RMSE/MAE table for a dict of {model_name: yhat}."""
    rows = []
    for name, yhat in forecasts.items():
        rows.append({"model": name, "RMSE": rmse(y_true, yhat), "MAE": mae(y_true, yhat)})
    return pd.DataFrame(rows).set_index("model")
