"""Bootstrapped predictive bands from realised forecast errors.

A point forecast is only half an answer. These helpers turn a model's *own track
record* into an honest interval: for each horizon we take the forecast errors the
model actually made, bootstrap their empirical quantiles, and hang the band on the
point forecast. No normality is assumed, so a skewed error distribution produces an
asymmetric band, which is exactly what a series like investment growth deserves.

Two rules keep the bands honest:

* **No look-ahead.** Only errors whose target had already been *published* at the
  forecast origin are used (target quarter end + publication delay <= origin).
* **COVID excluded.** The 2020 to 2021 collapse and rebound are a structural break;
  leaving them in would inflate every band by a factor of several.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def bootstrap_quantiles(
    errors: Sequence[float],
    *,
    level: float = 0.90,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bagged bootstrap quantiles of ``errors`` at the given central ``level``.

    Resampling with replacement and averaging the quantiles smooths the tail
    estimates, which are otherwise very noisy in samples of a few dozen errors.
    Returns ``(lo, hi)`` offsets to add to a point forecast.
    """
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 5:
        return (np.nan, np.nan)
    a = (1.0 - level) / 2.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(n_boot, len(e)))
    draws = e[idx]
    lo = np.quantile(draws, a, axis=1).mean()
    hi = np.quantile(draws, 1.0 - a, axis=1).mean()
    return float(lo), float(hi)


def horizon_bands(
    bt: pd.DataFrame,
    model: str,
    *,
    as_of: str | pd.Timestamp,
    by: str = "horizon",
    lookback_years: int = 10,
    exclude_years: Sequence[int] = (2020, 2021),
    level: float = 0.90,
    n_boot: int = 2000,
    delay_days: int = 52,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrapped band offsets, per group, from errors known at ``as_of``.

    ``by`` selects the grouping: ``"horizon"`` for a multi-horizon backtest, or
    ``"rel_month"`` (or ``"origin_label"``) for a vintage backtest, where accuracy
    improves as the quarter fills and each vintage position deserves its own band.

    Parameters
    ----------
    bt:
        Multi-horizon backtest frame (see :func:`MIDAS.backtest.run_horizon_backtest`).
    as_of:
        The forecast origin. Only errors whose target was already published at this
        date enter, so the bands are constructible in real time.
    lookback_years:
        Use the most recent ``lookback_years`` of (publishable) errors.

    Returns
    -------
    DataFrame indexed by ``horizon`` with ``lo``, ``hi``, ``rmse`` and ``n``.
    """
    as_of = pd.Timestamp(as_of)
    d = bt[bt.model == model].copy()
    d["err"] = d.y_true - d.y_hat  # actual minus forecast: band = point + quantile(err)
    published = d.ref_quarter + pd.offsets.MonthEnd(1) + pd.Timedelta(days=delay_days)
    d = d[published <= as_of]                                   # no look-ahead
    d = d[~d.ref_quarter.dt.year.isin(list(exclude_years))]     # COVID is not noise
    d = d[d.ref_quarter >= as_of - pd.DateOffset(years=lookback_years)]

    rows = []
    for key, g in d.groupby(by):
        e = g["err"].dropna().to_numpy()
        lo, hi = bootstrap_quantiles(e, level=level, n_boot=n_boot, seed=seed)
        rows.append({by: key, "lo": lo, "hi": hi,
                     "rmse": float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
                     "n": len(e)})
    return pd.DataFrame(rows).set_index(by).sort_index()
