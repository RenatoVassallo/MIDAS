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


def release_cycle_bands(rc, model, *, index_col="info_index", n_bins=5,
                        levels=(0.5, 0.68, 0.9), exclude_years=(), min_quarters=10):
    """Real-time empirical predictive bands for a model, conditioned on the Information Index.

    The honest, central-bank-standard nowcast-uncertainty measure for a combination: at each origin,
    the interval at probability ``level`` is the empirical quantile band of the model's past
    out-of-sample residuals (``y_true - y_hat``) from EARLIER quarters in the same Information-Index
    bin, excluding ``exclude_years`` (e.g. COVID). One residual per (past quarter, bin) is used so a
    quarter with many origins in a bin is not over-weighted. Adds ``lo_<level>``/``hi_<level>``
    columns and a ``pit`` (probability integral transform) column for calibration testing; a
    well-calibrated density has uniform PIT and coverage matching the nominal level.
    """
    import numpy as np
    import pandas as pd

    d = rc[rc.model == model].copy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    d["bin"] = np.clip(np.digitize(d[index_col].to_numpy(), edges[1:-1]), 0, n_bins - 1)
    d["resid"] = d.y_true - d.y_hat
    d["yr"] = pd.to_datetime(d.ref_quarter).dt.year
    perqb = d.groupby(["ref_quarter", "bin"], as_index=False).resid.mean()
    perqb["yr"] = pd.to_datetime(perqb.ref_quarter).dt.year
    out = []
    for Q in sorted(d.ref_quarter.unique()):
        for b, g in d[d.ref_quarter == Q].groupby("bin"):
            past = perqb[(perqb.ref_quarter < Q) & (perqb.bin == b) & (~perqb.yr.isin(exclude_years))].resid.dropna()
            g = g.copy()
            if len(past) >= min_quarters:
                for lv in levels:
                    a = (1 - lv) / 2
                    g[f"lo_{lv}"] = g.y_hat + past.quantile(a)
                    g[f"hi_{lv}"] = g.y_hat + past.quantile(1 - a)
                g["pit"] = (past.to_numpy()[None, :] <= g.resid.to_numpy()[:, None]).mean(axis=1)
            else:
                for lv in levels:
                    g[f"lo_{lv}"] = np.nan; g[f"hi_{lv}"] = np.nan
                g["pit"] = np.nan
            out.append(g)
    return pd.concat(out, ignore_index=True)
