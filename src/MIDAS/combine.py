"""Forecast combination.

Two entry points, for the two different jobs:

``CombinationNowcaster``
    The *live* path: wraps member nowcasters, fits them on one information set and
    returns a weighted average. Weights are fixed (equal by default), because at a
    single origin a model has no history of its own past errors to learn from.

``combine_backtest``
    The *evaluation* path: given the tidy backtest frame with every member's
    nowcasts, it adds a combination model computed with **real-time** weights. For
    each target quarter it uses only past quarters whose actual had already been
    released at that origin, and compares members at the *same* vintage position
    (``origin_label``), since relative accuracy varies a lot across the quarter.

Supported schemes: ``equal`` (the classic, hard-to-beat benchmark), ``inv_mse`` and
``inv_rmse`` (performance weights), ``median`` (robust to one member blowing up),
and ``best`` (pick the historically best member, a selection rather than a blend).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .base import BaseNowcaster, InformationSet, NowcastResult

METHODS = ("equal", "inv_mse", "inv_rmse", "median", "best")


class CombinationNowcaster(BaseNowcaster):
    """Weighted average of member nowcasters (live path).

    ``weights`` maps member name to a non-negative weight (renormalised over the
    members that actually produce a finite nowcast, so the combination still works
    when one member is blind at an early origin). ``None`` means equal weights.
    """

    def __init__(
        self,
        members: Mapping[str, BaseNowcaster],
        weights: Mapping[str, float] | None = None,
        name: str | None = None,
    ) -> None:
        self.members = dict(members)
        self.weights = None if weights is None else dict(weights)
        self._name = name or ("Combo(" + ", ".join(self.members) + ")")

    def fit(self, info: InformationSet) -> "CombinationNowcaster":
        for m in self.members.values():
            m.fit(info)
        return self

    def nowcast(self, info: InformationSet) -> NowcastResult:
        vals, ws, used = [], [], []
        for key, model in self.members.items():
            try:
                r = model.nowcast(info)
            except Exception:
                continue
            if np.isfinite(r.mean):
                vals.append(float(r.mean))
                ws.append(float(self.weights.get(key, 1.0)) if self.weights else 1.0)
                used.append(key)
        if not vals:
            return NowcastResult(mean=float("nan"), model=self.name)
        w = np.asarray(ws, dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.repeat(1.0 / len(w), len(w))
        return NowcastResult(
            mean=float(np.dot(w, vals)),
            model=self.name,
            extra={"weights": dict(zip(used, np.round(w, 3)))},
        )


# ------------------------------------------------------------------ evaluation
def _weights_from_errors(err: pd.DataFrame, method: str) -> np.ndarray:
    """Weights over columns of ``err`` (past forecast errors, rows = past quarters)."""
    n = err.shape[1]
    if method == "equal" or err.empty:
        return np.repeat(1.0 / n, n)
    mse = (err ** 2).mean(axis=0)
    if not np.isfinite(mse).all() or (mse <= 0).any():
        return np.repeat(1.0 / n, n)
    if method == "inv_mse":
        w = 1.0 / mse
    elif method == "inv_rmse":
        w = 1.0 / np.sqrt(mse)
    elif method == "best":
        w = np.zeros(n)
        w[int(np.argmin(mse.to_numpy()))] = 1.0
        return w
    else:
        raise ValueError(f"unknown method {method!r}")
    return (w / w.sum()).to_numpy()


def combine_backtest(
    bt: pd.DataFrame,
    members: Sequence[str],
    *,
    method: str = "inv_mse",
    name: str | None = None,
    min_train: int = 8,
    delay_days: int = 52,
    return_weights: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Add a combination model to a backtest frame using real-time weights.

    Parameters
    ----------
    bt:
        Tidy backtest frame (see :func:`MIDAS.backtest.run_backtest`) containing
        every member in ``members``.
    method:
        One of :data:`METHODS`.
    min_train:
        Minimum number of past quarters with released actuals before performance
        weights are used; below it the combination falls back to equal weights.
    delay_days:
        Publication delay of the target, used to decide which past actuals were
        already known at each origin (no look-ahead in the weights).

    Returns
    -------
    A frame with the same schema as ``bt`` and ``model`` set to ``name``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    name = name or f"Combo-{method}({'+'.join(members)})"

    idx = ["ref_quarter", "origin_date", "rel_month", "day_part", "origin_label", "lead_days"]
    wide = bt.pivot_table(index=idx, columns="model", values="y_hat", aggfunc="first")
    truth = bt.pivot_table(index=idx, values="y_true", aggfunc="first")["y_true"]
    missing = [m for m in members if m not in wide.columns]
    if missing:
        raise KeyError(f"members not in backtest frame: {missing}")

    wide = wide[list(members)].join(truth)
    wide = wide.reset_index()
    wide["release"] = wide.ref_quarter + pd.offsets.MonthEnd(1) + pd.Timedelta(days=delay_days)
    err = wide[list(members)].sub(wide.y_true, axis=0)
    for m in members:
        wide[f"_e_{m}"] = err[m]

    rows, wrows = [], []
    for label, block in wide.groupby("origin_label", sort=False):
        block = block.sort_values("ref_quarter").reset_index(drop=True)
        ecols = [f"_e_{m}" for m in members]
        for i, r in block.iterrows():
            # past quarters at this same vintage whose actual was released by this origin
            past = block.loc[: max(i - 1, 0)] if i > 0 else block.iloc[0:0]
            past = past[(past.ref_quarter < r.ref_quarter) & (past.release <= r.origin_date)]
            past_err = past[ecols].dropna()
            use = "equal" if len(past_err) < min_train else method
            vals = np.array([r[m] for m in members], dtype=float)
            ok = np.isfinite(vals)
            if not ok.any():
                yhat = np.nan
            elif method == "median":
                yhat = float(np.median(vals[ok]))
            else:
                w = _weights_from_errors(past_err.loc[:, np.array(ecols)[ok]], use)
                yhat = float(np.dot(w, vals[ok]))
                wrows.append({"ref_quarter": r.ref_quarter, "origin_date": r.origin_date,
                              "origin_label": r.origin_label, "scheme": use,
                              **dict(zip(np.array(list(members))[ok], np.round(w, 4)))})
            rows.append({
                "target": bt.target.iloc[0] if "target" in bt.columns else None,
                "ref_quarter": r.ref_quarter, "origin_date": r.origin_date,
                "rel_month": r.rel_month, "day_part": r.day_part,
                "origin_label": r.origin_label, "lead_days": r.lead_days,
                "model": name, "y_true": r.y_true, "y_hat": yhat, "y_std": None,
            })
    out = pd.DataFrame(rows)
    return (out, pd.DataFrame(wrows)) if return_weights else out
