"""Pseudo-real-time expanding-window backtest harness.

For each target quarter in the evaluation window and each intra-quarter origin
(begin / middle / end of the relevant months), the harness reconstructs the
information set at that origin, then fits and nowcasts every model. The result is
one tidy long frame that feeds every evaluation view, including the signature
nowcast-evolution curve (accuracy as a function of how much of the quarter has
been released).

Origins are labelled by their position *relative to the target quarter's end
month* (``rel_month`` in months, ``day_part`` in {begin, mid, end}) so that the
same label is comparable across quarters. ``lead_days`` is the signed distance in
days from the origin to the quarter end (positive = before quarter end).

Unlike the legacy loops in :mod:`MIDAS.forecasting`, estimation failures are
counted and summarised rather than silently emitted as ``NaN``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .base import BaseNowcaster
from .metadata import MetadataPanel
from .realtime import RealtimeEngine

from collections.abc import Sequence
import pandas as pd

DAY_PARTS: tuple[tuple[str, int], ...] = (
    ("begin", 1),
    ("mid", 15),
    ("end", 27),
)

MONTH_OFFSETS: tuple[int, ...] = (-2, -1, 0, 1, 2)


def quarter_end(quarter: pd.Timestamp) -> pd.Timestamp:
    """Last day of the quarter."""
    return pd.Timestamp(quarter) + pd.offsets.MonthEnd(1)


def make_origin_grid(
    quarter: pd.Timestamp,
    month_offsets: Sequence[int] = MONTH_OFFSETS,
    day_parts: Sequence[tuple[str, int]] = DAY_PARTS,
    *,
    daily: bool = False,
) -> list[tuple[pd.Timestamp, int, str]]:
    """
    Origins for nowcasting `quarter`.

    Returns
    -------
    list of tuples:
        (origin_date, relative_month, day_label)
    """
    end_month_start = pd.Timestamp(quarter)
    grid = []

    for off in month_offsets:
        month_start = end_month_start + pd.DateOffset(months=off)

        if daily:
            month_end = month_start + pd.offsets.MonthEnd(0)

            for origin in pd.date_range(
                start=month_start,
                end=month_end,
                freq="D",
            ):
                grid.append(
                    (
                        origin,
                        off,
                        origin.strftime("%d"),
                    )
                )

        else:
            for label, day in day_parts:
                origin = month_start.replace(day=day)
                grid.append((origin, off, label))

    return grid


def make_daily_origin_grid(
    publication_date: pd.Timestamp,
    days_before: int = 180,
) -> list[pd.Timestamp]:
    """Daily forecast origins from `days_before` through publication day."""
    publication_date = pd.Timestamp(publication_date)

    return list(
        pd.date_range(
            start=publication_date - pd.Timedelta(days=days_before),
            end=publication_date,
            freq="D",
        )
    )

def quarter_timestamp(period: pd.Period) -> pd.Timestamp:
    """Panel convention: a quarter is dated at the first day of its end month."""
    return pd.Timestamp(period.end_time).normalize().replace(day=1)


def run_horizon_backtest(
    panel: MetadataPanel,
    target: str,
    models: Mapping[str, BaseNowcaster],
    *,
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    horizons: Sequence[int] = tuple(range(9)),
    model_horizons: Mapping[str, Sequence[int]] | None = None,
    origin_day: int = 1,
    min_train: int = 20,
    verbose: bool = False,
) -> pd.DataFrame:
    """Multi-horizon backtest with a single origin per quarter.

    For each base quarter ``Q0`` the origin is ``origin_day`` of the first month of the
    *next* quarter (early January / April / July / October). At that instant ``Q0`` has
    just ended and is not yet published (the target carries a ~52 day delay), so ``h=0``
    is a genuine nowcast of the just-ended quarter and ``h>=1`` are true forecasts.

    ``model_horizons`` optionally restricts horizons per model, e.g. for an expensive
    model that must refit per horizon.

    Returns a tidy frame with ``[target, base_quarter, horizon, ref_quarter, origin_date,
    model, y_true, y_hat, y_std]``; only quarters with a realised value are scored.
    """
    engine = RealtimeEngine(panel)
    q_index = panel.quarterly.index
    y_full = panel.quarterly[target]
    eval_start = pd.Timestamp(eval_start)
    eval_end = q_index[-1] if eval_end is None else pd.Timestamp(eval_end)
    base_quarters = q_index[(q_index >= eval_start) & (q_index <= eval_end)]

    rows: list[dict] = []
    failures: dict[str, int] = {}
    for Q0 in base_quarters:
        p0 = pd.Period(Q0, freq="Q")
        origin = pd.Timestamp((p0 + 1).start_time).normalize().replace(day=origin_day)
        for name, model in models.items():
            for h in (model_horizons or {}).get(name, horizons):
                ts = quarter_timestamp(p0 + h)
                y_true = y_full.get(ts, np.nan)
                if not np.isfinite(y_true):
                    continue  # no realised value to score against
                info = engine.information_set(origin, target, target_period=ts)
                if info.observed_quarters().size < min_train:
                    continue
                try:
                    res = model.fit(info).nowcast(info)
                    yhat, ystd = res.mean, res.std
                except Exception as exc:  # never swallow silently
                    yhat, ystd = float("nan"), None
                    failures[name] = failures.get(name, 0) + 1
                    if verbose:
                        print(f"[{name}] Q0={Q0.date()} h={h}: {type(exc).__name__}: {exc}")
                if not np.isfinite(yhat):
                    failures[name] = failures.get(name, 0) + 1
                rows.append({
                    "target": target, "base_quarter": Q0, "horizon": int(h),
                    "ref_quarter": ts, "origin_date": origin, "model": name,
                    "y_true": float(y_true), "y_hat": float(yhat),
                    "y_std": None if ystd is None else float(ystd),
                })
    if failures:
        print("[run_horizon_backtest] non-finite/failed by model -> "
              + ", ".join(f"{n}: {c}" for n, c in failures.items()))
    return pd.DataFrame(rows)


def run_backtest(
    panel: MetadataPanel,
    target: str,
    models: Mapping[str, BaseNowcaster],
    *,
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    month_offsets: Sequence[int] = MONTH_OFFSETS,
    day_parts: Sequence[tuple[str, int]] = DAY_PARTS,
    min_train: int = 20,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the expanding-window pseudo-real-time backtest.

    Parameters
    ----------
    panel:
        The (unmasked) :class:`MetadataPanel`; ``y_true`` is read from it.
    target:
        Quarterly target column, for example ``"g_invq"``.
    models:
        Mapping ``name -> BaseNowcaster``. Each is refit at every origin.
    eval_start, eval_end:
        First / last target quarter to evaluate (inclusive). ``eval_end`` may be
        ``None`` for "through the end of the sample".
    min_train:
        Skip an origin if fewer than this many target quarters are released at it.

    Returns
    -------
    DataFrame with columns ``[target, ref_quarter, origin_date, rel_month,
    day_part, origin_label, lead_days, model, y_true, y_hat, y_std]``.
    """
    engine = RealtimeEngine(panel)
    q_index = panel.quarterly.index
    y_full = panel.quarterly[target]

    eval_start = pd.Timestamp(eval_start)
    eval_end = q_index[-1] if eval_end is None else pd.Timestamp(eval_end)
    eval_quarters = q_index[(q_index >= eval_start) & (q_index <= eval_end)]

    rows: list[dict] = []
    failures: dict[str, int] = {name: 0 for name in models}
    for Q in eval_quarters:
        y_true = y_full.loc[Q]
        if pd.isna(y_true):
            continue  # no realised value to score against
        q_end = quarter_end(Q)
        for origin, rel_month, day_part in make_origin_grid(Q, month_offsets, day_parts):
            info = engine.information_set(origin, target, target_period=Q)
            # A nowcast is only meaningful before the target is released. If Q is
            # already out at this origin, skip it (reading the realised value would
            # not be a forecast). The pre-release backcast window is retained.
            if pd.notna(info.quarterly.at[Q, target]):
                continue
            if info.observed_quarters().size < min_train:
                continue
            for name, model in models.items():
                try:
                    res = model.fit(info).nowcast(info)
                    yhat, ystd = res.mean, res.std
                except Exception as exc:  # never swallow silently
                    yhat, ystd = float("nan"), None
                    failures[name] += 1
                    if verbose:
                        print(f"[{name}] {Q.date()} @ {origin.date()}: {type(exc).__name__}: {exc}")
                if not np.isfinite(yhat):
                    failures[name] += 1
                rows.append(
                    {
                        "target": target,
                        "ref_quarter": Q,
                        "origin_date": origin,
                        "rel_month": rel_month,
                        "day_part": day_part,
                        "origin_label": f"{rel_month:+d}:{day_part}",
                        "lead_days": (q_end - origin).days,
                        "model": name,
                        "y_true": float(y_true),
                        "y_hat": float(yhat),
                        "y_std": None if ystd is None else float(ystd),
                    }
                )

    total = {name: cnt for name, cnt in failures.items() if cnt}
    if total:
        summary = ", ".join(f"{n}: {c}" for n, c in total.items())
        print(f"[run_backtest] non-finite/failed nowcasts by model -> {summary}")
    return pd.DataFrame(rows)
