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

try:  # progress bars are an optional convenience, never a hard requirement
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    class tqdm:  # noqa: N801  minimal no-op stand-in
        def __init__(self, iterable=None, **kwargs):
            self._it = [] if iterable is None else iterable

        def __iter__(self):
            return iter(self._it)

        def set_postfix_str(self, *args, **kwargs):
            pass

        @staticmethod
        def write(*args, **kwargs):
            print(*args)

from .base import BaseNowcaster
from .evaluation import rmse
from .metadata import MetadataPanel
from .realtime import RealtimeEngine

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
) -> list[tuple[pd.Timestamp, int, str]]:
    """Coarse intra-quarter origins for nowcasting ``quarter``.

    One origin per (month offset, day-part), e.g. begin/mid/end across the months
    around the quarter. For the finer, continuous release-cycle sweep use
    :func:`make_daily_origin_grid` with :func:`run_release_cycle_backtest`.

    Returns a list of ``(origin_date, relative_month, day_label)``.
    """
    end_month_start = pd.Timestamp(quarter)
    grid = []
    for off in month_offsets:
        month_start = end_month_start + pd.DateOffset(months=off)
        for label, day in day_parts:
            grid.append((month_start.replace(day=day), off, label))
    return grid


def publication_date(target_period: pd.Timestamp, delay_days: int) -> pd.Timestamp:
    """Official publication date of a target quarter: quarter end + ``delay_days``.

    ``target_period`` follows the panel convention (first day of the quarter's end
    month), so the quarter end is ``target_period + MonthEnd(1)``.
    """
    return quarter_end(target_period) + pd.Timedelta(days=int(delay_days))


def make_daily_origin_grid(
    publication_date: pd.Timestamp,
    days_before: int = 180,
    *,
    days_after: int = 0,
    step_days: int = 1,
) -> list[pd.Timestamp]:
    """Daily (or ``step_days``-spaced) forecast origins across the release cycle.

    Sweeps from ``days_before`` days before the target's official ``publication_date``
    to ``days_after`` days after it. This is the x-axis of a Bank-of-England-style
    'horse race through the release cycle': plotting a model's RMSE against
    ``days_to_publication = origin - publication_date`` (negative before publication)
    shows accuracy improving in a step pattern as each release clears its lag.

    Returns a list of origin timestamps.
    """
    pub = pd.Timestamp(publication_date)
    return list(pd.date_range(start=pub - pd.Timedelta(days=days_before),
                              end=pub + pd.Timedelta(days=days_after),
                              freq=f"{int(step_days)}D"))


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



def _release_cycle_quarter(
    Q: pd.Timestamp,
    panel: MetadataPanel,
    target: str,
    models: Mapping[str, BaseNowcaster],
    *,
    delay: int,
    days_before: int,
    days_after: int,
    step_days: int,
    min_train: int,
    verbose: bool,
) -> tuple[list[dict], dict[str, int]]:
    """One quarter's full release-cycle sweep.

    Quarters are independent, so this is the unit of parallel work. The models are
    **deep-copied** here so each quarter (and each worker process) gets fresh, unshared
    state; behaviour is identical to the sequential loop because every model already
    re-estimates per quarter. Returns ``(rows, failures)``.
    """
    import copy

    engine = RealtimeEngine(panel)
    y_true = panel.quarterly[target].loc[Q]
    if pd.isna(y_true):
        return [], {}
    models = {name: copy.deepcopy(m) for name, m in models.items()}

    pub = publication_date(Q, delay)
    q_end = quarter_end(Q)
    origins = make_daily_origin_grid(pub, days_before=days_before, days_after=days_after,
                                     step_days=step_days)

    rows: list[dict] = []
    failures: dict[str, int] = {}
    last_count: int | None = None
    cached: dict[str, tuple[float, float | None]] = {}

    for origin in origins:
        info = engine.information_set(origin, target, target_period=Q)
        if pd.notna(info.quarterly.at[Q, target]):
            continue
        if info.observed_quarters().size < min_train:
            continue
        # The information set only grows as the origin advances, so an unchanged non-NaN
        # cell count means an identical panel: refit only at these changepoints.
        count = int(info.monthly.notna().to_numpy().sum() + info.quarterly.notna().to_numpy().sum())
        if count != last_count:
            for name, model in models.items():
                try:
                    res = model.fit(info).nowcast(info)
                    cached[name] = (res.mean, res.std)
                except Exception as exc:
                    cached[name] = (float("nan"), None)
                    failures[name] = failures.get(name, 0) + 1
                    if verbose:
                        print(f"[{name}] {Q.date()} @ {origin.date()}: {type(exc).__name__}: {exc}")
            last_count = count
        for name, (yhat, ystd) in cached.items():
            rows.append({
                "target": target, "ref_quarter": Q, "origin_date": origin,
                "days_to_publication": (origin - pub).days,
                "days_to_quarter_end": (origin - q_end).days,
                "model": name, "y_true": float(y_true), "y_hat": float(yhat),
                "y_std": None if ystd is None else float(ystd),
            })
    return rows, failures


def run_release_cycle_backtest(
    panel: MetadataPanel,
    target: str,
    models: Mapping[str, BaseNowcaster],
    *,
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    days_before: int = 180,
    days_after: int = 0,
    step_days: int = 1,
    min_train: int = 20,
    verbose: bool = False,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Backtest across the full release cycle at daily resolution.

    Each quarter's 180-day sweep is independent, so set ``n_jobs > 1`` (or ``-1`` for every
    core) to run the quarters in parallel with joblib; the models are deep-copied per quarter
    so nothing is shared between workers. ``n_jobs=1`` keeps the sequential path with a
    progress bar. Expensive models (the DFM refits per quarter, the pooled MIDAS and SC-MIDAS
    refit per release) dominate the cost, so parallelism gives a near-linear speed-up.
    """
    q_index = panel.quarterly.index
    delay = panel.delay_of(target)
    eval_start = pd.Timestamp(eval_start)
    eval_end = q_index[-1] if eval_end is None else pd.Timestamp(eval_end)
    quarters = list(q_index[(q_index >= eval_start) & (q_index <= eval_end)])

    kw = dict(delay=delay, days_before=days_before, days_after=days_after,
              step_days=step_days, min_train=min_train, verbose=verbose)

    rows: list[dict] = []
    failures: dict[str, int] = {}
    if n_jobs == 1:
        for Q in tqdm(quarters, desc="Release-cycle backtest", unit="quarter",
                      disable=not show_progress, dynamic_ncols=True):
            r, f = _release_cycle_quarter(Q, panel, target, models, **kw)
            rows.extend(r)
            for k, v in f.items():
                failures[k] = failures.get(k, 0) + v
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=10 if show_progress else 0)(
            delayed(_release_cycle_quarter)(Q, panel, target, models, **kw) for Q in quarters
        )
        for r, f in results:
            rows.extend(r)
            for k, v in f.items():
                failures[k] = failures.get(k, 0) + v

    if failures:
        print("[run_release_cycle_backtest] failed nowcasts by model -> "
              + ", ".join(f"{name}: {count}" for name, count in failures.items()))
    return pd.DataFrame(rows)


def release_cycle_rmse(
    bt: pd.DataFrame,
    models: Sequence[str],
    *,
    by: str = "days_to_publication",
    exclude_years: Sequence[int] = (),
) -> pd.DataFrame:
    """RMSE of each model as a function of ``by`` (default: days to publication).

    The index is the lead coordinate and the columns are the models: the data behind the
    horse-race step chart (:func:`MIDAS.plot_release_cycle_rmse`). Each model is scored on
    its own finite rows; with a daily grid coverage is near-complete, so this is close to
    a matched comparison.
    """
    d = bt[~bt.ref_quarter.dt.year.isin(list(exclude_years))]
    out = {m: d[d.model == m].groupby(by).apply(lambda x: rmse(x.y_true, x.y_hat), include_groups=False)
           for m in models}
    return pd.DataFrame(out).sort_index()
