"""Reusable, didactic plots for pseudo-real-time nowcast evaluation.

All functions take the tidy long backtest frame produced by
:func:`MIDAS.backtest.run_backtest` (columns ``model, ref_quarter, origin_date,
rel_month, day_part, lead_days, y_true, y_hat, y_std``) so notebooks and the slide
deck draw the same figures from one source. Colours follow ``MIDAS.plotting.PALETTE``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plotting import PALETTE

# Consistent colour / weight per model across every figure.
_STYLE = {
    # Proposed models (emphasized)
    "DFM": dict(
        color=PALETTE.get("linknavy", "#28468C"),
        lw=2.8,
        zorder=6,
    ),

    "sg-LASSO": dict(
        color="#D4A000",          # darker gold
        lw=2.8,
        zorder=6,
    ),

    "GB-MIDAS": dict(
        color="#762A83",          # purple: the non-linear ML entry
        lw=2.6,
        zorder=6,
    ),

    "SC-MIDAS": dict(
        color=PALETTE.get("mygreen", "#2E7D32"),
        lw=2.5,
        zorder=5,
    ),

    # Benchmarks
    "Bridge": dict(
        color="#E67E22",          # vivid orange
        lw=1.9,
    ),

    "P-MIDAS": dict(
        color="#7B68B3",          # richer purple
        lw=1.9,
    ),

    "Q-AR": dict(
        color=PALETTE.get("myred", "#C0392B"),
        lw=1.9,
    ),

    "M-AR": dict(
        color="#4F81BD",          # steel blue
        lw=1.9,
    ),

    "RW": dict(
        color="0.45",
        lw=1.6,
        ls="--",
    ),

    "Mean": dict(
        color="0.72",
        lw=1.4,
        ls=":",
    ),
}


def _style(model: str) -> dict:
    return dict(_STYLE.get(model, dict(lw=1.8)))


def _quarter_label(q) -> str:
    return pd.Timestamp(q).to_period("Q").strftime("%YQ%q")


# --------------------------------------------------------------- vintage evolution
def plot_vintage_evolution(
    bt: pd.DataFrame,
    quarter: str | pd.Timestamp,
    models: Sequence[str],
    *,
    std_models: Sequence[str] = (),
    band_z: float = 1.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """For one target quarter, plot each model's nowcast as vintages arrive.

    The x-axis is the forecast origin (data accumulates left to right); the dashed
    horizontal line is the realised value. Models in ``std_models`` get a shaded
    ``band_z``-sigma band from ``y_std`` (off by default to keep model comparisons
    legible; use a dedicated fan chart for uncertainty).
    """
    q = pd.Timestamp(quarter)
    sub = bt[bt.ref_quarter == q]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.6))
    actual = float(sub.y_true.dropna().iloc[0])
    for m in models:
        d = sub[sub.model == m].sort_values("origin_date")
        if d.empty:
            continue
        x = d.origin_date.to_numpy()
        ax.plot(x, d.y_hat, marker="o", ms=4, label=m, **_style(m))
        if m in std_models and d.y_std.notna().any():
            lo = d.y_hat - band_z * d.y_std
            hi = d.y_hat + band_z * d.y_std
            ax.fill_between(x, lo, hi, color=_style(m)["color"], alpha=0.15, zorder=1)
    ax.axhline(actual, color="0.15", ls="--", lw=1.2, label=f"actual = {actual:.1f}")
    ax.set_xlabel("forecast origin (vintage)")
    ax.set_ylabel("nowcast of YoY growth (%)")
    ax.set_title(title or f"Nowcast evolution — {_quarter_label(q)}")
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=30)
    return ax


def plot_quarters_grid(
    bt: pd.DataFrame,
    quarters: Sequence[str | pd.Timestamp],
    models: Sequence[str],
    *,
    std_models: Sequence[str] = (),
    labels: Sequence[str] | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """Small multiples: one vintage-evolution panel per quarter (shared story)."""
    n = len(quarters)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for i, (q, ax) in enumerate(zip(quarters, axes)):
        plot_vintage_evolution(bt, q, models, std_models=std_models, ax=ax)
        if labels is not None:
            ax.set_title(f"{_quarter_label(q)} — {labels[i]}")
        if i > 0:
            ax.set_ylabel("")
        if i < n - 1:
            ax.legend().remove()
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- tracking
def plot_tracking(
    bt: pd.DataFrame,
    models: Sequence[str],
    *,
    rel_month: int = 0,
    day_part: str = "mid",
    shade: Sequence[tuple[str, str]] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Actual vs each model's nowcast at a fixed intra-quarter vintage, over time."""
    sub = bt[(bt.rel_month == rel_month) & (bt.day_part == day_part)]
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.2))
    truth = sub[sub.model == models[0]].sort_values("ref_quarter")
    ax.plot(truth.ref_quarter, truth.y_true, color="0.15", lw=2.2, label="actual", zorder=6)
    for m in models:
        d = sub[sub.model == m].sort_values("ref_quarter")
        ax.plot(d.ref_quarter, d.y_hat, label=m, **_style(m))
    for lo, hi in (shade or []):
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="0.88", zorder=0)
    ax.set_ylabel("YoY growth (%)")
    ax.set_title(title or f"Tracking the target (vintage: rel_month {rel_month:+d}, {day_part})")
    ax.legend(fontsize=8, ncol=len(models) + 1, loc="lower left")
    return ax


def plot_vintage_tracking(
    bt: pd.DataFrame,
    model: str,
    *,
    quarters: Sequence | None = None,
    last: int = 10,
    bands: pd.DataFrame | None = None,
    band_key: str = "rel_month",
    ax: plt.Axes | None = None,
    title: str | None = None,
    label_quarters: bool = True,
) -> plt.Axes:
    """GDPNow-style tracking chart: one nowcast path per target quarter.

    The x-axis is calendar time (the vintage/origin date) and each target quarter gets its
    own path, restarting when a new quarter begins, exactly as the Atlanta Fed's GDPNow and
    the New York Fed's Staff Nowcast are presented. The realised value is drawn as a black
    dashed segment across that quarter's vintage window, so the reader sees each path
    converging (or not) on its own target.

    ``bands`` is indexed by ``band_key`` (usually ``rel_month``) with ``lo``/``hi`` offsets,
    giving a wider band early in the quarter and a tighter one once the data are in.
    """
    d = bt[bt.model == model]
    qs = sorted(pd.unique(d.ref_quarter))[-last:] if quarters is None else [pd.Timestamp(q) for q in quarters]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.6))
    shades = [PALETTE.get("myblue", "#1f4e79"), "#5b9bd5"]
    published_first = None
    for i, q in enumerate(qs):
        g = d[d.ref_quarter == pd.Timestamp(q)].sort_values("origin_date")
        g = g[g.y_hat.notna()]
        if g.empty:
            continue
        c = shades[i % 2]
        if bands is not None:
            lo = g.y_hat.to_numpy() + bands["lo"].reindex(g[band_key]).to_numpy()
            hi = g.y_hat.to_numpy() + bands["hi"].reindex(g[band_key]).to_numpy()
            # a soft vertical separator between quarters keeps the overlapping bands legible
            ax.axvspan(g.origin_date.min(), g.origin_date.max(),
                       color=c, alpha=0.05, lw=0, zorder=0)
            ax.fill_between(g.origin_date, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
        ax.plot(g.origin_date, g.y_hat, color=c, lw=2.0, marker="o", ms=3.4, zorder=3,
                label=model if i == 0 else None)
        actual = float(g.y_true.iloc[0])
        if np.isfinite(actual):   # only draw the realised value once it exists
            ax.hlines(actual, g.origin_date.min(), g.origin_date.max(), color="0.12", lw=1.8,
                      ls="--", zorder=2, label="realised" if published_first is None else None)
            ax.plot([g.origin_date.max()], [actual], marker="D", ms=5, color="0.12", zorder=4)
            published_first = published_first or q
        if label_quarters:
            ax.annotate(pd.Timestamp(q).to_period("Q").strftime("%yQ%q"),
                        (g.origin_date.iloc[len(g) // 2], ax.get_ylim()[0]),
                        fontsize=7, color="0.4", ha="center", va="bottom")
    ax.set_xlabel("vintage (forecast origin)")
    ax.set_ylabel("YoY growth (%)")
    ax.set_title(title or f"Nowcast tracking by vintage: {model}")
    ax.legend(fontsize=8, loc="best")
    return ax


def plot_vintage_panels(
    bt: pd.DataFrame,
    model: str,
    *,
    quarters: Sequence,
    bands: pd.DataFrame | None = None,
    band_key: str = "rel_month",
    ncols: int = 4,
    suptitle: str | None = None,
) -> plt.Figure:
    """Small multiples: one panel per target quarter, the vintage path against the actual."""
    qs = [pd.Timestamp(q) for q in quarters]
    nrows = int(np.ceil(len(qs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.7 * nrows), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    d = bt[bt.model == model]
    color = PALETTE.get("myblue", "#1f4e79")
    for ax, q in zip(axes, qs):
        g = d[d.ref_quarter == q].sort_values("origin_date")
        g = g[g.y_hat.notna()]
        if g.empty:
            ax.axis("off"); continue
        x = range(len(g))
        if bands is not None:
            lo = g.y_hat.to_numpy() + bands["lo"].reindex(g[band_key]).to_numpy()
            hi = g.y_hat.to_numpy() + bands["hi"].reindex(g[band_key]).to_numpy()
            ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
        ax.plot(x, g.y_hat, color=color, lw=1.9, marker="o", ms=3.5)
        ax.axhline(float(g.y_true.iloc[0]), color="0.12", ls="--", lw=1.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels([pd.Timestamp(o).strftime("%b") for o in g.origin_date], fontsize=7)
        ax.set_title(f"{pd.Timestamp(q).to_period('Q').strftime('%YQ%q')}  (actual {g.y_true.iloc[0]:.1f})",
                     fontsize=9)
    for ax in axes[len(qs):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, y=1.01, fontsize=12)
    fig.tight_layout()
    return fig


def plot_forecast_fan(
    history: pd.Series,
    point: pd.Series,
    bands: pd.DataFrame,
    *,
    actual: pd.Series | None = None,
    origin: str | pd.Timestamp | None = None,
    inner_level: str | None = "50",
    ax: plt.Axes | None = None,
    color: str | None = None,
    label: str = "forecast",
    title: str | None = None,
) -> plt.Axes:
    """History, a forecast path and its bootstrapped fan; optionally what happened.

    ``point`` is indexed by target quarter (h=0 first). ``bands`` shares that index and
    carries absolute ``lo``/``hi`` columns (and optionally ``lo50``/``hi50`` for an inner
    band). Pass ``actual`` to overlay the realised path for a pseudo-out-of-sample test:
    the honest question is whether the fan covered the truth.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.8))
    color = color or PALETTE.get("myblue", "#1f4e79")

    ax.plot(history.index, history.values, color="0.15", lw=2.0, label="realised (known at origin)")
    # join the last realised point to the forecast path so the line is continuous
    x = list(point.index)
    if len(history):
        xs = [history.index[-1]] + x
        ys = [history.iloc[-1]] + list(point.values)
    else:
        xs, ys = x, list(point.values)

    if inner_level and f"lo{inner_level}" in bands.columns:
        ax.fill_between(x, bands[f"lo{inner_level}"], bands[f"hi{inner_level}"],
                        color=color, alpha=0.32, lw=0, label=f"{inner_level}% band")
    ax.fill_between(x, bands["lo"], bands["hi"], color=color, alpha=0.16, lw=0, label="90% band")
    ax.plot(xs, ys, color=color, lw=2.4, marker="o", ms=4, label=label)

    if actual is not None and len(actual):
        ax.plot(actual.index, actual.values, color=PALETTE.get("myred", "#c0392b"),
                lw=2.0, ls="--", marker="s", ms=4, label="what actually happened")
    if origin is not None:
        ax.axvline(pd.Timestamp(origin), color="0.45", ls=":", lw=1.4)
        ax.annotate("origin", (pd.Timestamp(origin), ax.get_ylim()[1]), fontsize=8,
                    color="0.35", ha="center", va="bottom")
    ax.axhline(0, color="0.75", lw=0.7, zorder=0)
    ax.set_ylabel("YoY growth (%)")
    ax.set_title(title or "Nowcast and forecast path with bootstrapped bands")
    ax.legend(fontsize=8, ncol=2)
    return ax


def plot_release_cycle_rmse(
    bt: pd.DataFrame,
    models: Sequence[str],
    *,
    by: str = "days_to_publication",
    exclude_years: Sequence[int] = (2020, 2021),
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Horse-race RMSE step chart across the release cycle (Bank of England style).

    Expects the frame from :func:`MIDAS.run_release_cycle_backtest`. The x-axis is
    ``by`` (days to publication, negative before publication); each model's RMSE is a
    step function that drops as releases arrive. Uses ``steps-post`` so a level holds
    until the next release, matching how the information set actually updates.
    """
    from .backtest import release_cycle_rmse

    curves = release_cycle_rmse(bt, models, by=by, exclude_years=exclude_years)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        if m in curves.columns:
            ax.plot(curves.index, curves[m], drawstyle="steps-post", label=m, **_style(m))
    ax.set_xlabel("Days to publication" if by == "days_to_publication" else by)
    ax.set_ylabel("RMSE (pp)")
    ax.set_title(title or "RMSEs through the release cycle")
    ax.set_xlim(-150, 0)
    ax.set_xticks([-150, -120, -90, -60, -30, 0])
    ax.legend(fontsize=8, ncol=2)
    return ax


def plot_rmse_by_origin(
    bt: pd.DataFrame,
    models: Sequence[str],
    *,
    exclude_years: Sequence[int] = (2020, 2021),
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Nowcast-evolution curves: RMSE as a function of the intra-quarter origin."""
    from .evaluation import rmse

    d = bt[~bt.ref_quarter.dt.year.isin(exclude_years)]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        dm = d[d.model == m]
        curve = dm.groupby("rel_month").apply(lambda x: rmse(x.y_true, x.y_hat), include_groups=False)
        ax.plot(curve.index, curve.values, marker="o", label=m, **_style(m))
    ax.set_xlabel("origin month relative to quarter end  (data accumulates left → right)")
    ax.set_ylabel("RMSE (ex-COVID)")
    ax.set_title(title or "Nowcast evolution: accuracy as the quarter fills")
    ax.set_xticks(sorted(d.rel_month.unique()))
    ax.legend(fontsize=8)
    return ax
