"""Interpretability plots: how a sparse model's chosen predictors evolve over time.

``plot_selection_evolution`` turns a tidy ``[quarter, variable, l2_norm, group]`` frame (one row
per selected predictor per origin, e.g. from :meth:`SparseMIDASNowcaster.coefficient_table` across
quarters) into a picture of *which* predictors carry the signal *when*: a heatmap of the top
predictors' importance through time, and a stacked-area view of importance share by variable group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting import PALETTE

# a stable colour per metadata group
_GROUP_PALETTE = ["#29466E", "#1f9e89", "#e08214", "#8073ac", "#c0392b",
                  "#2e7d32", "#5b9bd5", "#b58900", "#6c757d", "#d33682"]


def _group_colors(groups) -> dict:
    uniq = list(dict.fromkeys(g for g in groups if g))
    return {g: _GROUP_PALETTE[i % len(_GROUP_PALETTE)] for i, g in enumerate(uniq)}


def _period(sel: pd.DataFrame, period: str) -> pd.Series:
    q = pd.to_datetime(sel["quarter"])
    return q.dt.year if period == "year" else q.dt.to_period("Q").astype(str)


def plot_selection_evolution(
    sel: pd.DataFrame,
    *,
    top: int = 12,
    period: str = "year",
    gamma: float = 0.5,
    axes=None,
    title: str = "sg-LASSO: which predictors carry the signal, and when",
):
    """Two-panel view of predictor importance over time.

    Parameters
    ----------
    sel:
        Tidy frame with columns ``quarter`` (timestamp), ``variable``, ``l2_norm`` and ``group``.
    top:
        Number of predictors (by total importance) to show in the heatmap.
    period:
        ``"year"`` or ``"quarter"`` for the time axis.
    gamma:
        Power-law colour scaling for the heatmap (``<1`` compresses a dominant predictor so the
        supporting cast stays visible; monthly GDP otherwise saturates the map).

    Returns the ``(ax_heat, ax_area)`` axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    s = sel.copy()
    s["period"] = _period(s, period)
    grp = s.drop_duplicates("variable").set_index("variable")["group"].to_dict()

    # importance matrix (variable x period); a variable absent from a period scored 0
    mat = s.pivot_table(index="variable", columns="period", values="l2_norm",
                        aggfunc="mean", fill_value=0.0)
    order = mat.sum(axis=1).sort_values(ascending=False)
    keep = order.head(top).index.tolist()
    heat = mat.loc[keep]
    gcol = _group_colors([grp.get(v, "") for v in mat.index])

    if axes is None:
        fig, (ax_h, ax_a) = plt.subplots(
            2, 1, figsize=(max(9, 0.5 * heat.shape[1] + 3), 0.42 * len(keep) + 4.2),
            gridspec_kw={"height_ratios": [len(keep), 6], "hspace": 0.35})
    else:
        ax_h, ax_a = axes
        fig = ax_h.figure

    # --- panel 1: heatmap of the top predictors ---------------------------------------
    cmap = LinearSegmentedColormap.from_list("imp", ["#f7fbff", PALETTE.get("myblue", "#29466E")])
    vmax = float(np.nanmax(heat.to_numpy())) or 1.0
    im = ax_h.imshow(heat.to_numpy(), aspect="auto", cmap=cmap,
                     norm=PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax))
    ax_h.set_xticks(range(heat.shape[1]))
    ax_h.set_xticklabels(heat.columns, fontsize=8)
    ax_h.set_yticks(range(len(keep)))
    ax_h.set_yticklabels(keep, fontsize=9)
    for tick, v in zip(ax_h.get_yticklabels(), keep):
        tick.set_color(gcol.get(grp.get(v, ""), "black"))
    ax_h.set_title(title, fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax_h, fraction=0.02, pad=0.01)
    cb.set_label("importance (L2)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # --- panel 2: importance share by variable group over time ------------------------
    gmat = (s.groupby(["period", "group"])["l2_norm"].sum().unstack("group").fillna(0.0))
    gmat = gmat.div(gmat.sum(axis=1).replace(0, np.nan), axis=0)   # share within each period
    gmat = gmat[gmat.sum().sort_values(ascending=False).index]     # biggest groups first
    ax_a.stackplot(range(len(gmat.index)), *[gmat[c].values for c in gmat.columns],
                   labels=list(gmat.columns),
                   colors=[gcol.get(c, "#999999") for c in gmat.columns], alpha=0.9)
    ax_a.set_xlim(0, len(gmat.index) - 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xticks(range(len(gmat.index)))
    ax_a.set_xticklabels(gmat.index, fontsize=8)
    ax_a.set_ylabel("importance share", fontsize=9)
    ax_a.set_title("composition by variable group", fontsize=10, loc="left")
    ax_a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=min(5, gmat.shape[1]), frameon=False)
    fig.tight_layout()
    return ax_h, ax_a


def plot_rank_evolution(curve, *, ax=None, title="Model rank through the release cycle"):
    """Each model's RMSE **rank** (1 = best) as information accumulates.

    ``curve`` is the ``days_to_publication x model`` RMSE frame from
    :func:`MIDAS.release_cycle_rmse`. This is the paper's model-ranking-evolution figure:
    it shows the ordering churn (complex models on top mid-cycle, monthly-GDP bridges on top
    late) far more legibly than the raw RMSE lines.
    """
    import matplotlib.pyplot as plt
    from .nowcast_plots import _STYLE

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    ranks = curve.rank(axis=1, method="min")
    for m in curve.columns:
        st = _STYLE.get(m, {})
        ax.plot(ranks.index, ranks[m], color=st.get("color", "#888888"),
                lw=st.get("lw", 2.0), label=m)
    ax.invert_yaxis()                       # rank 1 (best) at the top
    ax.set_yticks(range(1, len(curve.columns) + 1))
    ax.set_xlabel("days to publication")
    ax.set_ylabel("RMSE rank (1 = best)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="lower left", frameon=False)
    fig.tight_layout()
    return ax


def plot_winner_heatmap(rc, models, *, exclude_years=(), bucket=15, ax=None,
                        title="Winning model by quarter and forecast origin"):
    """Categorical heatmap of the best model per (quarter, lead-bucket).

    ``rc`` is the release-cycle backtest frame; a cell's winner is the model with the smallest
    mean absolute error in that quarter and ``bucket``-day window. Shows how the winner shifts
    from complex models (early) to monthly-GDP bridges (late), quarter by quarter.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from .nowcast_plots import _STYLE

    d = rc[(~rc.ref_quarter.dt.year.isin(list(exclude_years))) & (rc.model.isin(models))].copy()
    d["lead_bucket"] = (d.days_to_publication // bucket) * bucket
    d["abserr"] = (d.y_hat - d.y_true).abs()
    g = d.groupby(["ref_quarter", "lead_bucket", "model"], as_index=False).abserr.mean()
    win = g.loc[g.groupby(["ref_quarter", "lead_bucket"]).abserr.idxmin()]
    piv = win.pivot(index="ref_quarter", columns="lead_bucket", values="model")

    codes = {m: i for i, m in enumerate(models)}
    colors = [_STYLE.get(m, {}).get("color", "#999999") for m in models]
    mat = piv.apply(lambda col: col.map(codes)).to_numpy(dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    ax.imshow(mat, aspect="auto", cmap=ListedColormap(colors), vmin=0, vmax=len(models) - 1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([int(c) for c in piv.columns], fontsize=7)
    yr = piv.index.year
    ticks = [i for i in range(len(yr)) if i == 0 or yr[i] != yr[i - 1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([piv.index[i].year for i in ticks], fontsize=7)
    ax.set_xlabel("days to publication (bucket)")
    ax.set_ylabel("quarter")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(handles=[Patch(color=colors[i], label=m) for i, m in enumerate(models)],
              fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)
    fig.tight_layout()
    return ax


def plot_relative_rmse(curve, *, baseline="Q-AR", ax=None, title=None):
    """Each model's RMSE as a percentage of ``baseline``'s, through the release cycle.

    ``curve`` is the ``days_to_publication x model`` RMSE frame from
    :func:`MIDAS.release_cycle_rmse``. Values below 100 beat the baseline; the figure makes the
    *size* of each model's edge (and where in the cycle it appears) directly readable.
    """
    import matplotlib.pyplot as plt
    from .nowcast_plots import _STYLE

    rel = curve.div(curve[baseline], axis=0) * 100.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    for m in curve.columns:
        if m == baseline:
            continue
        st = _STYLE.get(m, {})
        ax.plot(rel.index, rel[m], color=st.get("color", "#888888"), lw=st.get("lw", 2.0), label=m)
    ax.axhline(100, color="0.3", ls="--", lw=1.2)
    ax.set_xlabel("days to publication")
    ax.set_ylabel(f"RMSE relative to {baseline} (%)")
    ax.set_title(title or f"Relative RMSE ({baseline} = 100)", loc="left", fontsize=11)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    return ax
