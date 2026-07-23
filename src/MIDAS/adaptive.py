"""Information-conditional forecast combination: the paper's contribution.

The release-cycle horse race shows there is no universally best model (the DFM wins mid-cycle,
monthly-GDP bridges win late, sparse ML picks up the consistently useful hard indicators). This
module turns that into a policy: a **dynamic forecast combination whose weights depend on an
Information Index** measuring how complete the real-time dataset is, rather than on calendar time.

Two pieces:

* :func:`information_index` / :func:`add_information_index` - a scalar in ``[0, 1]`` giving the
  share of the recent, target-relevant panel that has been released at an origin. It rises as
  surveys (timely), then hard indicators, then monthly GDP clear their publication lags.
* :func:`combine_release_cycle` - inverse-MSE combination of member models estimated **within
  Information-Index bins, on past quarters only** (real-time). Weights therefore shift from the
  complex/global models when information is scarce toward the parsimonious monthly-GDP bridge as
  the hard data arrive, exactly the ranking churn the horse race documents.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def add_information_index(
    rc: pd.DataFrame,
    panel,
    *,
    window_months: int = 6,
) -> pd.DataFrame:
    """Add an ``info_index`` column to a release-cycle backtest frame.

    For each target quarter ``Q``, the *frontier* is every monthly cell dated in
    ``[Q - window_months, Q]`` that eventually exists; the index at an origin is the fraction of
    those cells already released (``release_date <= origin``), using the panel's publication
    delays. It is computed per quarter by a vectorised ``searchsorted`` over the cells' release
    dates, so it is cheap even on a daily grid.
    """
    delays = panel.delays()
    out = []
    for Q, g in rc.groupby("ref_quarter", sort=False):
        Q = pd.Timestamp(Q)
        lo = Q - pd.DateOffset(months=window_months)
        sub = panel.monthly.loc[(panel.monthly.index >= lo) & (panel.monthly.index <= Q)]
        if sub.empty:
            g = g.copy(); g["info_index"] = 0.0; out.append(g); continue
        month_end = (sub.index + pd.offsets.MonthEnd(1)).to_numpy()          # (n_months,)
        dvec = np.array([int(delays[c]) for c in sub.columns])               # (n_cols,)
        rel = month_end[:, None] + dvec[None, :] * np.timedelta64(1, "D")    # (n_months, n_cols)
        reldates = np.sort(rel[sub.notna().to_numpy()])                      # existing cells only
        g = g.copy()
        if len(reldates) == 0:
            g["info_index"] = 0.0
        else:
            origins = g.origin_date.to_numpy().astype("datetime64[ns]")
            g["info_index"] = np.searchsorted(reldates, origins, side="right") / len(reldates)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def combine_release_cycle(
    rc: pd.DataFrame,
    members: Sequence[str],
    *,
    index_col: str = "info_index",
    n_bins: int = 4,
    min_train: int = 6,
    method: str = "inv_mse",
    power: float = 1.0,
    name: str = "Adaptive-IC",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Information-conditional inverse-MSE combination of ``members`` over the release cycle.

    The Information Index is binned into ``n_bins`` equal-width bins on ``[0, 1]``. For each target
    quarter (in time order) and bin, member weights are inverse mean-squared-error estimated on the
    origins of **earlier** quarters that fall in the same bin (real-time; equal weights until
    ``min_train`` past quarters are available). The combined nowcast for every origin in that
    (quarter, bin) is the weighted average of the members' nowcasts.

    Returns ``(combined_rows, weights)`` where ``combined_rows`` has the release-cycle schema
    (``ref_quarter, origin_date, days_to_publication, model, y_true, y_hat``) and ``weights``
    records the weight on each member by quarter and bin.
    """
    members = list(members)
    d = rc[rc.model.isin(members)].copy()
    keys = ["ref_quarter", "origin_date", "days_to_publication", index_col]
    wide = d.pivot_table(index=keys, columns="model", values="y_hat", aggfunc="first")
    truth = d.pivot_table(index=keys, values="y_true", aggfunc="first")["y_true"]
    wide = wide.reindex(columns=members).join(truth).reset_index()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    wide["bin"] = np.clip(np.digitize(wide[index_col].to_numpy(), edges[1:-1]), 0, n_bins - 1)

    rows, wrows = [], []
    quarters = sorted(wide.ref_quarter.unique())
    for Q in quarters:
        for b in range(n_bins):
            cur = wide[(wide.ref_quarter == Q) & (wide.bin == b)]
            if cur.empty:
                continue
            past = wide[(wide.ref_quarter < Q) & (wide.bin == b)].dropna(subset=members + ["y_true"])
            if method != "equal" and len(past["ref_quarter"].unique()) >= min_train:
                mse = ((past[members].sub(past["y_true"], axis=0)) ** 2).mean()
                if np.isfinite(mse).all() and (mse > 0).all():
                    if method == "best":            # hard selection of the in-bin best member
                        w = np.zeros(len(members)); w[int(np.argmin(mse.to_numpy()))] = 1.0
                    else:                            # inverse-MSE, sharpened by ``power``
                        w = (1.0 / mse.to_numpy()) ** power; w = w / w.sum()
                else:
                    w = np.repeat(1.0 / len(members), len(members))
            else:
                w = np.repeat(1.0 / len(members), len(members))
            M = cur[members].to_numpy(dtype=float)
            Wm = np.where(np.isfinite(M), w[None, :], 0.0)
            denom = Wm.sum(axis=1)
            yhat = np.where(denom > 0, np.nansum(np.where(np.isfinite(M), M * w[None, :], 0.0), axis=1) / denom, np.nan)
            block = cur[["ref_quarter", "origin_date", "days_to_publication", "y_true"]].copy()
            block["y_hat"] = yhat
            block["model"] = name
            rows.append(block)
            wrows.append({"ref_quarter": Q, "bin": b, "info_lo": edges[b], "info_hi": edges[b + 1],
                          **dict(zip(members, np.round(w, 3)))})
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return combined, pd.DataFrame(wrows)


def plot_weight_evolution(rc, weights, members, *, index_col="info_index", n_bins=4, ax=None,
                          title="Combination weights through the release cycle"):
    """Continuous stacked-area view of the combination's applied weights over the cycle.

    Each origin is mapped to its Information-Index bin, given that quarter's learned weights, and the
    weights are averaged across quarters at each ``days_to_publication``. Cross-quarter averaging
    (quarters reach a given lead at slightly different Index values) smooths the discrete bins into a
    continuous transition from global-factor to hard-bridge.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from .nowcast_plots import _STYLE

    members = list(members)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    d = rc.drop_duplicates(["ref_quarter", "days_to_publication"]).copy()
    d["bin"] = np.clip(np.digitize(d[index_col].to_numpy(), edges[1:-1]), 0, n_bins - 1)
    w = d.merge(weights[["ref_quarter", "bin"] + members], on=["ref_quarter", "bin"], how="left")
    wl = w.groupby("days_to_publication")[members].mean().sort_index()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.figure
    ax.stackplot(wl.index, *[wl[m].to_numpy() for m in members], labels=members,
                 colors=[_STYLE.get(m, {}).get("color", "#999999") for m in members], alpha=0.85)
    ax.set_xlim(float(wl.index.min()), float(wl.index.max()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("days to publication")
    ax.set_ylabel("combination weight")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(fontsize=8, ncol=len(members), loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.tight_layout()
    return ax
