"""Publication-style forecast-evaluation tables.

``horizon_rmse_table`` scores every model at each horizon on a **matched** sample (the
base quarters where all models produce a forecast), and attaches Diebold-Mariano stars
comparing a focus model against a named benchmark. ``horizon_table_latex`` renders
several targets side by side as one booktabs table.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation import dm_test, rmse


def stars(p: float) -> str:
    """Significance markers: 1%, 5%, 10%."""
    if not np.isfinite(p):
        return ""
    if p <= 0.01:
        return "***"
    if p <= 0.05:
        return "**"
    if p <= 0.10:
        return "*"
    return ""


def horizon_rmse_table(
    bt: pd.DataFrame,
    *,
    models: Sequence[str],
    horizons: Sequence[int] = (4, 1, 0),
    focus: Sequence[str] = ("DFM", "sg-LASSO"),
    benchmark: str = "AR(2)",
    exclude_years: Sequence[int] = (2020, 2021),
) -> pd.DataFrame:
    """RMSE by horizon and model, plus a DM p-value for each ``focus`` model.

    Models are compared on the base quarters where *all* of them produce a forecast, so
    differing coverage cannot flatter anyone. Each focus model is tested against
    ``benchmark``; the DM HAC bandwidth is the horizon (an optimal h-step forecast error
    is MA(h-1)). p-values land in ``dm_p_<model>`` columns.
    """
    focus = [focus] if isinstance(focus, str) else list(focus)
    rows = []
    for h in horizons:
        d = bt[(bt.horizon == h) & (~bt.ref_quarter.dt.year.isin(list(exclude_years)))]
        p = d.pivot_table(index="base_quarter", columns="model", values="y_hat", aggfunc="first")
        yt = d.pivot_table(index="base_quarter", values="y_true", aggfunc="first")["y_true"]
        avail = [m for m in models if m in p.columns]
        c = p[avail].dropna().index
        row = {"horizon": int(h), "n": len(c)}
        for m in models:
            row[m] = rmse(yt.loc[c], p.loc[c, m]) if m in avail and len(c) else np.nan
        for f in focus:
            pv = np.nan
            if f in avail and benchmark in avail and f != benchmark and len(c) > 8:
                _, pv = dm_test((p.loc[c, f] - yt.loc[c]).to_numpy(),
                                (p.loc[c, benchmark] - yt.loc[c]).to_numpy(),
                                h=max(1, int(h)))
            row[f"dm_p_{f}"] = pv
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")


def horizon_table_latex(
    tables: Mapping[str, pd.DataFrame],
    *,
    models: Sequence[str],
    focus: Sequence[str] = ("DFM", "sg-LASSO"),
    decimals: int = 2,
    label: str = "tab:horizon",
    caption: str = "Forecast evaluation: RMSE by horizon.",
    benchmark: str = "AR(2)",
    float_env: bool = True,
) -> str:
    """Render ``{panel name: horizon_rmse_table}`` as one booktabs LaTeX table.

    ``float_env=True`` wraps it in a ``table`` float with caption and notes (for a
    paper). ``False`` emits the bare ``tabular`` only, which is what a Beamer frame
    wants inside a ``\\resizebox``.
    """
    focus = [focus] if isinstance(focus, str) else list(focus)
    panels = list(tables)
    ncol = len(models)
    fmt = "l" + " ".join(["c" * ncol] * len(panels))
    head_panels = " & ".join(rf"\multicolumn{{{ncol}}}{{c}}{{{p}}}" for p in panels)
    head_models = " & ".join(" & ".join(_esc(m) for m in models) for _ in panels)

    lines = []
    if float_env:
        lines += [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
                  rf"\label{{{label}}}", r"\small"]
    lines += [rf"\begin{{tabular}}{{{fmt}}}", r"\toprule", rf"Horizon & {head_panels} \\"]
    starts = [2 + i * ncol for i in range(len(panels))]
    lines.append(" ".join(rf"\cmidrule(lr){{{s}-{s + ncol - 1}}}" for s in starts))
    lines.append(rf" & {head_models} \\")
    lines.append(r"\midrule")

    horizons = list(next(iter(tables.values())).index)
    for h in horizons:
        cells = []
        for p in panels:
            t = tables[p]
            for m in models:
                v = t.loc[h, m]
                s = "" if not np.isfinite(v) else f"{v:.{decimals}f}"
                col = f"dm_p_{m}"
                if m in focus and col in t.columns and np.isfinite(t.loc[h, col]):
                    mark = stars(t.loc[h, col])
                    if mark:
                        s += rf"$^{{{mark}}}$"
                cells.append(s if s else r"n.a.")
        lines.append(rf"{h} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    if not float_env:
        return "\n".join(lines)
    lines += [
        r"\begin{minipage}{\textwidth}\vspace{0.4em}\scriptsize",
        rf"\textit{{Notes}}: Root mean squared forecast errors for year-on-year growth, by horizon "
        rf"$h$ (quarters). The origin is the first day of the quarter after the base quarter, so "
        rf"$h=0$ is a nowcast of the just-ended, not-yet-published quarter and $h\geq1$ are forecasts. "
        rf"All models are scored on the same base quarters (matched sample); 2020 and 2021 are excluded. "
        rf"Asterisks on {', '.join(_esc(f) for f in focus)} indicate that the null of equal "
        rf"predictive accuracy against {_esc(benchmark)} is rejected by the Diebold-Mariano test: "
        rf"$^{{*}}$ 10\%, $^{{**}}$ 5\%, $^{{***}}$ 1\%. The test is two-sided, so the direction is "
        rf"read from the RMSE itself.",
        r"\end{minipage}", r"\end{table}",
    ]
    return "\n".join(lines)


def _esc(s: str) -> str:
    return s.replace("_", r"\_")
