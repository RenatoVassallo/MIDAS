"""Matplotlib style matching the Beamer deck (myblue/myred/mygreen palette)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "myblue":   "#29466E",
    "myred":    "#802828",
    "mygreen":  "#2D6E50",
    "linknavy": "#28468C",
    "pastgray": "#999999",
}

CYCLE = [PALETTE["myblue"], PALETTE["myred"], PALETTE["mygreen"], PALETTE["pastgray"]]


def use_aer_style() -> None:
    """Apply minimalist white-bg style consistent with Sessions 1-2."""
    mpl.rcParams.update({
        "figure.figsize":   (7.0, 4.0),
        "figure.dpi":       110,
        "savefig.dpi":      200,
        "savefig.bbox":     "tight",
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "axes.linewidth":   0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.color":       "#e6e6e6",
        "grid.linewidth":   0.5,
        "axes.titlesize":   11,
        "axes.titleweight": "regular",
        "axes.labelsize":   10,
        "axes.labelcolor":  "#333333",
        "xtick.color":      "#333333",
        "ytick.color":      "#333333",
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "legend.frameon":   False,
        "lines.linewidth":  1.4,
        "font.family":      "serif",
        "mathtext.fontset": "cm",
        "axes.prop_cycle":  mpl.cycler(color=CYCLE),
    })


def shade_periods(ax, periods, color: str = PALETTE["pastgray"], alpha: float = 0.18) -> None:
    """Shade recession / crisis windows on a time-series axis.

    `periods` is an iterable of (start, end) tuples (any pd.Timestamp-coercible).
    """
    import pandas as pd
    for start, end in periods:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=alpha, lw=0)


def save_for_slides(fig, path, *, format: str = "pdf") -> None:
    """Save with the slide-friendly defaults."""
    fig.savefig(path, format=format)
    plt.close(fig)
