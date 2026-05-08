"""Loader for the EconAI adjusted EPU panel (Brochet, Mueller & Rauh 2025).

Source: https://data.econai.org/EPU/EPU_data.csv
Schema: isocode, country, period (YYYYMM), epu_adjusted, epu_unadjusted, nb_articles,
        armedconf, e, p, u

`epu_adjusted` is the LLM-adjusted EPU series. `e`, `p`, `u` are the
economic / policy / uncertainty component shares.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

EPU_URL = "https://data.econai.org/EPU/EPU_data.csv"


def download_epu(cache_dir: Path | str, force: bool = False) -> Path:
    """Download (or reuse cached) EPU CSV. Returns the local path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "EPU_data.csv"
    if path.exists() and not force:
        return path
    r = requests.get(EPU_URL, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def load_epu_panel(cache_dir: Path | str, countries: list[str] | None = None) -> pd.DataFrame:
    """Load EPU as a tidy long DataFrame with monthly DatetimeIndex.

    Returns columns: country, isocode, epu_adjusted, epu_unadjusted, nb_articles,
                     e, p, u, armedconf
    Indexed by `date` (first-of-month).
    """
    path = download_epu(cache_dir)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["period"].astype(str), format="%Y%m")
    df = df.drop(columns=["period"])
    if countries is not None:
        df = df[df["isocode"].isin(countries)]
    return df.set_index("date").sort_index()


def epu_wide(df: pd.DataFrame, value: str = "epu_adjusted") -> pd.DataFrame:
    """Pivot the long EPU panel to wide (date x isocode) for plotting."""
    return df.reset_index().pivot_table(index="date", columns="isocode", values=value)


# Backward-compatible alias used by the session scripts.
load_epu = load_epu_panel
