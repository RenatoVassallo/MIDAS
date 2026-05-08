"""Recession indicators per country, sourced from FRED OECD-based series.

Series codes: <CCC>RECDM (peak-to-trough monthly recession dummy, OECD-based),
e.g. USRECDM, GBRRECDM, DEURECDM. We expose a thin wrapper around
fredapi for Renato's class — students can also use a free-tier API key.

If no API key is available the module falls back to scraping the public
CSV download for each series.
"""

from __future__ import annotations
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"

ISO3_TO_FRED = {
    "USA": "USREC",         # NBER monthly
    "GBR": "GBRRECDM",
    "DEU": "DEURECDM",
    "FRA": "FRARECDM",
    "ITA": "ITARECDM",
    "ESP": "ESPRECDM",
    "CAN": "CANRECDM",
    "JPN": "JPNRECDM",
    "AUS": "AUSRECDM",
    "MEX": "MEXRECDM",
    "BRA": "BRARECDM",
    "KOR": "KORRECDM",
}


def fetch_recession_csv(code: str, cache_dir: Path | str, force: bool = False) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{code}.csv"
    if path.exists() and not force:
        text = path.read_text()
    else:
        r = requests.get(FRED_CSV.format(code=code), timeout=60)
        r.raise_for_status()
        text = r.text
        path.write_text(text)
    df = pd.read_csv(StringIO(text))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["observation_date"] if "observation_date" in df.columns else df.iloc[:, 0])
    val_col = [c for c in df.columns if c not in ("date", "observation_date")][0]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    return df[["date", val_col]].rename(columns={val_col: "recession"}).set_index("date")


def load_recession_panel(
    iso3_list: list[str],
    cache_dir: Path | str,
    force: bool = False,
) -> pd.DataFrame:
    """Return a long DataFrame: date, country (iso3), recession (0/1)."""
    frames = []
    for iso in iso3_list:
        code = ISO3_TO_FRED.get(iso)
        if code is None:
            continue
        try:
            df = fetch_recession_csv(code, cache_dir, force=force).reset_index()
            df["country"] = iso
            frames.append(df)
        except Exception as exc:
            print(f"  ! recession fetch failed for {iso} ({code}): {exc}")
    if not frames:
        return pd.DataFrame(columns=["date", "country", "recession"])
    return pd.concat(frames, ignore_index=True).sort_values(["country", "date"])


def build_recession_target(
    rec_panel: pd.DataFrame,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """For each (country, date), set y=1 if any recession month in (t, t+horizon]."""
    out = rec_panel.sort_values(["country", "date"]).copy()
    out["recession_ahead"] = (
        out.groupby("country")["recession"]
        .transform(lambda s: s.shift(-1).rolling(horizon_months).max())
    )
    return out
