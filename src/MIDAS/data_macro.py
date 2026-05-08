"""Macro data loaders for the teaching panel.

The default panel is intentionally compact and stable: seven advanced
economies with good coverage for GDP, industrial production, prices,
unemployment, and short rates.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"

# Default teaching panel used throughout notebooks 01-06.
COUNTRIES: list[str] = ["CAN", "DEU", "FRA", "GBR", "ITA", "JPN", "USA"]

# (indicator_label, frequency) -> dict[iso3, fred_code]
SERIES: dict[tuple[str, str], dict[str, str]] = {
    ("gdp_real", "Q"): {
        "USA": "GDPC1",
        "GBR": "CLVMNACSCAB1GQUK",
        "DEU": "CLVMNACSCAB1GQDE",
        "FRA": "CLVMNACSCAB1GQFR",
        "ITA": "CLVMNACSCAB1GQIT",
        "CAN": "NGDPRSAXDCCAQ",
        "JPN": "JPNRGDPEXP",
    },
    ("ip", "M"): {
        "USA": "INDPRO",
        "GBR": "GBRPROINDMISMEI",
        "DEU": "DEUPROINDMISMEI",
        "FRA": "FRAPROINDMISMEI",
        "ITA": "ITAPROINDMISMEI",
        "CAN": "CANPROINDMISMEI",
        "JPN": "JPNPROINDMISMEI",
    },
    ("cpi", "M"): {
        "USA": "CPIAUCSL",
        "GBR": "GBRCPIALLMINMEI",
        "DEU": "DEUCPIALLMINMEI",
        "FRA": "FRACPIALLMINMEI",
        "ITA": "ITACPIALLMINMEI",
        "CAN": "CANCPIALLMINMEI",
        "JPN": "JPNCPIALLMINMEI",
    },
    ("unemp", "M"): {
        "USA": "UNRATE",
        "GBR": "LRHUTTTTGBM156S",
        "DEU": "LRHUTTTTDEM156S",
        "FRA": "LRHUTTTTFRM156S",
        "ITA": "LRHUTTTTITM156S",
        "CAN": "LRUN64TTCAM156S",
        "JPN": "LRUN64TTJPM156S",
    },
    ("rate3m", "M"): {
        "USA": "TB3MS",
        "GBR": "IR3TIB01GBM156N",
        "DEU": "IR3TIB01DEM156N",
        "FRA": "IR3TIB01FRM156N",
        "ITA": "IR3TIB01ITM156N",
        "CAN": "IR3TIB01CAM156N",
        "JPN": "IR3TIB01JPM156N",
    },
}


def fetch_fred_series(code: str, cache_dir: Path | str, force: bool = False) -> pd.DataFrame:
    """Fetch one FRED series and cache it locally as CSV."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{code}.csv"
    if path.exists() and not force:
        text = path.read_text()
    else:
        response = requests.get(FRED_CSV.format(code=code), timeout=60)
        response.raise_for_status()
        text = response.text
        path.write_text(text)

    df = pd.read_csv(StringIO(text))
    df.columns = [col.lower() for col in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    value_col = [col for col in df.columns if col != date_col][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.rename(columns={date_col: "date", value_col: "value"})[["date", "value"]]


def load_macro_panel(
    cache_dir: Path | str,
    countries: list[str] | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Build the long macro panel with columns date/country/indicator/value/freq."""
    countries = countries or COUNTRIES
    rows: list[pd.DataFrame] = []
    for (indicator, freq), mapping in SERIES.items():
        for country, code in mapping.items():
            if country not in countries:
                continue
            try:
                df = fetch_fred_series(code, cache_dir, force=force)
            except Exception as exc:
                print(f"  ! fetch failed: {country}/{indicator} ({code}): {exc}")
                continue
            rows.append(df.assign(country=country, indicator=indicator, freq=freq))

    if not rows:
        return pd.DataFrame(columns=["date", "country", "indicator", "value", "freq"])
    return pd.concat(rows, ignore_index=True).sort_values(["country", "indicator", "date"]).reset_index(drop=True)


# Backward-compatible alias used by the session scripts.
load_panel = load_macro_panel
