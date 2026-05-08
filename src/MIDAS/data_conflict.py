"""ConflictForecast topic loaders for teaching experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TOPIC_COLUMNS = [f"stock_topic_{i}" for i in range(15)]


def load_conflict_topics(
    path_or_dir: Path | str,
    *,
    countries: list[str] | None = None,
    topic_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the monthly ConflictForecast topic stocks.

    Parameters
    ----------
    path_or_dir:
        Either the CSV file itself or the directory containing
        ``df_merge_last.csv``.
    countries:
        Optional list of ISO3 country codes to retain.
    topic_columns:
        Optional subset of topic columns. Defaults to the 15 stock topics.
    """
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "df_merge_last.csv"

    topic_columns = list(topic_columns or TOPIC_COLUMNS)
    usecols = ["isocode", "period"] + topic_columns
    df = pd.read_csv(path, usecols=usecols)
    if countries is not None:
        df = df[df["isocode"].isin(countries)].copy()

    df["date"] = pd.to_datetime(df["period"].astype(str) + "01", format="%Y%m%d")
    df = df.rename(columns={"isocode": "country"})
    keep = ["country", "date"] + topic_columns
    return df.loc[:, keep].sort_values(["country", "date"]).reset_index(drop=True)


def aggregate_conflict_topics(
    monthly: pd.DataFrame,
    *,
    freq: str = "Q",
    how: str = "mean",
) -> pd.DataFrame:
    """Aggregate monthly topic stocks to quarter-level features."""
    topic_columns = [col for col in monthly.columns if col.startswith("stock_topic_")]
    if not topic_columns:
        raise ValueError("No stock_topic_* columns found in monthly ConflictForecast data.")

    out = monthly.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period(freq).dt.to_timestamp()
    grouped = out.groupby(["country", "date"], as_index=False)[topic_columns]
    if how == "mean":
        return grouped.mean()
    if how == "last":
        return grouped.last()
    raise ValueError("how must be 'mean' or 'last'")
