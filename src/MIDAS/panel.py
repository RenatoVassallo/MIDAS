"""Panel-building utilities for the teaching repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_epu import load_epu_panel
from .data_macro import COUNTRIES, load_macro_panel
from .recession import load_recession_panel


@dataclass
class PanelArtifacts:
    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    coverage: pd.DataFrame


@dataclass
class PanelBuilder:
    """Build the monthly and quarterly teaching panels from raw cached data."""

    raw_dir: Path | str
    processed_dir: Path | str
    countries: list[str] | None = None

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.countries = list(self.countries or COUNTRIES)

    @staticmethod
    def _pivot_macro(panel: pd.DataFrame) -> pd.DataFrame:
        return (
            panel.pivot_table(index=["country", "date"], columns="indicator", values="value")
            .reset_index()
        )

    @staticmethod
    def _monthly_yoy(series: pd.Series) -> pd.Series:
        return 100.0 * (series / series.shift(12) - 1.0)

    @staticmethod
    def _quarterly_yoy(series: pd.Series) -> pd.Series:
        return 100.0 * (series / series.shift(4) - 1.0)

    @staticmethod
    def _qoq_annualized(series: pd.Series) -> pd.Series:
        return 100.0 * ((series / series.shift(1)) ** 4 - 1.0)

    def _build_monthly_panel(self, macro_wide: pd.DataFrame, epu: pd.DataFrame, recession: pd.DataFrame) -> pd.DataFrame:
        monthly_indicators = ["ip", "cpi", "unemp", "rate3m"]
        monthly = (
            macro_wide.dropna(subset=monthly_indicators, how="all")
            .loc[:, ["country", "date"] + [col for col in monthly_indicators if col in macro_wide.columns]]
            .copy()
        )
        monthly["date"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
        monthly = monthly.sort_values(["country", "date"]).drop_duplicates(["country", "date"])

        if "cpi" in monthly.columns:
            monthly["cpi_yoy"] = monthly.groupby("country")["cpi"].transform(self._monthly_yoy)
        if "ip" in monthly.columns:
            monthly["ip_yoy"] = monthly.groupby("country")["ip"].transform(self._monthly_yoy)

        epu_keep = epu[["date", "isocode", "epu_adjusted", "epu_unadjusted", "nb_articles"]].rename(
            columns={"isocode": "country"}
        )
        monthly = monthly.merge(epu_keep, on=["date", "country"], how="left")

        recession_monthly = recession.rename(columns={"recession": "rec_m"})
        monthly = monthly.merge(recession_monthly, on=["date", "country"], how="left")
        return monthly

    def _build_quarterly_panel(self, macro_wide: pd.DataFrame, monthly: pd.DataFrame, recession: pd.DataFrame) -> pd.DataFrame:
        if "gdp_real" not in macro_wide.columns:
            return pd.DataFrame(columns=["country", "date", "gdp_real", "gdp_yoy", "gdp_qoq_ann"])

        qframe = macro_wide.loc[macro_wide["gdp_real"].notna(), ["country", "date", "gdp_real"]].copy()
        qframe["date"] = qframe["date"].dt.to_period("Q").dt.to_timestamp()
        qframe = qframe.sort_values(["country", "date"]).drop_duplicates(["country", "date"])
        qframe["gdp_yoy"] = qframe.groupby("country")["gdp_real"].transform(self._quarterly_yoy)
        qframe["gdp_qoq_ann"] = qframe.groupby("country")["gdp_real"].transform(self._qoq_annualized)

        agg_cols = [
            col for col in ["ip", "ip_yoy", "cpi_yoy", "unemp", "rate3m", "epu_adjusted", "nb_articles"]
            if col in monthly.columns
        ]
        monthly_for_agg = monthly.copy()
        monthly_for_agg["quarter"] = monthly_for_agg["date"].dt.to_period("Q").dt.to_timestamp()
        qagg = (
            monthly_for_agg.groupby(["country", "quarter"], as_index=False)[agg_cols]
            .mean()
            .rename(columns={"quarter": "date"})
        )
        quarterly = qframe.merge(qagg, on=["country", "date"], how="left")

        recession_quarterly = (
            recession.assign(date=recession["date"].dt.to_period("Q").dt.to_timestamp())
            .groupby(["country", "date"], as_index=False)["recession"]
            .max()
            .rename(columns={"recession": "rec_q"})
        )
        return quarterly.merge(recession_quarterly, on=["country", "date"], how="left")

    def _build_coverage_table(self, macro: pd.DataFrame, epu: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for (country, indicator), group in macro.groupby(["country", "indicator"]):
            observed = group.dropna(subset=["value"])
            if observed.empty:
                continue
            rows.append(
                {
                    "country": country,
                    "indicator": indicator,
                    "n": len(observed),
                    "first": observed["date"].min().date(),
                    "last": observed["date"].max().date(),
                }
            )
        for country, group in epu.groupby("isocode"):
            observed = group.dropna(subset=["epu_adjusted"])
            if observed.empty:
                continue
            rows.append(
                {
                    "country": country,
                    "indicator": "epu_adjusted",
                    "n": len(observed),
                    "first": observed["date"].min().date(),
                    "last": observed["date"].max().date(),
                }
            )
        return pd.DataFrame(rows).sort_values(["country", "indicator"]).reset_index(drop=True)

    def build(self) -> PanelArtifacts:
        macro = load_macro_panel(self.raw_dir / "fred", countries=self.countries)
        epu = load_epu_panel(self.raw_dir / "epu", countries=self.countries).reset_index()
        recession = load_recession_panel(self.countries, self.raw_dir / "recession")

        macro_wide = self._pivot_macro(macro)
        macro_wide["date"] = pd.to_datetime(macro_wide["date"])

        monthly = self._build_monthly_panel(macro_wide, epu, recession)
        quarterly = self._build_quarterly_panel(macro_wide, monthly, recession)
        coverage = self._build_coverage_table(macro, epu)
        return PanelArtifacts(monthly=monthly, quarterly=quarterly, coverage=coverage)

    def build_and_save(self) -> PanelArtifacts:
        artifacts = self.build()
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        artifacts.monthly.to_parquet(self.processed_dir / "panel_monthly.parquet", index=False)
        artifacts.quarterly.to_parquet(self.processed_dir / "panel_quarterly.parquet", index=False)
        artifacts.coverage.to_csv(self.processed_dir / "coverage.csv", index=False)
        return artifacts
