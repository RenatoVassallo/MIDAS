"""Client for public macroeconomic data from China's National Bureau of Statistics.

Endpoint choice
===============

This module uses only official NBS web properties, but not a documented public
developer API. The request schemas below are reverse-engineered from the live
National Data portal and the official release archive.

Verified structured backends
----------------------------
1. Long-history structured series:

   ``POST https://data.stats.gov.cn/dg/website/publicrelease/web/external/stream/esData``

   This is the strongest endpoint found for deep historical pulls. In this
   environment it supports long monthly histories for a curated set of verified
   indicator IDs such as industrial value added and headline CPI.

2. Recent-window headline series:

   ``POST https://data.stats.gov.cn/dg/website/publicrelease/web/external/queryMacroecData``

   This is simpler and stable for a small recent rolling window, but it does
   not reliably expose deep history.

Official release-page backends
------------------------------
3. Quarterly real GDP growth:

   ``https://www.stats.gov.cn/sj/zxfb/``

   The client scans the official release archive for GDP release pages and
   parses the published year-on-year and seasonally adjusted quarter-on-quarter
   tables. Values are not hard-coded. Only the identifiers and parsing rules are
   curated.

4. Official manufacturing PMI:

   The client also uses the official release archive for PMI because a stable
   structured portal request for PMI could not be verified consistently from
   this environment.

Expected fragility
------------------
- ``esData`` and ``queryMacroecData`` are official portal endpoints, but not
  documented public APIs.
- The indicator registry for ``esData`` uses curated official IDs because the
  discovery endpoints were intermittently behind an anti-bot layer from this
  environment.
- GDP and PMI archive parsing is more fragile than the JSON endpoints because it
  depends on article titles and table structure.

Fallback options if a backend changes
-------------------------------------
1. Re-inspect the live National Data front-end bundle for replacement
   ``/publicrelease/web/external/...`` endpoints and updated indicator IDs.
2. Re-test the legacy ``easyquery`` portal from the user's own network.
3. Fall back to official release pages for headline indicators when only a small
   recent window is needed.

Usage
=====

Quarterly real GDP growth:

>>> gdp = get_nbs_data({"gdp_yoy": "gdp_real_yoy"}, "quarterly", start_period="2010Q1")

Long-history monthly core:

>>> monthly = get_nbs_data(
...     {
...         "ip_yoy": "industrial_value_added_yoy",
...         "cpi_index_yoy": "cpi_yoy",
...         "ppi_index_yoy": "ppi_yoy",
...     },
...     "monthly",
...     start_period="1990M01",
... )
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Literal
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

PORTAL_HOME_URL = "https://data.stats.gov.cn/dg/website/page.html#/pc/national/home"
PORTAL_MACRO_URL = (
    "https://data.stats.gov.cn/dg/website/publicrelease/web/external/queryMacroecData"
)
PORTAL_STREAM_URL = (
    "https://data.stats.gov.cn/dg/website/publicrelease/web/external/stream/esData"
)
NBS_RELEASE_ARCHIVE_URL = "https://www.stats.gov.cn/sj/zxfb/"
DEFAULT_TIMEOUT = 30
DEFAULT_RATE_LIMIT_SECONDS = 0.35
DEFAULT_MAX_RELEASE_ARCHIVE_PAGES = 220

Frequency = Literal["monthly", "quarterly", "annual"]
Backend = Literal["portal_macro", "portal_stream", "pmi_release", "gdp_release"]

_FREQUENCY_ALIASES: dict[str, Frequency] = {
    "m": "monthly",
    "month": "monthly",
    "monthly": "monthly",
    "q": "quarterly",
    "quarter": "quarterly",
    "quarterly": "quarterly",
    "a": "annual",
    "annual": "annual",
    "y": "annual",
    "year": "annual",
    "yearly": "annual",
}

_PORTAL_FREQUENCY_CODES = {"monthly": "MM", "quarterly": "SS", "annual": "YY"}

PMI_TITLE_RE = re.compile(
    r"(?P<year>\d{4})年(?P<month>\d{1,2})月中国采购经理指数运行情况"
)
GDP_RELEASE_TITLE_RE = re.compile(
    r"(?P<year>\d{4})年(?P<label>一季度|二季度和上半年|上半年|三季度|前三季度|四季度和全年|全年)"
    r"国内生产总值初步核算结果"
)


class NBSChinaError(RuntimeError):
    """Base exception for the NBS China client."""


class NBSChinaIndicatorError(NBSChinaError):
    """Raised when a requested indicator cannot be resolved safely."""


class NBSChinaRequestError(NBSChinaError):
    """Raised on request failures or anti-bot responses."""


class NBSChinaSchemaError(NBSChinaError):
    """Raised when a response does not match the expected schema."""


@dataclass(frozen=True)
class NBSIndicatorSpec:
    """Metadata and retrieval instructions for one supported indicator."""

    code: str
    clean_name: str
    label: str
    frequency: Frequency
    backend: Backend
    source_url: str
    unit: str | None = None
    portal_ek: str | None = None
    portal_du: str | None = None
    portal_dp: str | None = None
    portal_type: str | None = None
    stream_cid: str | None = None
    stream_root_id: str | None = None
    stream_indicator_id: str | None = None
    stream_area_code: str | None = "000000000000"
    stream_area_name: str | None = "全国"
    history_start: str | None = None
    release_series: Literal["gdp_real_yoy", "gdp_real_qoq_sa"] | None = None
    notes: str | None = None

    def request_payload(self) -> dict[str, str]:
        if self.backend != "portal_macro":
            raise NBSChinaIndicatorError(
                f"Indicator '{self.code}' does not use the portal macro backend."
            )
        if not all([self.portal_ek, self.portal_du, self.portal_dp, self.portal_type]):
            raise NBSChinaIndicatorError(
                f"Indicator '{self.code}' has incomplete portal request metadata."
            )
        return {
            "ek": self.portal_ek,
            "du": self.portal_du,
            "dp": self.portal_dp,
            "isMap": "0",
            "period": "",
            "type": self.portal_type,
        }


NBS_INDICATORS: dict[str, NBSIndicatorSpec] = {
    "gdp_current_price": NBSIndicatorSpec(
        code="gdp_current_price",
        clean_name="gdp_current_price",
        label="Quarterly GDP at current prices",
        frequency="quarterly",
        backend="portal_macro",
        source_url=PORTAL_HOME_URL,
        unit="亿元",
        portal_ek="f483cc485ca84aa0b75668871aca70af",
        portal_du="7dae305350e74cf793520af55bceffa4",
        portal_dp="21",
        portal_type="SS",
        notes="Recent official headline series from the National Data portal.",
    ),
    "gdp_real_yoy": NBSIndicatorSpec(
        code="gdp_real_yoy",
        clean_name="gdp_real_yoy",
        label="Real GDP growth, year-on-year",
        frequency="quarterly",
        backend="gdp_release",
        source_url=NBS_RELEASE_ARCHIVE_URL,
        unit="%",
        history_start="1992Q1",
        release_series="gdp_real_yoy",
        notes=(
            "Parsed from official GDP release tables in the NBS archive. This is the "
            "real GDP year-on-year growth table, not nominal GDP."
        ),
    ),
    "gdp_real_qoq_sa": NBSIndicatorSpec(
        code="gdp_real_qoq_sa",
        clean_name="gdp_real_qoq_sa",
        label="Real GDP growth, quarter-on-quarter, seasonally adjusted",
        frequency="quarterly",
        backend="gdp_release",
        source_url=NBS_RELEASE_ARCHIVE_URL,
        unit="%",
        history_start="1992Q1",
        release_series="gdp_real_qoq_sa",
        notes=(
            "Parsed from official GDP release tables in the NBS archive. This is the "
            "seasonally adjusted quarter-on-quarter growth table."
        ),
    ),
    "industrial_value_added_yoy": NBSIndicatorSpec(
        code="industrial_value_added_yoy",
        clean_name="industrial_value_added_yoy",
        label="Industrial value added growth, current month year-on-year",
        frequency="monthly",
        backend="portal_stream",
        source_url=PORTAL_STREAM_URL,
        unit="%",
        stream_cid="e2d9463aceae483eb122794e53180bf9",
        stream_root_id="0cff94832c7f4cbe9ca57b7c0ef09704",
        stream_indicator_id="6d249959166b4b07aad922e2aa51097d",
        history_start="1983M01",
        notes=(
            "Verified long-history structured pull from the official portal. This is "
            "the current-month year-on-year growth rate, not the cumulative growth rate."
        ),
    ),
    "industrial_value_added_cum_yoy": NBSIndicatorSpec(
        code="industrial_value_added_cum_yoy",
        clean_name="industrial_value_added_cum_yoy",
        label="Industrial value added growth, cumulative year-on-year",
        frequency="monthly",
        backend="portal_stream",
        source_url=PORTAL_STREAM_URL,
        unit="%",
        stream_cid="e2d9463aceae483eb122794e53180bf9",
        stream_root_id="0cff94832c7f4cbe9ca57b7c0ef09704",
        stream_indicator_id="de0f099f87f84f4a85ad5227e94f5f9d",
        history_start="1983M01",
        notes="Verified long-history structured pull from the official portal.",
    ),
    "retail_sales_nominal": NBSIndicatorSpec(
        code="retail_sales_nominal",
        clean_name="retail_sales_nominal",
        label="Retail sales of consumer goods, nominal level",
        frequency="monthly",
        backend="portal_macro",
        source_url=PORTAL_MACRO_URL,
        unit="亿元",
        portal_ek="0a31c6ae3efc489299149069ea71749a",
        portal_du="7dae305350e74cf793520af55bceffa4",
        portal_dp="21",
        portal_type="MM",
        notes="Recent-window headline series from the official portal.",
    ),
    "retail_sales_yoy": NBSIndicatorSpec(
        code="retail_sales_yoy",
        clean_name="retail_sales_yoy",
        label="Retail sales of consumer goods growth",
        frequency="monthly",
        backend="portal_macro",
        source_url=PORTAL_MACRO_URL,
        unit="%",
        portal_ek="0a31c6ae3efc489299149069ea71749a",
        portal_du="414774dee2bc47f392cf13abfa9de882",
        portal_dp="12",
        portal_type="MM",
        notes=(
            "Recent-window headline series from the official portal. Deep history for "
            "this series has not yet been pinned down through the structured backend."
        ),
    ),
    "fixed_asset_investment_yoy": NBSIndicatorSpec(
        code="fixed_asset_investment_yoy",
        clean_name="fixed_asset_investment_yoy",
        label="Fixed-asset investment growth",
        frequency="monthly",
        backend="portal_macro",
        source_url=PORTAL_MACRO_URL,
        unit="%",
        portal_ek="860475139a534e20b0aafb562335e114",
        portal_du="414774dee2bc47f392cf13abfa9de882",
        portal_dp="12",
        portal_type="MM",
        notes=(
            "Recent-window headline series from the official portal. Deep history for "
            "this series has not yet been pinned down through the structured backend."
        ),
    ),
    "cpi_yoy": NBSIndicatorSpec(
        code="cpi_yoy",
        clean_name="cpi_yoy",
        label="Headline CPI index, same month last year equals 100",
        frequency="monthly",
        backend="portal_stream",
        source_url=PORTAL_STREAM_URL,
        unit="上年同月=100",
        stream_cid="5353d942c68f42c789c7d8c546510ff4",
        stream_root_id="0cff94832c7f4cbe9ca57b7c0ef09704",
        stream_indicator_id="20f149caf55c4df5b41c1c5abf167e60",
        history_start="1983M01",
        notes=(
            "Verified long-history structured pull from the official portal. This is "
            "the index level with same month last year set to 100, not log inflation."
        ),
    ),
    "ppi_yoy": NBSIndicatorSpec(
        code="ppi_yoy",
        clean_name="ppi_yoy",
        label="Headline PPI index, same month last year equals 100",
        frequency="monthly",
        backend="portal_stream",
        source_url=PORTAL_STREAM_URL,
        unit="上年同月=100",
        stream_cid="aad939ce315847f79be5b73f90f4be4d",
        stream_root_id="67645373f52548baa69f9f24b2e6b5dc",
        stream_indicator_id="30f231ef06ff4c20b396812c367810e3",
        history_start="1990M01",
        notes=(
            "Verified structured pull from the official portal. This is the index with "
            "same month last year set to 100, not the percentage-point inflation rate."
        ),
    ),
    "grain_output": NBSIndicatorSpec(
        code="grain_output",
        clean_name="grain_output",
        label="Grain output",
        frequency="annual",
        backend="portal_macro",
        source_url=PORTAL_MACRO_URL,
        unit="万吨",
        portal_ek="6039804100003|4bb8aa1f3d53464fb687517573aa613c",
        portal_du="ad72a4425b6f4299bb9e34d3a6d5087e",
        portal_dp="1",
        portal_type="YY",
        notes="Recent official portal pull. The annual history is longer than the monthly headline window.",
    ),
    "manufacturing_pmi": NBSIndicatorSpec(
        code="manufacturing_pmi",
        clean_name="manufacturing_pmi",
        label="Official manufacturing PMI",
        frequency="monthly",
        backend="pmi_release",
        source_url=NBS_RELEASE_ARCHIVE_URL,
        unit="index",
        history_start="2005M01",
        notes=(
            "Parsed from official NBS release pages because a stable structured portal "
            "request for PMI could not be verified consistently from this environment."
        ),
    ),
}


def available_nbs_indicators(frequency: str | None = None) -> pd.DataFrame:
    """Return the built-in NBS indicator registry as a DataFrame."""
    target_frequency = None if frequency is None else _normalize_frequency(frequency)
    rows = []
    for spec in NBS_INDICATORS.values():
        if target_frequency is not None and spec.frequency != target_frequency:
            continue
        rows.append(asdict(spec))
    return pd.DataFrame(rows).sort_values(["frequency", "code"]).reset_index(drop=True)


def get_nbs_data(
    indicators: dict[str, str],
    frequency: str,
    start_period: str | None = None,
    end_period: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Retrieve public NBS macro data as a wide DataFrame keyed by ``period``.

    Parameters
    ----------
    indicators:
        Mapping of output column name -> registered indicator code. For example
        ``{"ip_yoy": "industrial_value_added_yoy"}``.
    frequency:
        ``"monthly"``, ``"quarterly"``, or ``"annual"``.
    start_period, end_period:
        Optional inclusive filters. Accepted formats:
        monthly ``YYYY-MM``, ``YYYYMM``, or ``YYYYMmm``;
        quarterly ``YYYYQn``;
        annual ``YYYY``.
    cache_dir:
        Directory for cached raw responses and logs. Defaults to
        ``Path.cwd() / ".cache" / "nbs_china"``.
    refresh:
        If ``True``, bypass the local cache and re-download responses.
    """
    target_frequency = _normalize_frequency(frequency)
    start = _parse_bound(start_period, target_frequency)
    end = _parse_bound(end_period, target_frequency)
    if start is not None and end is not None and start > end:
        raise ValueError("start_period must not be later than end_period.")

    client = _NBSChinaClient(cache_dir=cache_dir, refresh=refresh)
    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for output_name, code in indicators.items():
        spec = _resolve_indicator(code, target_frequency)
        series, meta = client.fetch_series(
            spec=spec,
            output_name=output_name,
            start=start,
            end=end,
        )
        frames.append(series)
        metadata_rows.append(meta)

    if not frames:
        return pd.DataFrame(columns=["period"])

    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="period", how="outer", validate="one_to_one")

    panel = panel.sort_values("period").reset_index(drop=True)
    panel.attrs["nbs_frequency"] = target_frequency
    panel.attrs["nbs_metadata"] = metadata_rows
    panel.attrs["nbs_raw_files"] = {
        row["output_name"]: row["raw_file"] for row in metadata_rows if row.get("raw_file")
    }
    return panel


class _NBSChinaClient:
    def __init__(
        self,
        *,
        cache_dir: str | Path | None,
        refresh: bool,
        timeout: int = DEFAULT_TIMEOUT,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
    ) -> None:
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.refresh = refresh
        self.timeout = timeout
        self.rate_limit_seconds = rate_limit_seconds
        self.raw_dir = self.cache_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.cache_dir / "download_log.jsonl"
        self._last_request_at = 0.0
        self._gdp_history_cache: pd.DataFrame | None = None
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset({"GET", "POST"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"
                ),
                "Accept": "application/json, text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
                "Referer": PORTAL_HOME_URL,
                "Origin": "https://data.stats.gov.cn",
            }
        )
        return session

    def fetch_series(
        self,
        *,
        spec: NBSIndicatorSpec,
        output_name: str,
        start: pd.Period | None,
        end: pd.Period | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if spec.backend == "portal_macro":
            frame, meta = self._fetch_portal_series(spec, output_name, start, end)
        elif spec.backend == "portal_stream":
            frame, meta = self._fetch_stream_series(spec, output_name, start, end)
        elif spec.backend == "pmi_release":
            frame, meta = self._fetch_pmi_series(spec, output_name, start, end)
        elif spec.backend == "gdp_release":
            frame, meta = self._fetch_gdp_release_series(spec, output_name, start, end)
        else:
            raise NBSChinaIndicatorError(f"Unsupported backend '{spec.backend}'.")

        self._append_log(meta)
        return frame, meta

    def _fetch_portal_series(
        self,
        spec: NBSIndicatorSpec,
        output_name: str,
        start: pd.Period | None,
        end: pd.Period | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        payload = spec.request_payload()
        body, raw_path, download_meta = self._request_cached(
            "POST",
            PORTAL_MACRO_URL,
            slug=spec.code,
            json_body=payload,
            expected="json",
        )
        obj = self._parse_json(body, url=PORTAL_MACRO_URL)
        rows = _validate_macro_payload(obj, spec.code)
        frame = _portal_rows_to_frame(rows, spec.frequency)
        frame = _filter_periods(frame, start, end).rename(columns={"value": output_name})
        warnings: list[str] = []
        if frame[output_name].isna().any():
            warnings.append("numeric coercion produced missing values")
        warnings.extend(_coverage_warnings(frame, output_name, start))

        latest_period = _latest_non_missing_period(frame, output_name)
        meta = {
            "output_name": output_name,
            "indicator_code": spec.code,
            "source": "NBS National Data portal",
            "source_url": PORTAL_MACRO_URL,
            "backend": spec.backend,
            "frequency": spec.frequency,
            "unit": spec.unit,
            "original_code": spec.portal_ek,
            "download_date": download_meta["downloaded_at"],
            "latest_period": latest_period,
            "parsing_warnings": warnings,
            "raw_file": str(raw_path),
        }
        LOGGER.info(
            "Fetched %s from %s to %s via %s",
            spec.code,
            PORTAL_MACRO_URL,
            latest_period,
            raw_path,
        )
        return frame, meta

    def _fetch_stream_series(
        self,
        spec: NBSIndicatorSpec,
        output_name: str,
        start: pd.Period | None,
        end: pd.Period | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not all([spec.stream_cid, spec.stream_root_id, spec.stream_indicator_id]):
            raise NBSChinaIndicatorError(
                f"Indicator '{spec.code}' has incomplete portal stream metadata."
            )

        query_start = start or _history_start_for_spec(spec)
        query_end = end or _current_period(spec.frequency)
        if query_start > query_end:
            frame = pd.DataFrame(columns=["period", output_name])
            meta = {
                "output_name": output_name,
                "indicator_code": spec.code,
                "source": "NBS National Data portal stream endpoint",
                "source_url": PORTAL_STREAM_URL,
                "backend": spec.backend,
                "frequency": spec.frequency,
                "unit": spec.unit,
                "original_code": spec.stream_indicator_id,
                "download_date": None,
                "latest_period": None,
                "parsing_warnings": ["requested range starts after the current period"],
                "raw_file": None,
            }
            return frame, meta

        payload = {
            "cid": spec.stream_cid,
            "rootId": spec.stream_root_id,
            "indicatorIds": [spec.stream_indicator_id],
            "das": [{"text": spec.stream_area_name, "value": spec.stream_area_code}],
            "showType": 1,
            "dts": [f"{_format_portal_period(query_start)}-{_format_portal_period(query_end)}"],
        }
        body, raw_path, download_meta = self._request_cached(
            "POST",
            PORTAL_STREAM_URL,
            slug=spec.code,
            json_body=payload,
            expected="json",
        )
        obj = self._parse_json(body, url=PORTAL_STREAM_URL)
        rows = _validate_stream_payload(obj, spec.code)
        frame = _stream_rows_to_frame(rows, spec.frequency, indicator_id=spec.stream_indicator_id)
        frame = _filter_periods(frame, start, end).rename(columns={"value": output_name})
        warnings: list[str] = []
        if frame.empty:
            raise NBSChinaSchemaError(
                f"NBS stream response for '{spec.code}' returned no rows for payload {payload}."
            )
        if frame[output_name].isna().any():
            warnings.append("stream response includes blank or non-numeric observations")
        warnings.extend(_coverage_warnings(frame, output_name, start))

        latest_period = _latest_non_missing_period(frame, output_name)
        meta = {
            "output_name": output_name,
            "indicator_code": spec.code,
            "source": "NBS National Data portal stream endpoint",
            "source_url": PORTAL_STREAM_URL,
            "backend": spec.backend,
            "frequency": spec.frequency,
            "unit": spec.unit,
            "original_code": spec.stream_indicator_id,
            "download_date": download_meta["downloaded_at"],
            "latest_period": latest_period,
            "parsing_warnings": warnings,
            "raw_file": str(raw_path),
        }
        LOGGER.info(
            "Fetched %s from %s to %s via %s",
            spec.code,
            PORTAL_STREAM_URL,
            latest_period,
            raw_path,
        )
        return frame, meta

    def _fetch_pmi_series(
        self,
        spec: NBSIndicatorSpec,
        output_name: str,
        start: pd.Period | None,
        end: pd.Period | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        links = self._discover_pmi_release_links(start=start, end=end)
        periods: list[pd.Period] = []
        values: list[float] = []
        raw_files: list[str] = []
        warnings: list[str] = []
        latest_download_date: str | None = None

        for link_period, url in links:
            body, raw_path, download_meta = self._request_cached(
                "GET",
                url,
                slug=f"pmi_{link_period.strftime('%Y%m')}",
                expected="html",
            )
            value = _extract_manufacturing_pmi(body, expected_period=link_period)
            periods.append(link_period)
            values.append(value)
            raw_files.append(str(raw_path))
            latest_download_date = download_meta["downloaded_at"]
            LOGGER.info(
                "Fetched manufacturing_pmi from %s for %s via %s",
                url,
                link_period,
                raw_path,
            )

        frame = pd.DataFrame({"period": periods, output_name: values})
        frame = _deduplicate_frame(frame, output_name)
        frame = _filter_periods(frame, start, end)
        latest_period = _latest_non_missing_period(frame, output_name)
        meta = {
            "output_name": output_name,
            "indicator_code": spec.code,
            "source": "NBS release archive",
            "source_url": NBS_RELEASE_ARCHIVE_URL,
            "backend": spec.backend,
            "frequency": spec.frequency,
            "unit": spec.unit,
            "original_code": "release_archive_pmi",
            "download_date": latest_download_date,
            "latest_period": latest_period,
            "parsing_warnings": warnings,
            "raw_file": raw_files[-1] if raw_files else None,
            "raw_files": raw_files,
        }
        return frame, meta

    def _fetch_gdp_release_series(
        self,
        spec: NBSIndicatorSpec,
        output_name: str,
        start: pd.Period | None,
        end: pd.Period | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if spec.release_series is None:
            raise NBSChinaIndicatorError(
                f"Indicator '{spec.code}' is missing the GDP release series selector."
            )

        history = self._build_gdp_release_history(desired_start=start)
        if spec.release_series not in history.columns:
            raise NBSChinaSchemaError(
                f"GDP release history does not contain '{spec.release_series}'."
            )
        frame = history[["period", spec.release_series]].rename(
            columns={spec.release_series: output_name}
        )
        frame = _filter_periods(frame, start, end)
        frame = _deduplicate_frame(frame, output_name)
        latest_period = _latest_non_missing_period(frame, output_name)
        raw_files = sorted(history["raw_file"].dropna().astype(str).unique().tolist())
        release_dates = (
            history.loc[history["gdp_release_date"].notna(), ["period", "gdp_release_date"]]
            .drop_duplicates()
            .sort_values("period")
        )
        meta = {
            "output_name": output_name,
            "indicator_code": spec.code,
            "source": "NBS GDP release archive tables",
            "source_url": NBS_RELEASE_ARCHIVE_URL,
            "backend": spec.backend,
            "frequency": spec.frequency,
            "unit": spec.unit,
            "original_code": spec.release_series,
            "download_date": history["download_date"].dropna().astype(str).max() if not history.empty else None,
            "latest_period": latest_period,
            "parsing_warnings": _coverage_warnings(frame, output_name, start),
            "raw_file": raw_files[-1] if raw_files else None,
            "raw_files": raw_files,
            "release_dates": [
                {"period": str(row.period), "release_date": row.gdp_release_date.isoformat()}
                for row in release_dates.itertuples(index=False)
            ],
        }
        return frame, meta

    def _discover_pmi_release_links(
        self,
        *,
        start: pd.Period | None,
        end: pd.Period | None,
    ) -> list[tuple[pd.Period, str]]:
        found: dict[pd.Period, str] = {}
        consecutive_no_match_pages = 0

        for page_idx in range(DEFAULT_MAX_RELEASE_ARCHIVE_PAGES):
            page_url = (
                urljoin(NBS_RELEASE_ARCHIVE_URL, "index.html")
                if page_idx == 0
                else urljoin(NBS_RELEASE_ARCHIVE_URL, f"index_{page_idx}.html")
            )
            try:
                body, _, _ = self._request_cached(
                    "GET",
                    page_url,
                    slug=f"release_archive_{page_idx}",
                    expected="html",
                )
            except NBSChinaRequestError as exc:
                if "HTTP 404" in str(exc) and page_idx > 0:
                    break
                raise

            matches_on_page = 0
            oldest_match: pd.Period | None = None
            for title, href in _extract_links(body, base_url=page_url):
                match = PMI_TITLE_RE.search(title)
                if not match:
                    continue
                matches_on_page += 1
                period = pd.Period(
                    f"{match.group('year')}-{int(match.group('month')):02d}",
                    freq="M",
                )
                oldest_match = period if oldest_match is None else min(oldest_match, period)
                if start is not None and period < start:
                    continue
                if end is not None and period > end:
                    continue
                found.setdefault(period, href)

            if matches_on_page == 0:
                consecutive_no_match_pages += 1
                if found and consecutive_no_match_pages >= 12:
                    break
            else:
                consecutive_no_match_pages = 0
                if start is not None and oldest_match is not None and oldest_match < start:
                    break

        if not found:
            raise NBSChinaSchemaError(
                "No PMI release links were discovered in the official NBS release archive."
            )
        return sorted(found.items(), key=lambda item: item[0])

    def _build_gdp_release_history(self, *, desired_start: pd.Period | None) -> pd.DataFrame:
        if self._gdp_history_cache is not None:
            cached_min = self._gdp_history_cache["period"].min()
            if desired_start is None or cached_min <= desired_start:
                return self._gdp_history_cache.copy()

        value_frames: list[pd.DataFrame] = []
        release_rows: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        earliest_covered: pd.Period | None = None
        saw_any_gdp_release = False
        consecutive_no_match_pages = 0

        for page_idx in range(DEFAULT_MAX_RELEASE_ARCHIVE_PAGES):
            page_url = (
                urljoin(NBS_RELEASE_ARCHIVE_URL, "index.html")
                if page_idx == 0
                else urljoin(NBS_RELEASE_ARCHIVE_URL, f"index_{page_idx}.html")
            )
            try:
                body, _, _ = self._request_cached(
                    "GET",
                    page_url,
                    slug=f"release_archive_{page_idx}",
                    expected="html",
                )
            except NBSChinaRequestError as exc:
                if "HTTP 404" in str(exc) and page_idx > 0:
                    break
                raise

            gdp_links_on_page = 0
            for title, href in _extract_links(body, base_url=page_url):
                title_match = GDP_RELEASE_TITLE_RE.search(title)
                if title_match is None:
                    continue
                gdp_links_on_page += 1
                saw_any_gdp_release = True
                if href in seen_urls:
                    continue
                seen_urls.add(href)
                release_date = _release_date_from_url(href)
                current_quarter = pd.Period(
                    f"{title_match.group('year')}Q{_infer_current_quarter(title_match.group('label'))}",
                    freq="Q-DEC",
                )
                release_rows.append(
                    {
                        "period": current_quarter,
                        "gdp_release_date": release_date,
                        "release_url": href,
                    }
                )
                article_body, raw_path, download_meta = self._request_cached(
                    "GET",
                    href,
                    slug=f"gdp_release_{release_date.strftime('%Y%m%d')}",
                    expected="html",
                )
                parsed = _extract_gdp_release_tables(article_body)
                if parsed.empty:
                    continue
                parsed["page_release_date"] = release_date
                parsed["page_url"] = href
                parsed["raw_file"] = str(raw_path)
                parsed["download_date"] = download_meta["downloaded_at"]
                value_frames.append(parsed)
                page_min = parsed["period"].min()
                earliest_covered = page_min if earliest_covered is None else min(earliest_covered, page_min)

            if gdp_links_on_page == 0:
                consecutive_no_match_pages += 1
                if saw_any_gdp_release and consecutive_no_match_pages >= 12:
                    break
            else:
                consecutive_no_match_pages = 0

            if desired_start is not None and earliest_covered is not None and earliest_covered <= desired_start:
                break

        if not saw_any_gdp_release:
            raise NBSChinaSchemaError(
                "No GDP release links were discovered in the official NBS release archive."
            )
        if not value_frames:
            raise NBSChinaSchemaError(
                "GDP release links were found, but no parsable GDP growth tables were extracted."
            )

        combined = pd.concat(value_frames, ignore_index=True)
        history = combined[["period"]].drop_duplicates().sort_values("period").reset_index(drop=True)
        for value_col in ["gdp_real_yoy", "gdp_real_qoq_sa"]:
            if value_col not in combined.columns:
                continue
            sub = combined.loc[combined[value_col].notna(), ["period", value_col, "page_release_date"]].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["period", "page_release_date"]).drop_duplicates("period", keep="last")
            history = history.merge(sub.drop(columns=["page_release_date"]), on="period", how="left")

        latest_meta = (
            combined.sort_values(["period", "page_release_date"])
            .drop_duplicates("period", keep="last")[["period", "page_release_date", "page_url", "raw_file", "download_date"]]
        )
        history = history.merge(latest_meta, on="period", how="left")

        release_schedule = pd.DataFrame(release_rows)
        if not release_schedule.empty:
            release_schedule = (
                release_schedule.sort_values(["period", "gdp_release_date"])
                .drop_duplicates("period", keep="first")
            )
            history = history.merge(release_schedule, on="period", how="left")
        else:
            history["gdp_release_date"] = history["period"].apply(_default_gdp_release_date)
            history["release_url"] = None

        missing_release = history["gdp_release_date"].isna()
        if missing_release.any():
            history.loc[missing_release, "gdp_release_date"] = history.loc[missing_release, "period"].apply(
                _default_gdp_release_date
            )

        history = history.sort_values("period").reset_index(drop=True)
        self._gdp_history_cache = history
        return history.copy()

    def _request_cached(
        self,
        method: Literal["GET", "POST"],
        url: str,
        *,
        slug: str,
        expected: Literal["json", "html"],
        json_body: dict[str, Any] | None = None,
    ) -> tuple[str, Path, dict[str, Any]]:
        cache_key = _request_cache_key(method=method, url=url, json_body=json_body)
        extension = "json" if expected == "json" else "html"
        raw_path = self.raw_dir / f"{slug}.{cache_key}.{extension}"
        meta_path = raw_path.with_suffix(raw_path.suffix + ".meta.json")

        if raw_path.exists() and meta_path.exists() and not self.refresh:
            return raw_path.read_text(encoding="utf-8"), raw_path, json.loads(
                meta_path.read_text(encoding="utf-8")
            )

        self._rate_limit()
        try:
            response = self.session.request(
                method,
                url,
                json=json_body,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise NBSChinaRequestError(f"Request failed for {url}: {exc}") from exc

        if response.status_code >= 400:
            raise NBSChinaRequestError(f"HTTP {response.status_code} for {url}")

        header_encoding = None
        headers = getattr(response, "headers", {}) or {}
        content_type = str(headers.get("Content-Type", ""))
        charset_match = re.search(r"charset=([\w-]+)", content_type, flags=re.I)
        if charset_match:
            header_encoding = charset_match.group(1)
        response.encoding = header_encoding or "utf-8"
        body = response.text
        if expected == "json" and body.lstrip().startswith("<!DOCTYPE html"):
            raise NBSChinaSchemaError(
                f"Expected JSON from {url}, but received HTML. "
                "The endpoint may be behind an anti-bot challenge or may have changed."
            )

        raw_path.write_text(body, encoding="utf-8")
        download_meta = {
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "source_url": response.url,
            "status_code": response.status_code,
            "expected": expected,
            "request_method": method,
            "request_payload": json_body,
        }
        meta_path.write_text(
            json.dumps(download_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return body, raw_path, download_meta

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def _parse_json(self, body: str, *, url: str) -> dict[str, Any]:
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            raise NBSChinaSchemaError(f"Invalid JSON returned by {url}: {exc}") from exc
        if not isinstance(obj, dict):
            raise NBSChinaSchemaError(f"Expected a JSON object from {url}.")
        return obj

    def _append_log(self, meta: dict[str, Any]) -> None:
        log_row = {
            "download_date": meta.get("download_date"),
            "source_url": meta.get("source_url"),
            "indicator_code": meta.get("indicator_code"),
            "output_name": meta.get("output_name"),
            "latest_period": meta.get("latest_period"),
            "parsing_warnings": meta.get("parsing_warnings", []),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")


def _resolve_indicator(code: str, frequency: Frequency) -> NBSIndicatorSpec:
    spec = NBS_INDICATORS.get(code)
    if spec is None:
        raise NBSChinaIndicatorError(
            f"Unknown indicator '{code}'. Use available_nbs_indicators() to inspect "
            "the built-in registry."
        )
    if spec.frequency != frequency:
        raise NBSChinaIndicatorError(
            f"Indicator '{code}' has frequency '{spec.frequency}', not '{frequency}'."
        )
    return spec


def _normalize_frequency(frequency: str) -> Frequency:
    key = str(frequency).strip().lower()
    try:
        return _FREQUENCY_ALIASES[key]
    except KeyError as exc:
        raise ValueError("frequency must be monthly, quarterly, or annual.") from exc


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    base = Path.cwd() / ".cache" / "nbs_china" if cache_dir is None else Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _request_cache_key(
    *,
    method: str,
    url: str,
    json_body: dict[str, Any] | None,
) -> str:
    payload = {"method": method.upper(), "url": url, "json": json_body}
    serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return sha256(serialised.encode("utf-8")).hexdigest()[:16]


def _validate_macro_payload(obj: dict[str, Any], indicator_code: str) -> list[dict[str, Any]]:
    if obj.get("success") is not True:
        raise NBSChinaSchemaError(
            f"NBS macro response for '{indicator_code}' did not report success: {obj!r}"
        )
    rows = obj.get("data")
    if not isinstance(rows, list):
        raise NBSChinaSchemaError(
            f"NBS macro response for '{indicator_code}' does not contain a data list."
        )
    required = {"dt", "v"}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise NBSChinaSchemaError(
                f"NBS macro response for '{indicator_code}' has malformed rows."
            )
    return rows


def _validate_stream_payload(obj: dict[str, Any], indicator_code: str) -> list[dict[str, Any]]:
    if obj.get("success") is not True:
        raise NBSChinaSchemaError(
            f"NBS stream response for '{indicator_code}' did not report success: {obj!r}"
        )
    rows = obj.get("data")
    if not isinstance(rows, list):
        raise NBSChinaSchemaError(
            f"NBS stream response for '{indicator_code}' does not contain a data list."
        )
    for row in rows:
        if not isinstance(row, dict) or "code" not in row or "values" not in row:
            raise NBSChinaSchemaError(
                f"NBS stream response for '{indicator_code}' has malformed rows."
            )
    return rows


def _portal_rows_to_frame(rows: list[dict[str, Any]], frequency: Frequency) -> pd.DataFrame:
    data = []
    for row in rows:
        period = _parse_portal_period(str(row["dt"]), frequency)
        value = pd.to_numeric(row["v"], errors="coerce")
        data.append({"period": period, "value": float(value) if pd.notna(value) else float("nan")})
    frame = pd.DataFrame(data)
    return _deduplicate_frame(frame, "value")


def _stream_rows_to_frame(
    rows: list[dict[str, Any]],
    frequency: Frequency,
    *,
    indicator_id: str | None,
) -> pd.DataFrame:
    data = []
    for row in rows:
        period = _parse_portal_period(str(row["code"]), frequency)
        values = row.get("values")
        if not isinstance(values, list):
            raise NBSChinaSchemaError("NBS stream row does not contain a values list.")
        selected: dict[str, Any] | None = None
        if indicator_id is not None:
            for candidate in values:
                if isinstance(candidate, dict) and candidate.get("_id") == indicator_id:
                    selected = candidate
                    break
        if selected is None and values:
            selected = values[0] if isinstance(values[0], dict) else None
        if selected is None:
            raise NBSChinaSchemaError("NBS stream row did not contain a usable value block.")
        value = pd.to_numeric(selected.get("value"), errors="coerce")
        data.append({"period": period, "value": float(value) if pd.notna(value) else float("nan")})
    frame = pd.DataFrame(data)
    return _deduplicate_frame(frame, "value")


def _parse_portal_period(raw_period: str, frequency: Frequency) -> pd.Period:
    if frequency == "monthly":
        match = re.fullmatch(r"(\d{4})(\d{2})MM", raw_period)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            return pd.Period(f"{year}-{month:02d}", freq="M")
    elif frequency == "quarterly":
        match = re.fullmatch(r"(\d{4})(\d{2})SS", raw_period)
        if match:
            year, quarter = int(match.group(1)), int(match.group(2))
            if quarter not in {1, 2, 3, 4}:
                raise NBSChinaSchemaError(f"Unexpected quarterly period '{raw_period}'.")
            return pd.Period(f"{year}Q{quarter}", freq="Q-DEC")
    elif frequency == "annual":
        match = re.fullmatch(r"(\d{4})YY", raw_period)
        if match:
            return pd.Period(match.group(1), freq="Y-DEC")
    raise NBSChinaSchemaError(
        f"Could not parse portal period '{raw_period}' for frequency '{frequency}'."
    )


def _format_portal_period(period: pd.Period) -> str:
    if period.freqstr.startswith("M"):
        return f"{period.year}{period.month:02d}MM"
    if period.freqstr.startswith("Q"):
        return f"{period.year}{period.quarter:02d}SS"
    if period.freqstr.startswith("Y"):
        return f"{period.year}YY"
    raise NBSChinaSchemaError(f"Unsupported pandas period frequency '{period.freqstr}'.")


def _filter_periods(
    frame: pd.DataFrame,
    start: pd.Period | None,
    end: pd.Period | None,
) -> pd.DataFrame:
    out = frame.copy()
    if start is not None:
        out = out[out["period"] >= start]
    if end is not None:
        out = out[out["period"] <= end]
    return out.sort_values("period").reset_index(drop=True)


def _deduplicate_frame(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    duplicated = frame.duplicated(subset=["period"], keep=False)
    if duplicated.any():
        dupes = frame.loc[duplicated].sort_values("period")
        for period, block in dupes.groupby("period", sort=False):
            unique_values = block[value_column].dropna().astype(str).unique().tolist()
            if len(unique_values) > 1:
                raise NBSChinaSchemaError(
                    f"Conflicting duplicate values detected for period {period}: {unique_values}"
                )
        frame = frame.drop_duplicates(subset=["period"], keep="first")
    return frame.sort_values("period").reset_index(drop=True)


def _parse_bound(value: str | None, frequency: Frequency) -> pd.Period | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace("/", "-")
    if not text:
        return None
    if frequency == "monthly":
        digits = text.replace("-", "")
        if re.fullmatch(r"\d{6}", digits):
            return pd.Period(f"{digits[:4]}-{digits[4:6]}", freq="M")
        match = re.fullmatch(r"(\d{4})M(\d{1,2})", text)
        if match:
            return pd.Period(f"{match.group(1)}-{int(match.group(2)):02d}", freq="M")
    elif frequency == "quarterly":
        match = re.fullmatch(r"(\d{4})[- ]?Q([1-4])", text)
        if match:
            return pd.Period(f"{match.group(1)}Q{match.group(2)}", freq="Q-DEC")
    elif frequency == "annual":
        if re.fullmatch(r"\d{4}", text):
            return pd.Period(text, freq="Y-DEC")
    raise ValueError(f"Could not parse '{value}' as a {frequency} period.")


def _current_period(frequency: Frequency) -> pd.Period:
    now = pd.Timestamp.utcnow().tz_localize(None)
    if frequency == "monthly":
        return now.to_period("M")
    if frequency == "quarterly":
        return now.to_period("Q")
    if frequency == "annual":
        return now.to_period("Y")
    raise ValueError(f"Unsupported frequency '{frequency}'.")


def _history_start_for_spec(spec: NBSIndicatorSpec) -> pd.Period:
    if spec.history_start is not None:
        return _parse_bound(spec.history_start, spec.frequency)  # type: ignore[return-value]
    fallback = {"monthly": "1990M01", "quarterly": "1990Q1", "annual": "1990"}
    return _parse_bound(fallback[spec.frequency], spec.frequency)  # type: ignore[return-value]


def _latest_non_missing_period(frame: pd.DataFrame, value_column: str) -> str | None:
    if frame.empty or value_column not in frame:
        return None
    observed = frame.loc[frame[value_column].notna(), "period"]
    if observed.empty:
        return None
    return str(observed.iloc[-1])


def _coverage_warnings(
    frame: pd.DataFrame,
    value_column: str,
    requested_start: pd.Period | None,
) -> list[str]:
    if frame.empty or value_column not in frame:
        return ["no rows were returned"]
    observed = frame.loc[frame[value_column].notna(), "period"]
    if observed.empty:
        return ["all retrieved observations are missing"]
    warnings: list[str] = []
    first_obs = observed.iloc[0]
    if requested_start is not None and first_obs > requested_start:
        warnings.append(f"first non-missing observation is {first_obs}")
    return warnings


def _extract_manufacturing_pmi(body: str, *, expected_period: pd.Period) -> float:
    text = _normalise_text(_html_to_text(body))
    title_match = PMI_TITLE_RE.search(text)
    if not title_match:
        raise NBSChinaSchemaError("Could not identify the PMI release title in the article.")
    title_period = pd.Period(
        f"{title_match.group('year')}-{int(title_match.group('month')):02d}",
        freq="M",
    )
    if title_period != expected_period:
        raise NBSChinaSchemaError(
            f"PMI article period mismatch: expected {expected_period}, found {title_period}."
        )

    patterns = [
        r"(\d{1,2})月份，制造业采购经理指数.?PMI.?为([0-9]+(?:\.[0-9]+)?)%",
        r"制造业采购经理指数.?PMI.?为([0-9]+(?:\.[0-9]+)?)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        if len(match.groups()) == 2:
            month, value = int(match.group(1)), float(match.group(2))
            if month != expected_period.month:
                raise NBSChinaSchemaError(
                    f"PMI article body month mismatch: expected {expected_period.month}, found {month}."
                )
            return value
        return float(match.group(1))
    raise NBSChinaSchemaError("Could not parse the manufacturing PMI value from the article.")


def _extract_gdp_release_tables(body: str) -> pd.DataFrame:
    if "国内生产总值初步核算结果" not in body:
        raise NBSChinaSchemaError("Page does not look like an official GDP release page.")

    grouped_frames: dict[str, list[pd.DataFrame]] = {"gdp_real_yoy": [], "gdp_real_qoq_sa": []}
    for rows, context in _extract_html_tables_with_context(body):
        if not _looks_like_gdp_quarter_grid(rows):
            continue
        label = _classify_gdp_table(context)
        if label is None:
            continue
        grouped_frames[label].append(_parse_gdp_quarter_grid(rows, label))

    parsed_frames: list[pd.DataFrame] = []
    for label, frames in grouped_frames.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined = _deduplicate_frame(combined, label)
        parsed_frames.append(combined)

    if not parsed_frames:
        raise NBSChinaSchemaError(
            "Could not isolate the GDP growth tables from the official release page."
        )

    out = parsed_frames[0]
    for frame in parsed_frames[1:]:
        out = out.merge(frame, on="period", how="outer", validate="one_to_one")
    return out.sort_values("period").reset_index(drop=True)


def _extract_html_tables_with_context(body: str) -> list[tuple[list[list[str]], str]]:
    tables: list[tuple[list[list[str]], str]] = []
    for match in re.finditer(r"<table.*?</table>", body, flags=re.S | re.I):
        table_html = match.group(0)
        rows: list[list[str]] = []
        for tr_html in re.findall(r"<tr.*?</tr>", table_html, flags=re.S | re.I):
            cells = []
            for cell_html in re.findall(r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", tr_html, flags=re.S | re.I):
                cells.append(_clean_text(_html_to_text(cell_html)))
            if cells:
                rows.append(cells)
        if not rows:
            continue
        pre = body[max(0, match.start() - 900):match.start()]
        context = _normalise_text(_html_to_text(pre))
        tables.append((rows, context))
    return tables


def _looks_like_gdp_quarter_grid(rows: list[list[str]]) -> bool:
    if len(rows) < 3:
        return False
    for row in rows[:4]:
        if len(row) >= 5 and row[0] == "年份":
            wanted = {"1季度", "2季度", "3季度", "4季度"}
            if wanted.issubset(set(row[1:5])):
                return True
    return False


def _classify_gdp_table(context: str) -> Literal["gdp_real_yoy", "gdp_real_qoq_sa"] | None:
    plain = _clean_text(context)
    yoy_positions = [plain.rfind("GDP同比增长速度"), plain.rfind("同比增长速度为与上年同期对比的增长速度")]
    qoq_positions = [plain.rfind("GDP环比增长速度"), plain.rfind("经季节调整后与上一季度对比的增长速度")]
    yoy_pos = max(yoy_positions)
    qoq_pos = max(qoq_positions)
    if yoy_pos == -1 and qoq_pos == -1:
        return None
    if qoq_pos > yoy_pos:
        return "gdp_real_qoq_sa"
    if yoy_pos > qoq_pos:
        return "gdp_real_yoy"
    return None


def _parse_gdp_quarter_grid(
    rows: list[list[str]],
    value_name: Literal["gdp_real_yoy", "gdp_real_qoq_sa"],
) -> pd.DataFrame:
    parsed: list[dict[str, Any]] = []
    header_found = False
    for row in rows:
        if len(row) >= 5 and row[0] == "年份":
            header_found = True
            continue
        if not header_found:
            continue
        if not row or not re.fullmatch(r"20\d{2}", row[0]):
            continue
        year = int(row[0])
        for quarter, value in enumerate(row[1:5], start=1):
            value = str(value).strip()
            if value in {"", "--", "...", "…"}:
                continue
            parsed.append(
                {
                    "period": pd.Period(f"{year}Q{quarter}", freq="Q-DEC"),
                    value_name: float(value),
                }
            )
    if not parsed:
        raise NBSChinaSchemaError(
            f"GDP table labelled '{value_name}' was found but did not contain parsable rows."
        )
    return pd.DataFrame(parsed).sort_values("period").drop_duplicates("period", keep="last")


def _infer_current_quarter(label: str) -> int:
    mapping = {
        "一季度": 1,
        "二季度和上半年": 2,
        "上半年": 2,
        "三季度": 3,
        "前三季度": 3,
        "四季度和全年": 4,
        "全年": 4,
    }
    try:
        return mapping[label]
    except KeyError as exc:
        raise NBSChinaSchemaError(f"Unknown GDP release label '{label}'.") from exc


def _default_gdp_release_date(period: pd.Period) -> pd.Timestamp:
    quarter_end_month = period.asfreq("M", how="end").to_timestamp()
    release_month = quarter_end_month + pd.DateOffset(months=1)
    return pd.Timestamp(year=release_month.year, month=release_month.month, day=18)


def _release_date_from_url(url: str) -> pd.Timestamp:
    match = re.search(r"t(\d{8})_", url)
    if not match:
        raise ValueError(f"Could not infer a release date from {url}")
    return pd.Timestamp(match.group(1))


def _extract_links(body: str, *, base_url: str) -> list[tuple[str, str]]:
    parser = _LinkExtractor()
    parser.feed(body)
    links = []
    for href, text in parser.links:
        clean_text = _normalise_text(text)
        if clean_text and href:
            links.append((clean_text, urljoin(base_url, href)))
    return links


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = dict(attrs)
        self._current_href = attr_map.get("href")
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = "".join(self._text_parts)
        self.links.append((self._current_href, text))
        self._current_href = None
        self._text_parts = []


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth == 0 and name in {"p", "div", "br", "li", "h1", "h2", "h3", "tr", "td", "th"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in {"script", "style"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if self._ignored_depth == 0 and name in {"p", "div", "li", "h1", "h2", "h3", "tr", "td", "th"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)


def _html_to_text(body: str) -> str:
    parser = _TextExtractor()
    parser.feed(body)
    return unescape("".join(parser.parts))


def _normalise_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _clean_text(text: str) -> str:
    return _normalise_text(text).replace(" ", "").replace(" ", "")
