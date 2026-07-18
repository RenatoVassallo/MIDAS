"""Dynamic Factor Model nowcaster.

``DFMNowcaster`` wraps ``statsmodels.tsa.DynamicFactorMQ`` (the Banbura-Modugno
EM implementation) behind the :class:`~MIDAS.base.BaseNowcaster` contract. It is
the primary nowcasting model: it handles the ragged edge natively through the
Kalman filter, uses *all* monthly indicators at once via a small number of common
factors, and estimates on the unbalanced panel without discarding long histories.

The quarterly target enters the state space as an observed quarterly series (with
its own AR(1) idiosyncratic component); its nowcast is the smoothed estimate at
the target quarter's end month. Factors can be a handful of global factors or a
global-plus-group block structure driven by the metadata ``group`` column.

Backtest efficiency: EM estimation (~seconds) runs once per target quarter and is
cached; each intra-quarter origin only re-runs the Kalman filter via
``results.apply`` (~0.1 s), keeping full backtests affordable. This is valid
because the harness visits all origins of a quarter consecutively.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import BaseNowcaster, InformationSet, NowcastResult


class DFMNowcaster(BaseNowcaster):
    """Mixed-frequency dynamic factor nowcaster (wraps ``DynamicFactorMQ``).

    Parameters
    ----------
    factors:
        Number of global factors (``int``), or ``"groups"`` for a global factor
        plus one block factor per economic ``group`` from the metadata.
    factor_orders:
        VAR order of the factor process.
    quarterly_vars:
        Quarterly series to place in the state space. ``None`` uses the target
        alone; pass several (for example all three aggregates) for joint models.
    monthly_vars:
        Monthly predictor subset. ``None`` uses every monthly column available.
    idiosyncratic_ar1, standardize, maxiter:
        Passed through to ``DynamicFactorMQ`` / its EM fit.
    """

    def __init__(
        self,
        factors: int | str = 2,
        factor_orders: int = 1,
        *,
        quarterly_vars: Sequence[str] | None = None,
        monthly_vars: Sequence[str] | None = None,
        idiosyncratic_ar1: bool = True,
        standardize: bool = True,
        maxiter: int = 200,
        min_series_obs: int = 12,
        covid_window: tuple[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        self.factors = factors
        self.factor_orders = factor_orders
        self.quarterly_vars = None if quarterly_vars is None else list(quarterly_vars)
        self.monthly_vars = None if monthly_vars is None else list(monthly_vars)
        self.idiosyncratic_ar1 = idiosyncratic_ar1
        self.standardize = standardize
        self.maxiter = maxiter
        self.min_series_obs = min_series_obs
        # (start, end) months to treat as missing (COVID outlier control). The
        # Kalman filter interpolates them, so extreme pandemic observations neither
        # distort the factor estimates nor drive explosive nowcasts.
        self.covid_window = covid_window
        self._name = name or ("DFM" if isinstance(factors, int) else "DFM-blocks")
        # Two-level cache. EM parameters are keyed by the origin's *quarter* (they move
        # slowly), the filtered results by the exact origin. A given origin's filtered
        # results serve every forecast horizon, since only the target period changes.
        self._param_key: pd.Period | None = None
        self._applied_key: pd.Timestamp | None = None
        self._cached_results = None
        self.results_ = None

    # ----------------------------------------------------------------- internals
    def _frames(self, info: InformationSet) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        m_vars = self.monthly_vars or list(info.monthly.columns)
        q_vars = self.quarterly_vars or [info.target]
        monthly = info.monthly[m_vars].copy()
        monthly.index = monthly.index.to_period("M")
        quarterly = info.quarterly[q_vars].copy()
        quarterly.index = quarterly.index.to_period("Q")
        if self.covid_window is not None:
            lo, hi = self.covid_window
            monthly.loc[(monthly.index >= pd.Period(lo, "M")) & (monthly.index <= pd.Period(hi, "M")), :] = np.nan
            quarterly.loc[(quarterly.index >= pd.Period(lo, "Q")) & (quarterly.index <= pd.Period(hi, "Q")), :] = np.nan
        return monthly, quarterly, m_vars

    def _factor_spec(self, info: InformationSet, m_vars: list[str], q_vars: list[str]):
        """Return the ``factors`` argument for ``DynamicFactorMQ``."""
        if isinstance(self.factors, int):
            return self.factors
        if self.factors == "groups":
            if info.metadata is None:
                raise ValueError("factors='groups' needs info.metadata")
            spec: dict[str, list[str]] = {}
            for col in m_vars:
                spec[col] = ["Global", info.metadata.group_of(col)]
            for col in q_vars:
                spec[col] = ["Global"]  # target loads on the global business-cycle factor
            return spec
        raise ValueError(f"unknown factors specification: {self.factors!r}")

    def _build_model(self, monthly: pd.DataFrame, quarterly: pd.DataFrame, info: InformationSet):
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        factors = self._factor_spec(info, list(monthly.columns), list(quarterly.columns))
        return DynamicFactorMQ(
            endog=monthly,
            endog_quarterly=quarterly,
            factors=factors,
            factor_orders=self.factor_orders,
            idiosyncratic_ar1=self.idiosyncratic_ar1,
            standardize=self.standardize,
        )

    # -------------------------------------------------------------------- fit/now
    def fit(self, info: InformationSet) -> "DFMNowcaster":
        pkey = pd.Period(pd.Timestamp(info.origin), freq="Q")
        akey = pd.Timestamp(info.origin)
        if self._applied_key != akey or self.results_ is None:
            monthly, quarterly, _ = self._frames(info)
            if self._param_key != pkey or self._cached_results is None:
                # Unbalanced panel: include only series with enough observations at this
                # origin (a not-yet-started series is entirely NaN and DynamicFactorMQ
                # rejects all-NaN columns). Held fixed while these parameters are reused.
                self._active_cols = [
                    c for c in monthly.columns if monthly[c].notna().sum() >= self.min_series_obs
                ]
                model = self._build_model(monthly[self._active_cols], quarterly, info)
                self._cached_results = model.fit(disp=False, maxiter=self.maxiter)
                self._param_key = pkey
                self.results_ = self._cached_results
            else:
                # parameters still valid, new origin: reuse them, only re-run the filter
                self.results_ = self._cached_results.apply(
                    endog=monthly[self._active_cols], endog_quarterly=quarterly,
                    retain_standardization=True,
                )
            self._applied_key = akey
            self._monthly_index = monthly.index
        self._target = info.target
        self._target_period = info.target_period
        return self

    def _extract(self, info: InformationSet) -> tuple[float, float | None]:
        end_month = pd.Period(pd.Timestamp(info.target_period), freq="M")
        # Position by period arithmetic rather than an index lookup: this also works
        # for target quarters BEYOND the sample, where the Kalman filter forecasts.
        pos = int((end_month - self._monthly_index[0]).n)
        if pos < 0:
            raise ValueError(f"target period {info.target_period} precedes the sample")
        pred = self.results_.get_prediction(start=pos, end=pos)
        mean = float(_cell(pred.predicted_mean, self._target))
        try:
            std = float(_cell(pred.se_mean, self._target))
        except Exception:
            std = None
        return mean, std

    def _plausible(self, value: float, info: InformationSet) -> bool:
        """Is this value anywhere near the target's historical range?"""
        if not np.isfinite(value):
            return False
        y = info.quarterly[info.target].dropna()
        if y.empty:
            return True
        lo, hi = float(y.min()), float(y.max())
        rng = max(hi - lo, 1e-6)
        return (lo - 5.0 * rng) <= value <= (hi + 5.0 * rng)

    def nowcast(self, info: InformationSet) -> NowcastResult:
        if self.results_ is None:
            raise ValueError("fit before nowcast")
        mean, std = self._extract(info)
        if not self._plausible(mean, info):
            # EM can return a marginally non-stationary parameter set (a root just above
            # one). It survives on its own fitting sample, but reusing it on a different
            # origin's data lets the Kalman state diverge: we observed a GDP nowcast of
            # -1.4e6 from parameters that were fine where they were estimated. Drop the
            # cache and re-estimate at this origin rather than emit garbage.
            self._param_key = None
            self._applied_key = None
            self._cached_results = None
            self.fit(info)
            mean, std = self._extract(info)
            if not self._plausible(mean, info):
                return NowcastResult(mean=float("nan"), model=self.name,
                                     extra={"diverged": True})
        return NowcastResult(mean=mean, std=std, model=self.name,
                             extra={"factors": self.factors,
                                    "end_month": str(pd.Period(pd.Timestamp(info.target_period), freq="M"))})

    # ------------------------------------------------------------------- helpers
    def smoothed_factors(self) -> pd.DataFrame:
        """Smoothed common factors from the most recent fit (for plotting)."""
        if self.results_ is None:
            raise ValueError("fit before requesting factors")
        return self.results_.factors.smoothed


def _cell(obj, column: str) -> float:
    """Extract a single value for ``column`` from a 1-row DataFrame or a Series."""
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[0][column]
    return obj[column] if column in getattr(obj, "index", []) else obj.iloc[0]
