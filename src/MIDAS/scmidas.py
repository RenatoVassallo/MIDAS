"""Staggered-Combination MIDAS (SC-MIDAS).

A compact implementation of the Bank of England's headline nowcasting method (Moreira, 2025,
"Nowcasting GDP at the Bank of England: a Staggered-Combination MIDAS approach"). SC-MIDAS
exploits ``hard`` monthly GDP and ``soft`` survey signals *optimally through the release cycle*:
soft data drives the nowcast early, when no monthly GDP of the target quarter is out, and the
weight shifts to hard monthly GDP as it arrives.

The class reuses the tested single-indicator pieces from :mod:`MIDAS.benchmarks`
(:class:`~MIDAS.benchmarks.ADLMIDASNowcaster` for the survey MIDAS regressions and
:class:`~MIDAS.benchmarks.MonthlyARNowcaster` for the time-aggregated monthly-GDP nowcast), and
adds the two-step staggered combination on top, exactly as :class:`~MIDAS.dfm.DFMNowcaster` wraps
``DynamicFactorMQ``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import BaseNowcaster, InformationSet, NowcastResult
from .benchmarks import ADLMIDASNowcaster, MonthlyARNowcaster, _observed_target, _quarter_months


class SCMIDASNowcaster(BaseNowcaster):
    """Staggered-Combination MIDAS on a curated hard + soft set (Moreira 2025, simplified).

    * **Step 1 (soft).** One restricted Beta-MIDAS regression per survey indicator
      (:class:`~MIDAS.benchmarks.ADLMIDASNowcaster`), combined by equal weights into a single
      *soft* nowcast. Surveys are timely (short publication delay), so this is available early.
    * **Step 2 (hard).** A monthly-GDP nowcast, time-aggregated to the quarter
      (:class:`~MIDAS.benchmarks.MonthlyARNowcaster`). Monthly GDP is lagged, so this sharpens
      only late in the cycle.
    * **Stagger.** The two are combined with a weight on the hard block equal to the share of the
      target quarter's three monthly-GDP months already released. The nowcast is therefore pure
      soft early and shifts toward hard as monthly GDP arrives: the *staggered* soft-to-hard
      transition that is the method's defining feature.

    Parameters
    ----------
    hard_var:
        Monthly GDP column (the ``hard`` signal).
    soft_vars:
        Survey columns (the ``soft`` block). ``None`` uses every monthly column except
        ``hard_var``; pass the curated survey list for the faithful BoE set.
    n_lags, use_target_lag:
        Passed to each survey MIDAS regression.
    hard_order:
        AR order of the monthly-GDP model (BoE uses AR(2)).

    Notes
    -----
    This captures SC-MIDAS's structure but simplifies the paper in two honest ways:
    (i) the soft/hard weights are **availability-driven** rather than learned from out-of-sample
    RMSE at each release stage; (ii) every component uses restricted MIDAS / time-aggregation,
    rather than the paper's mix of restricted (surveys) and unrestricted (monthly GDP) MIDAS.
    Both are natural extension points.
    """

    def __init__(
        self,
        hard_var: str = "g_pbim",
        soft_vars: Sequence[str] | None = None,
        *,
        n_lags: int = 12,
        use_target_lag: bool = True,
        hard_order: int = 2,
        min_obs: int = 20,
        name: str = "SC-MIDAS",
    ) -> None:
        self.hard_var = hard_var
        self.soft_vars = None if soft_vars is None else list(soft_vars)
        self.n_lags = n_lags
        self.use_target_lag = use_target_lag
        self.hard_order = hard_order
        self.min_obs = min_obs
        self._name = name

    def fit(self, info: InformationSet) -> "SCMIDASNowcaster":
        soft = self.soft_vars or [c for c in info.monthly.columns if c != self.hard_var]
        self._soft: dict[str, ADLMIDASNowcaster] = {}
        for v in soft:
            member = ADLMIDASNowcaster(indicators=[v], n_lags=self.n_lags,
                                       use_target_lag=self.use_target_lag, min_obs=self.min_obs)
            try:
                member.fit(info)
            except Exception:
                continue
            self._soft[v] = member
        self._hard = MonthlyARNowcaster(monthly_gdp=self.hard_var, order=self.hard_order,
                                        min_obs=self.min_obs).fit(info)
        return self

    def _stagger_weight(self, info: InformationSet) -> float:
        """Share of the target quarter's three monthly-GDP months released at this origin."""
        g = info.monthly[self.hard_var]
        months = _quarter_months(info.target_period)
        released = sum(1 for m in months if np.isfinite(g.get(m, np.nan)))
        return released / 3.0

    def nowcast(self, info: InformationSet) -> NowcastResult:
        soft_preds = []
        for member in self._soft.values():
            try:
                r = member.nowcast(info)
            except Exception:
                continue
            if np.isfinite(r.mean):
                soft_preds.append(float(r.mean))
        soft_hat = float(np.mean(soft_preds)) if soft_preds else float("nan")
        try:
            hard_hat = float(self._hard.nowcast(info).mean)
        except Exception:
            hard_hat = float("nan")

        w = self._stagger_weight(info)
        if np.isfinite(soft_hat) and np.isfinite(hard_hat):
            mean = (1.0 - w) * soft_hat + w * hard_hat
        elif np.isfinite(hard_hat):
            mean = hard_hat
        elif np.isfinite(soft_hat):
            mean = soft_hat
        else:
            y = _observed_target(info)
            mean = float(y.iloc[-1]) if not y.empty else float("nan")
        return NowcastResult(
            mean=float(mean), model=self.name,
            extra={"w_hard": round(w, 3), "soft": soft_hat, "hard": hard_hat,
                   "n_soft": len(soft_preds)},
        )
