"""Benchmark nowcasters.

The yardsticks every richer model (DFM, sparse MIDAS) must beat, all speaking the
:class:`~MIDAS.base.BaseNowcaster` contract so the harness drives them uniformly:

* :class:`RandomWalkNowcaster`   - last observed target value (no-change).
* :class:`HistoricalMeanNowcaster` - expanding-window mean of the target.
* :class:`ARNowcaster`           - AR(p) with BIC order choice, iterated forward
  to the target quarter when the most recent quarters are not yet released.
* :class:`BridgeNowcaster`       - OLS of the target on its lag and
  quarterly-aggregated monthly indicators (available months only).
* :class:`ADLMIDASNowcaster`     - restricted Beta-MIDAS on monthly indicators
  (plus an optional target lag), reusing :class:`~MIDAS.midas.BetaMIDASRegressor`
  and choosing the within-quarter anchor month from what is released at the origin.

Every model reads only the masked panels in the :class:`InformationSet`, so none
can see the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .align import align_monthly_to_quarter
from .base import BaseNowcaster, InformationSet, NowcastResult
from .midas import BetaMIDASRegressor, stack_midas_features


# --------------------------------------------------------------------- helpers
def _observed_target(info: InformationSet) -> pd.Series:
    """Target quarters released at the origin (contiguous tail, NaNs dropped)."""
    return info.quarterly[info.target].dropna()


def _steps_ahead(last: pd.Timestamp, target_period: pd.Timestamp) -> int:
    """Quarterly steps from the ``last`` observed quarter to ``target_period``.

    Uses period arithmetic rather than an index lookup so the target may lie
    beyond the observed sample (multi-horizon forecasting).
    """
    a = pd.Period(pd.Timestamp(last), freq="Q")
    b = pd.Period(pd.Timestamp(target_period), freq="Q")
    return int((b - a).n)


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def _ar_design(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix [const, y_{t-1}, ..., y_{t-p}] and aligned target."""
    n = len(y)
    X = np.ones((n - p, p + 1))
    for k in range(1, p + 1):
        X[:, k] = y[p - k : n - k]
    return X, y[p:]


def _is_stationary(ar_coefs: np.ndarray, tol: float = 0.999) -> bool:
    """True if the AR companion matrix has every eigenvalue inside the unit circle.

    An explosive companion makes the iterated multi-step forecast diverge, which is
    exactly how an over-parameterised AR ruins a nowcast several quarters ahead.
    """
    p = len(ar_coefs)
    if p == 0:
        return True
    comp = np.zeros((p, p))
    comp[0, :] = ar_coefs
    if p > 1:
        comp[1:, :-1] = np.eye(p - 1)
    return float(np.max(np.abs(np.linalg.eigvals(comp)))) < tol


def _ar_bic_order(y: np.ndarray, pmax: int) -> int:
    """AR order minimising BIC over 1..pmax, scored on a *common* sample.

    Every candidate order must be evaluated on the same observations (the last
    ``n - pmax``). Scoring each order on its own ``n - p`` rows would let a larger
    ``p`` drop early observations, mechanically shrinking the SSR and biasing the
    criterion toward the maximum order.
    """
    n = len(y)
    pmax = max(1, min(pmax, n - 3))
    best_p, best_bic = 1, np.inf
    for p in range(1, pmax + 1):
        X, yt = _ar_design(y, p)
        drop = pmax - p  # align all orders on the common effective sample
        X, yt = X[drop:], yt[drop:]
        if len(yt) <= pmax + 2:
            continue
        _, resid = _ols(X, yt)
        ssr = float(resid @ resid)
        m = len(yt)
        bic = m * np.log(ssr / m + 1e-12) + (p + 1) * np.log(m)
        if bic < best_bic:
            best_p, best_bic = p, bic
    return best_p


# --------------------------------------------------------------------- models
@dataclass
class RandomWalkNowcaster(BaseNowcaster):
    """No-change: nowcast equals the last observed target value."""

    _name: str = "RW"

    def fit(self, info: InformationSet) -> "RandomWalkNowcaster":
        self._y = _observed_target(info)
        return self

    def nowcast(self, info: InformationSet) -> NowcastResult:
        if self._y.empty:
            return NowcastResult(mean=float("nan"), model=self.name)
        return NowcastResult(mean=float(self._y.iloc[-1]), model=self.name)


@dataclass
class HistoricalMeanNowcaster(BaseNowcaster):
    """Expanding-window mean of the observed target."""

    _name: str = "Mean"

    def fit(self, info: InformationSet) -> "HistoricalMeanNowcaster":
        self._y = _observed_target(info)
        return self

    def nowcast(self, info: InformationSet) -> NowcastResult:
        if self._y.empty:
            return NowcastResult(mean=float("nan"), model=self.name)
        return NowcastResult(mean=float(self._y.mean()), std=float(self._y.std(ddof=1)), model=self.name)


@dataclass
class ARNowcaster(BaseNowcaster):
    """AR(p) on the target, iterated forward to the target quarter.

    ``order`` is an integer or ``"bic"`` (chosen over 1..``pmax``). When the most
    recent quarters are not yet released at the origin, the model iterates its own
    forecasts forward from the last observed quarter to ``target_period``.
    """

    order: int | str = "bic"
    pmax: int = 4
    min_obs: int = 12
    _name: str = "AR"

    def fit(self, info: InformationSet) -> "ARNowcaster":
        y = _observed_target(info)
        self._index = info.quarterly.index
        self._y = y
        if len(y) < self.min_obs:
            self._beta = None
            return self
        yv = y.to_numpy()
        p = _ar_bic_order(yv, self.pmax) if self.order == "bic" else int(self.order)
        p = max(1, min(p, len(yv) - 2))
        # Step the order down until the fitted AR is stationary: an explosive fit
        # would diverge when iterated forward to the target quarter.
        beta = None
        while p >= 1:
            X, yt = _ar_design(yv, p)
            beta, _ = _ols(X, yt)
            if _is_stationary(beta[1:]):
                break
            p -= 1
        if beta is None or not _is_stationary(beta[1:]):
            self._beta = None  # nothing stable: fall back to no-change
            return self
        self._beta, self._p = beta, p
        return self

    def nowcast(self, info: InformationSet) -> NowcastResult:
        if self._y.empty:
            return NowcastResult(mean=float("nan"), model=self.name)
        if self._beta is None:  # too little history: fall back to no-change
            return NowcastResult(mean=float(self._y.iloc[-1]), model=self.name)
        last = self._y.index[-1]
        h = _steps_ahead(last, info.target_period)
        if h <= 0:  # already observed at this origin
            if info.target_period in self._y.index:
                return NowcastResult(mean=float(self._y.loc[info.target_period]), model=self.name)
            return NowcastResult(mean=float(self._y.iloc[-1]), model=self.name)
        hist = list(self._y.to_numpy()[-self._p :][::-1])  # most recent first
        pred = float("nan")
        for _ in range(h):
            pred = float(self._beta[0] + self._beta[1:] @ np.asarray(hist[: self._p]))
            hist.insert(0, pred)
        return NowcastResult(mean=pred, model=self.name)


@dataclass
class BridgeNowcaster(BaseNowcaster):
    """OLS bridge: target on its lag and quarterly-aggregated monthly indicators.

    Indicators are averaged over their released months in each quarter
    (``align_monthly_to_quarter``). Indicators with no released month in the
    target quarter are dropped for that origin. The autoregressive term uses the
    most recent observed target value.
    """

    indicators: Sequence[str] = field(default_factory=list)
    min_obs: int = 16
    _name: str = "Bridge"

    def fit(self, info: InformationSet) -> "BridgeNowcaster":
        self._info = info
        target = info.target
        q = info.quarterly[target]
        aggs = align_monthly_to_quarter(info.monthly[list(self.indicators)], method="mean")
        # align_monthly_to_quarter emits quarter-END dates (e.g. 2020-03-31); the
        # panel dates quarters at the first day of the end month (2020-03-01). Map
        # onto the panel convention before reindexing, else every cell is NaN.
        aggs.index = aggs.index.to_period("M").to_timestamp()
        aggs = aggs.reindex(info.quarterly.index)
        design = pd.DataFrame({"y": q, "y_lag1": q.shift(1)}, index=info.quarterly.index)
        design = pd.concat([design, aggs], axis=1)
        self._cols = ["y_lag1"] + [c for c in self.indicators]
        train = design.loc[design.index < info.target_period].dropna(subset=["y"] + self._cols)
        self._usable = [c for c in self._cols if c in train.columns and train[c].notna().all()]
        if len(train) < self.min_obs or not self._usable:
            self._beta = None
            return self
        X = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in self._usable])
        self._beta, _ = _ols(X, train["y"].to_numpy())
        self._design = design
        return self

    def nowcast(self, info: InformationSet) -> NowcastResult:
        if self._beta is None:
            y = _observed_target(info)
            return NowcastResult(mean=(float(y.iloc[-1]) if not y.empty else float("nan")), model=self.name)
        row = self._design.loc[info.target_period]
        y_obs = _observed_target(info)
        feats = []
        for c in self._usable:
            v = row[c]
            if c == "y_lag1" and pd.isna(v):
                v = float(y_obs.iloc[-1]) if not y_obs.empty else np.nan
            feats.append(v)
        if any(pd.isna(feats)):
            return NowcastResult(mean=float("nan"), model=self.name)
        x = np.concatenate([[1.0], np.asarray(feats, dtype=float)])
        return NowcastResult(mean=float(self._beta @ x), model=self.name)


@dataclass
class ADLMIDASNowcaster(BaseNowcaster):
    """Restricted Beta-MIDAS on monthly indicators with an optional target lag.

    Reuses :class:`~MIDAS.midas.BetaMIDASRegressor`. The within-quarter anchor
    month is chosen as the latest month of the target quarter whose indicators are
    all released at the origin (falling back to the previous quarter), so the model
    exploits as much of the current quarter as the data flow allows.
    """

    indicators: Sequence[str] = field(default_factory=list)
    n_lags: int = 12
    use_target_lag: bool = True
    min_obs: int = 20
    _name: str = "ADL-MIDAS"

    def _pick_end_month(self, monthly: pd.DataFrame, quarter: pd.Timestamp) -> int:
        for E in (3, 2, 1):
            anchor = quarter + pd.DateOffset(months=E - 3)  # first day of month E
            if anchor in monthly.index and all(
                pd.notna(monthly.loc[anchor, ind]) for ind in self.indicators
            ):
                return E
        return 0  # previous-quarter anchor (L1Q)

    def fit(self, info: InformationSet) -> "ADLMIDASNowcaster":
        target = info.target
        q_index = info.quarterly.index
        self._E = self._pick_end_month(info.monthly, info.target_period)
        frame = stack_midas_features(
            info.monthly, q_index, list(self.indicators), n_lags=self.n_lags, end_month=self._E
        )
        frame[target] = info.quarterly[target]
        low_freq: list[str] = []
        if self.use_target_lag:
            frame["y_lag1"] = info.quarterly[target].shift(1)
            low_freq = ["y_lag1"]
        frame["date"] = q_index
        self._frame = frame
        self._low_freq = low_freq
        train = frame.loc[frame.index < info.target_period]
        needed = [target] + [f"{v}_L{i}" for v in self.indicators for i in range(self.n_lags)] + low_freq
        train = train.dropna(subset=needed)
        if len(train) < self.min_obs:
            self._reg = None
            return self
        self._reg = BetaMIDASRegressor(
            monthly_vars=list(self.indicators), n_lags=self.n_lags, low_freq_features=low_freq
        )
        try:
            self._reg.fit_frame(train, target=target)
        except Exception:
            self._reg = None
        return self

    def nowcast(self, info: InformationSet) -> NowcastResult:
        if self._reg is None:
            y = _observed_target(info)
            return NowcastResult(mean=(float(y.iloc[-1]) if not y.empty else float("nan")), model=self.name)
        row = self._frame.loc[[info.target_period]].copy()
        if self.use_target_lag and pd.isna(row["y_lag1"].iloc[0]):
            y = _observed_target(info)
            row["y_lag1"] = float(y.iloc[-1]) if not y.empty else np.nan
        lag_cols = [f"{v}_L{i}" for v in self.indicators for i in range(self.n_lags)]
        if row[lag_cols].isna().any(axis=None) or row[self._low_freq].isna().any(axis=None):
            return NowcastResult(mean=float("nan"), model=self.name)
        pred = float(np.asarray(self._reg.predict_frame(row)).ravel()[0])
        return NowcastResult(mean=pred, model=self.name, extra={"end_month": self._E})
