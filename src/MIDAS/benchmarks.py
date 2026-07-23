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


def _mask_period_window(
    series: pd.Series,
    *,
    window: tuple[str, str] | None,
    freq: str,
) -> pd.Series:
    """Set observations inside ``window`` to NaN, keeping the index unchanged."""

    if window is None:
        return series
    out = series.copy()
    lo, hi = window
    pidx = out.index.to_period(freq)
    mask = (pidx >= pd.Period(lo, freq=freq)) & (pidx <= pd.Period(hi, freq=freq))
    out.loc[mask] = np.nan
    return out


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
    exclude_quarter_window: tuple[str, str] | None = None
    _name: str = "AR"

    def fit(self, info: InformationSet) -> "ARNowcaster":
        y = _observed_target(info)
        y = _mask_period_window(y, window=self.exclude_quarter_window, freq="Q").dropna()
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


def _principal_components(monthly: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """First ``n_components`` principal components of a (ragged) monthly panel.

    Each series is forward-filled to its last release, standardised, and its remaining
    gaps (not-yet-started series) mean-imputed to zero, before PCA. Standardisation and
    the PCA rotation are estimated on the rows that carry a genuine (pre-ffill) release
    only, then every row is projected: this stops the forward-filled tail of
    not-yet-released months (each row a copy of the last release) from dominating the
    moments and rotating the components. It uses only the data in ``monthly`` (already
    masked to the origin), so it adds no look-ahead. This lets a small-predictor model
    use the *whole* panel through a few common components, one way to feed a large panel to
    a bridge or a MIDAS model (pooling single-indicator models is another; the Bank of
    England's Bridge and P-MIDAS pool rather than take factors).
    """
    from sklearn.decomposition import PCA

    filled = monthly.ffill()
    real = monthly.dropna(how="all")  # rows with at least one genuine release
    fit_rows = (filled.index <= real.index[-1]) if len(real) else np.ones(len(filled), bool)
    mu = filled.loc[fit_rows].mean()
    sd = filled.loc[fit_rows].std(ddof=0).replace(0.0, 1.0)
    Z = ((filled - mu) / sd).to_numpy()
    Z = np.where(np.isfinite(Z), Z, 0.0)
    k = int(min(n_components, Z.shape[1]))
    pca = PCA(n_components=k).fit(Z[fit_rows])
    pcs = pca.transform(Z)
    return pd.DataFrame(pcs, index=monthly.index, columns=[f"PC{i}" for i in range(k)])


def _effective_monthly(info: InformationSet, indicators, n_components):
    """Predictor frame and names: the named indicators, or the panel's PCs."""
    if n_components:
        pcs = _principal_components(info.monthly, n_components)
        return pcs, list(pcs.columns)
    return info.monthly, list(indicators)


@dataclass
class BridgeNowcaster(BaseNowcaster):
    """OLS bridge: target on its lag and quarterly-aggregated monthly indicators.

    Indicators are averaged over their released months in each quarter
    (``align_monthly_to_quarter``). Indicators with no released month in the target
    quarter are dropped for that origin. The autoregressive term uses the most recent
    observed target value.

    Set ``n_components`` to run a **factor bridge**: the predictors become the first
    ``n_components`` principal components of the full monthly panel, so the model uses
    every series without the rank problems of OLS on a wide indicator set.
    """

    indicators: Sequence[str] = field(default_factory=list)
    n_components: int | None = None
    min_obs: int = 16
    _name: str = "Bridge"

    def fit(self, info: InformationSet) -> "BridgeNowcaster":
        self._info = info
        target = info.target
        q = info.quarterly[target]
        monthly, indicators = _effective_monthly(info, self.indicators, self.n_components)
        self._indicators = indicators
        aggs = align_monthly_to_quarter(monthly[indicators], method="mean")
        # align_monthly_to_quarter emits quarter-END dates (e.g. 2020-03-31); the
        # panel dates quarters at the first day of the end month (2020-03-01). Map
        # onto the panel convention before reindexing, else every cell is NaN.
        aggs.index = aggs.index.to_period("M").to_timestamp()
        aggs = aggs.reindex(info.quarterly.index)
        design = pd.DataFrame({"y": q, "y_lag1": q.shift(1)}, index=info.quarterly.index)
        design = pd.concat([design, aggs], axis=1)
        self._cols = ["y_lag1"] + list(indicators)
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

    Reuses :class:`~MIDAS.midas.BetaMIDASRegressor`. The within-quarter anchor month is
    chosen as the latest month of the target quarter whose indicators are all released at
    the origin (falling back to the previous quarter), so the model exploits as much of
    the current quarter as the data flow allows.

    Set ``n_components`` to run **Factor-MIDAS** (Marcellino and Schumacher, 2010): the
    predictors become the first ``n_components`` principal components of the full monthly
    panel, so restricted MIDAS can use the whole panel without an unstable dozens-of-
    indicators NLS. This is a factor approach; it is **not** the Bank of England's
    "P-MIDAS", which is a *pooled* MIDAS: a symmetric combination of single-indicator
    MIDAS regressions (Anesti et al. 2017), a different benchmark.
    """

    indicators: Sequence[str] = field(default_factory=list)
    n_lags: int = 12
    use_target_lag: bool = True
    n_components: int | None = None
    min_obs: int = 20
    _name: str = "ADL-MIDAS"

    @staticmethod
    def _pick_end_month(monthly: pd.DataFrame, indicators, quarter: pd.Timestamp) -> int:
        for E in (3, 2, 1):
            anchor = quarter + pd.DateOffset(months=E - 3)  # first day of month E
            if anchor in monthly.index and all(pd.notna(monthly.loc[anchor, ind]) for ind in indicators):
                return E
        return 0  # previous-quarter anchor (L1Q)

    def fit(self, info: InformationSet) -> "ADLMIDASNowcaster":
        target = info.target
        q_index = info.quarterly.index
        monthly, indicators = _effective_monthly(info, self.indicators, self.n_components)
        self._indicators = indicators
        self._E = self._pick_end_month(monthly, indicators, info.target_period)
        frame = stack_midas_features(monthly, q_index, indicators, n_lags=self.n_lags, end_month=self._E)
        frame[target] = info.quarterly[target]
        low_freq: list[str] = []
        if self.use_target_lag:
            frame["y_lag1"] = info.quarterly[target].shift(1)
            low_freq = ["y_lag1"]
        frame["date"] = q_index
        self._frame = frame
        self._low_freq = low_freq
        train = frame.loc[frame.index < info.target_period]
        needed = [target] + [f"{v}_L{i}" for v in indicators for i in range(self.n_lags)] + low_freq
        train = train.dropna(subset=needed)
        if len(train) < self.min_obs:
            self._reg = None
            return self
        self._reg = BetaMIDASRegressor(
            monthly_vars=indicators, n_lags=self.n_lags, low_freq_features=low_freq
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
        lag_cols = [f"{v}_L{i}" for v in self._indicators for i in range(self.n_lags)]
        if row[lag_cols].isna().any(axis=None) or row[self._low_freq].isna().any(axis=None):
            return NowcastResult(mean=float("nan"), model=self.name)
        pred = float(np.asarray(self._reg.predict_frame(row)).ravel()[0])
        return NowcastResult(mean=pred, model=self.name, extra={"end_month": self._E})


# --------------------------------------------------- faithful Bank of England benchmarks
# These reproduce the horse-race competitors in Moreira (2025), "Nowcasting GDP at the
# Bank of England: a Staggered-Combination MIDAS approach". Crucially, BoE's Bridge and
# P-MIDAS use NO principal components: each is a POOL of single-indicator regressions.
# The DFM competitor (a state-space factor model) is DFMNowcaster in dfm.py.

def _quarter_months(target_period: pd.Timestamp) -> list[pd.Timestamp]:
    """The three month-start timestamps of the quarter containing ``target_period``."""
    q = pd.Period(pd.Timestamp(target_period), freq="Q")
    m0 = q.asfreq("M", how="start")
    return [pd.Timestamp((m0 + k).start_time).normalize() for k in range(3)]


def _iterate_monthly_ar(released: pd.Series, beta: np.ndarray, p: int, until: pd.Timestamp) -> pd.Series:
    """Extend a released monthly series forward to ``until`` by an AR(p) recursion.

    Released months are kept as they are; each later month is filled from the previous
    ``p`` values (released or already iterated). Returns a monthly series on a contiguous
    month-start index from the first release to ``max(last release, until)``.
    """
    end = max(released.index[-1], pd.Timestamp(until))
    idx = pd.date_range(released.index[0], end, freq="MS")
    arr = released.reindex(idx).to_numpy()
    for i in range(len(idx)):
        if np.isfinite(arr[i]):
            continue
        if i < p or not np.all(np.isfinite(arr[i - p:i])):
            continue
        lags = np.array([arr[i - k] for k in range(1, p + 1)])
        arr[i] = beta[0] + beta[1:] @ lags
    return pd.Series(arr, index=idx)


def _monthly_log_diff(levels: pd.Series, scale: float = 100.0) -> pd.Series:
    """Monthly log growth from a positive level series."""

    s = pd.to_numeric(levels, errors="coerce")
    if (s.dropna() <= 0).any():
        raise ValueError("monthly GDP levels must be strictly positive")
    return scale * (np.log(s) - np.log(s.shift(1)))


def _quarter_level_from_monthly(levels: pd.Series, quarter: pd.Timestamp) -> float:
    """Quarterly proxy level as the arithmetic mean of the quarter's three months."""

    vals = [levels.get(ts, np.nan) for ts in _quarter_months(quarter)]
    return float(np.mean(vals)) if np.all(np.isfinite(vals)) else float("nan")


def _quarter_target_from_monthly_levels(
    levels: pd.Series,
    target_period: pd.Timestamp,
    target_transform: str,
) -> float:
    """Convert a monthly level path into the quarterly target object's raw scale."""

    target_period = pd.Timestamp(target_period)
    q_level = _quarter_level_from_monthly(levels, target_period)
    if not np.isfinite(q_level):
        return float("nan")
    q = pd.Period(target_period, freq="Q")
    if target_transform == "qoq_ann":
        prev_q = pd.Timestamp((q - 1).end_time).normalize().replace(day=1)
        prev_level = _quarter_level_from_monthly(levels, prev_q)
        if not np.isfinite(prev_level) or prev_level <= 0.0 or q_level <= 0.0:
            return float("nan")
        return float(400.0 * (np.log(q_level) - np.log(prev_level)))
    if target_transform == "yoy":
        lag4_q = pd.Timestamp((q - 4).end_time).normalize().replace(day=1)
        lag4_level = _quarter_level_from_monthly(levels, lag4_q)
        if not np.isfinite(lag4_level) or lag4_level <= 0.0 or q_level <= 0.0:
            return float("nan")
        return float(100.0 * (np.log(q_level) - np.log(lag4_level)))
    raise ValueError(f"unsupported quarterly target transform: {target_transform!r}")


@dataclass
class MonthlyLevelARNowcaster(BaseNowcaster):
    """Monthly GDP level proxy AR benchmark, reported on the quarterly target scale."""

    monthly_gdp: str = "pbim_level"
    target_transform: str = "qoq_ann"
    order: int = 2
    min_obs: int = 36
    exclude_month_window: tuple[str, str] | None = None
    _name: str = "M-AR"

    # The benchmark must see the unstandardized monthly level path.
    requires_unstandardized_info: bool = True

    def fit(self, info: InformationSet) -> "MonthlyLevelARNowcaster":
        levels = pd.to_numeric(info.monthly[self.monthly_gdp], errors="coerce").dropna()
        self._beta, self._p = None, 0
        if len(levels) < self.min_obs + 1:
            return self
        growth = _monthly_log_diff(levels).dropna()
        growth = _mask_period_window(growth, window=self.exclude_month_window, freq="M").dropna()
        if len(growth) < self.min_obs:
            return self
        yv = growth.to_numpy()
        p = max(1, min(self.order, len(yv) - 2))
        beta = None
        while p >= 1:
            X, yt = _ar_design(yv, p)
            beta, _ = _ols(X, yt)
            if _is_stationary(beta[1:]):
                break
            p -= 1
        if beta is not None and _is_stationary(beta[1:]):
            self._beta, self._p = beta, p
        return self

    def _extend_levels(self, info: InformationSet) -> pd.Series | None:
        levels = pd.to_numeric(info.monthly[self.monthly_gdp], errors="coerce")
        released = levels.dropna()
        if released.empty:
            return None
        end = max(released.index[-1], _quarter_months(info.target_period)[-1])
        idx = pd.date_range(released.index[0], end, freq="MS")
        lvl = levels.reindex(idx).astype(float).copy()
        growth = _monthly_log_diff(levels).reindex(idx).astype(float)

        last_growth = growth.dropna()
        fallback_growth = float(last_growth.iloc[-1]) if not last_growth.empty else 0.0

        for i in range(1, len(idx)):
            if np.isfinite(lvl.iloc[i]):
                if not np.isfinite(growth.iloc[i]) and np.isfinite(lvl.iloc[i - 1]) and lvl.iloc[i - 1] > 0.0:
                    growth.iloc[i] = 100.0 * (np.log(lvl.iloc[i]) - np.log(lvl.iloc[i - 1]))
                continue
            prev = lvl.iloc[i - 1]
            if not np.isfinite(prev) or prev <= 0.0:
                continue
            if self._beta is None or self._p == 0:
                ghat = growth.iloc[i - 1] if np.isfinite(growth.iloc[i - 1]) else fallback_growth
            else:
                if i < self._p:
                    continue
                lags = np.array([growth.iloc[i - k] for k in range(1, self._p + 1)])
                if not np.all(np.isfinite(lags)):
                    continue
                ghat = float(self._beta[0] + self._beta[1:] @ lags)
            growth.iloc[i] = ghat
            lvl.iloc[i] = float(prev * np.exp(ghat / 100.0))
        return lvl

    def nowcast(self, info: InformationSet) -> NowcastResult:
        levels = self._extend_levels(info)
        if levels is None:
            return NowcastResult(mean=float("nan"), model=self.name)
        pred = _quarter_target_from_monthly_levels(levels, info.target_period, self.target_transform)
        return NowcastResult(mean=float(pred), model=self.name)


@dataclass
class PooledLevelBridgeNowcaster(BaseNowcaster):
    """Pooled monthly bridge on GDP levels, reported on the quarterly target scale."""

    monthly_gdp: str = "pbim_level"
    monthly_vars: Sequence[str] | None = None
    target_transform: str = "qoq_ann"
    gdp_lags: int = 2
    ind_ar_order: int = 2
    min_obs: int = 36
    exclude_month_window: tuple[str, str] | None = None
    _name: str = "Bridge"

    requires_unstandardized_info: bool = True

    def fit(self, info: InformationSet) -> "PooledLevelBridgeNowcaster":
        levels = pd.to_numeric(info.monthly[self.monthly_gdp], errors="coerce")
        growth = _monthly_log_diff(levels)
        growth = _mask_period_window(growth, window=self.exclude_month_window, freq="M")
        vars_ = [v for v in (self.monthly_vars or list(info.monthly.columns)) if v != self.monthly_gdp]
        self._members: list[dict] = []
        for v in vars_:
            mem = self._fit_member(info, growth, v)
            if mem is not None:
                self._members.append(mem)
        return self

    def _fit_member(self, info: InformationSet, growth: pd.Series, v: str) -> dict | None:
        L = self.gdp_lags
        x_series = pd.to_numeric(info.monthly[v], errors="coerce")
        x_series = _mask_period_window(x_series, window=self.exclude_month_window, freq="M")
        D = pd.DataFrame({"g": growth})
        for k in range(1, L + 1):
            D[f"g{k}"] = growth.shift(k)
        D["x"] = x_series
        D = D.dropna()
        if len(D) < self.min_obs:
            return None
        cols = [f"g{k}" for k in range(1, L + 1)] + ["x"]
        X = np.column_stack([np.ones(len(D))] + [D[c].to_numpy() for c in cols])
        beta, resid = _ols(X, D["g"].to_numpy())
        rmse_in = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.inf
        ar = None
        xv = x_series.dropna()
        if len(xv) >= self.min_obs:
            p = max(1, self.ind_ar_order)
            Xa, ya = _ar_design(xv.to_numpy(), p)
            ab, _ = _ols(Xa, ya)
            if _is_stationary(ab[1:]):
                ar = (ab, p)
        return {"v": v, "beta": beta, "rmse": rmse_in, "ar": ar, "x": x_series}

    def nowcast(self, info: InformationSet) -> NowcastResult:
        if not self._members:
            y = _observed_target(info)
            return NowcastResult(mean=(float(y.iloc[-1]) if not y.empty else float("nan")), model=self.name)
        levels = pd.to_numeric(info.monthly[self.monthly_gdp], errors="coerce")
        preds, weights = [], []
        for mem in self._members:
            qhat = self._predict_member(levels, mem, info.target_period)
            if np.isfinite(qhat):
                preds.append(qhat)
                weights.append(1.0 / max(mem["rmse"], 1e-6))
        if not preds:
            y = _observed_target(info)
            return NowcastResult(mean=(float(y.iloc[-1]) if not y.empty else float("nan")), model=self.name)
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return NowcastResult(mean=float(np.dot(w, preds)), model=self.name, extra={"n_members": len(preds)})

    def _predict_member(self, levels: pd.Series, mem: dict, target_period: pd.Timestamp) -> float:
        released = levels.dropna()
        if released.empty:
            return float("nan")
        months = _quarter_months(target_period)
        end = max(released.index[-1], months[-1])
        idx = pd.date_range(released.index[0], end, freq="MS")
        lvl = levels.reindex(idx).astype(float).copy()
        growth = _monthly_log_diff(levels).reindex(idx).astype(float)
        xf = self._extend_indicator(mem, idx)
        month_set = set(months)

        for i in range(1, len(idx)):
            if np.isfinite(lvl.iloc[i]):
                if not np.isfinite(growth.iloc[i]) and np.isfinite(lvl.iloc[i - 1]) and lvl.iloc[i - 1] > 0.0:
                    growth.iloc[i] = 100.0 * (np.log(lvl.iloc[i]) - np.log(lvl.iloc[i - 1]))
                continue
            prev = lvl.iloc[i - 1]
            if i < self.gdp_lags or not np.isfinite(prev) or prev <= 0.0:
                if idx[i] in month_set:
                    return float("nan")
                continue
            lags = np.array([growth.iloc[i - k] for k in range(1, self.gdp_lags + 1)])
            if not np.all(np.isfinite(lags)) or not np.isfinite(xf[i]):
                if idx[i] in month_set:
                    return float("nan")
                continue
            ghat = float(mem["beta"][0] + np.dot(mem["beta"][1:1 + self.gdp_lags], lags) + mem["beta"][1 + self.gdp_lags] * xf[i])
            growth.iloc[i] = ghat
            lvl.iloc[i] = float(prev * np.exp(ghat / 100.0))

        return _quarter_target_from_monthly_levels(lvl, target_period, self.target_transform)

    def _extend_indicator(self, mem: dict, idx: pd.DatetimeIndex) -> np.ndarray:
        arr = mem["x"].reindex(idx).to_numpy(dtype=float)
        ar = mem["ar"]
        if ar is None:
            return pd.Series(arr, index=idx).ffill().to_numpy()
        ab, p = ar
        for i in range(len(idx)):
            if np.isfinite(arr[i]):
                continue
            if i < p or not np.all(np.isfinite(arr[i - p:i])):
                if i > 0 and np.isfinite(arr[i - 1]):
                    arr[i] = arr[i - 1]
                continue
            lags = np.array([arr[i - k] for k in range(1, p + 1)])
            arr[i] = ab[0] + ab[1:] @ lags
        return arr


@dataclass
class MonthlyARNowcaster(BaseNowcaster):
    """Monthly AR(p) on monthly GDP, time-aggregated to the quarter (BoE 'M-AR').

    Moreira (2025, eq. 10): fit an AR(p) (p=2 by default, chosen there by OOS testing) to
    released monthly GDP growth, iterate it forward to complete the target quarter's three
    months, and average them to a quarterly nowcast. Peru's quarterly GDP growth is the mean
    of its three monthly growths (verified: corr 0.9996), so mean aggregation is exact here.
    """

    monthly_gdp: str = "g_pbim"
    order: int = 2
    min_obs: int = 24
    _name: str = "M-AR"

    def fit(self, info: InformationSet) -> "MonthlyARNowcaster":
        self._g = info.monthly[self.monthly_gdp]
        y = self._g.dropna()
        self._beta, self._p = None, 0
        if len(y) < self.min_obs:
            return self
        yv = y.to_numpy()
        p = max(1, min(self.order, len(yv) - 2))
        beta = None
        while p >= 1:  # step the order down until the fit is stationary (else iteration diverges)
            X, yt = _ar_design(yv, p)
            beta, _ = _ols(X, yt)
            if _is_stationary(beta[1:]):
                break
            p -= 1
        if beta is not None and _is_stationary(beta[1:]):
            self._beta, self._p = beta, p
        return self

    def nowcast(self, info: InformationSet) -> NowcastResult:
        g = info.monthly[self.monthly_gdp]
        released = g.dropna()
        if released.empty:
            return NowcastResult(mean=float("nan"), model=self.name)
        months = _quarter_months(info.target_period)
        if self._beta is None:  # no stable AR: use released quarter months, else the last value
            last = float(released.iloc[-1])
            vals = [g.get(mo, np.nan) for mo in months]
            vals = [float(v) if np.isfinite(v) else last for v in vals]
            return NowcastResult(mean=float(np.mean(vals)), model=self.name)
        series = _iterate_monthly_ar(released, self._beta, self._p, months[-1])
        vals = [series.get(mo, np.nan) for mo in months]
        if not np.all(np.isfinite(vals)):
            return NowcastResult(mean=float("nan"), model=self.name)
        return NowcastResult(mean=float(np.mean(vals)), model=self.name)


@dataclass
class PooledMIDASNowcaster(BaseNowcaster):
    """Pooled single-indicator MIDAS (BoE 'P-MIDAS', Anesti et al. 2017).

    Fits one restricted Beta-MIDAS regression per monthly indicator (reusing
    :class:`ADLMIDASNowcaster`, each with a single indicator plus a target lag) and pools
    the per-indicator nowcasts. The pool is **symmetric** (equal weights): this is the
    standard pooled-MIDAS benchmark the paper's SC-MIDAS is contrasted against ("P-MIDAS
    pools all predictions symmetrically"). No principal components, no hand-picking.

    Cost note: this refits one MIDAS NLS per indicator at every origin, so the release-cycle
    backtest with the full panel is expensive; cache it.
    """

    monthly_vars: Sequence[str] | None = None
    n_lags: int = 12
    use_target_lag: bool = True
    min_obs: int = 20
    _name: str = "P-MIDAS"

    def fit(self, info: InformationSet) -> "PooledMIDASNowcaster":
        vars_ = self.monthly_vars or list(info.monthly.columns)
        self._members: dict[str, ADLMIDASNowcaster] = {}
        for v in vars_:
            mem = ADLMIDASNowcaster(indicators=[v], n_lags=self.n_lags,
                                    use_target_lag=self.use_target_lag, min_obs=self.min_obs)
            try:
                mem.fit(info)
            except Exception:
                continue
            self._members[v] = mem
        return self

    def nowcast(self, info: InformationSet) -> NowcastResult:
        preds = []
        for mem in self._members.values():
            try:
                r = mem.nowcast(info)
            except Exception:
                continue
            if np.isfinite(r.mean):
                preds.append(float(r.mean))
        if not preds:
            y = _observed_target(info)
            return NowcastResult(mean=(float(y.iloc[-1]) if not y.empty else float("nan")), model=self.name)
        return NowcastResult(mean=float(np.mean(preds)), model=self.name, extra={"n_members": len(preds)})


@dataclass
class PooledBridgeNowcaster(BaseNowcaster):
    """Pooled single-indicator monthly bridge on monthly GDP (BoE 'Bridge', Bell et al. 2014).

    For each monthly indicator: (i) a monthly AR(``ind_ar_order``) forecasts the indicator's
    not-yet-released months (eq. 12); (ii) an ARDL regresses monthly GDP on its own
    ``gdp_lags`` lags and that single indicator (eq. 13); the fitted ARDL predicts the target
    quarter's monthly GDP, mean-aggregated to a quarterly figure. The per-indicator quarterly
    predictions are pooled by **inverse in-sample RMSE** (a single-origin proxy for the paper's
    OOS-RMSE pooling). No principal components. Monthly GDP itself is excluded as a predictor.
    """

    monthly_gdp: str = "g_pbim"
    monthly_vars: Sequence[str] | None = None
    gdp_lags: int = 2
    ind_ar_order: int = 2
    min_obs: int = 30
    _name: str = "Bridge"

    def fit(self, info: InformationSet) -> "PooledBridgeNowcaster":
        g = info.monthly[self.monthly_gdp]
        vars_ = [v for v in (self.monthly_vars or list(info.monthly.columns)) if v != self.monthly_gdp]
        self._members: list[dict] = []
        for v in vars_:
            mem = self._fit_member(info, g, v)
            if mem is not None:
                self._members.append(mem)
        return self

    def _fit_member(self, info: InformationSet, g: pd.Series, v: str) -> dict | None:
        L = self.gdp_lags
        D = pd.DataFrame({"g": g})
        for k in range(1, L + 1):
            D[f"g{k}"] = g.shift(k)
        D["x"] = info.monthly[v]
        D = D.dropna()
        if len(D) < self.min_obs:
            return None
        cols = [f"g{k}" for k in range(1, L + 1)] + ["x"]
        X = np.column_stack([np.ones(len(D))] + [D[c].to_numpy() for c in cols])
        beta, resid = _ols(X, D["g"].to_numpy())
        rmse_in = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.inf
        ar = None  # AR(p) to forecast the indicator's ragged edge (eq. 12)
        xv = info.monthly[v].dropna()
        if len(xv) >= self.min_obs:
            p = max(1, self.ind_ar_order)
            Xa, ya = _ar_design(xv.to_numpy(), p)
            ab, _ = _ols(Xa, ya)
            if _is_stationary(ab[1:]):
                ar = (ab, p)
        return {"v": v, "beta": beta, "rmse": rmse_in, "ar": ar, "x": info.monthly[v]}

    def nowcast(self, info: InformationSet) -> NowcastResult:
        if not self._members:
            y = _observed_target(info)
            return NowcastResult(mean=(float(y.iloc[-1]) if not y.empty else float("nan")), model=self.name)
        g = info.monthly[self.monthly_gdp]
        months = _quarter_months(info.target_period)
        preds, weights = [], []
        for mem in self._members:
            q = self._predict_member(g, mem, months)
            if np.isfinite(q):
                preds.append(q)
                weights.append(1.0 / max(mem["rmse"], 1e-6))
        if not preds:
            y = _observed_target(info)
            return NowcastResult(mean=(float(y.iloc[-1]) if not y.empty else float("nan")), model=self.name)
        w = np.asarray(weights)
        w = w / w.sum()
        return NowcastResult(mean=float(np.dot(w, preds)), model=self.name, extra={"n_members": len(preds)})

    def _predict_member(self, g: pd.Series, mem: dict, months: list[pd.Timestamp]) -> float:
        L, beta = self.gdp_lags, mem["beta"]
        released = g.dropna()
        if released.empty:
            return float("nan")
        end = max(released.index[-1], months[-1])
        idx = pd.date_range(released.index[0], end, freq="MS")
        gf = g.reindex(idx).to_numpy()
        xf = self._extend_indicator(mem, idx)
        month_set = set(months)
        for i in range(len(idx)):
            if np.isfinite(gf[i]):
                continue
            lags = [gf[i - k] for k in range(1, L + 1)] if i >= L else [np.nan]
            if not np.all(np.isfinite(lags)) or not np.isfinite(xf[i]):
                if idx[i] in month_set:
                    return float("nan")  # a needed month cannot be formed: drop this member
                continue
            gf[i] = beta[0] + np.dot(beta[1:1 + L], lags) + beta[1 + L] * xf[i]
        pos = {ts: k for k, ts in enumerate(idx)}
        vals = [gf[pos[mo]] if mo in pos else np.nan for mo in months]
        return float(np.mean(vals)) if np.all(np.isfinite(vals)) else float("nan")

    def _extend_indicator(self, mem: dict, idx: pd.DatetimeIndex) -> np.ndarray:
        arr = mem["x"].reindex(idx).to_numpy()
        ar = mem["ar"]
        if ar is None:  # no stable indicator AR: hold the last value flat
            return pd.Series(arr, index=idx).ffill().to_numpy()
        ab, p = ar
        for i in range(len(idx)):
            if np.isfinite(arr[i]):
                continue
            if i < p or not np.all(np.isfinite(arr[i - p:i])):
                if i > 0 and np.isfinite(arr[i - 1]):
                    arr[i] = arr[i - 1]  # fall back to a flat carry when lags are missing
                continue
            lags = np.array([arr[i - k] for k in range(1, p + 1)])
            arr[i] = ab[0] + ab[1:] @ lags
        return arr
