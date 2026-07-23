"""Gradient-boosted MIDAS: a non-linear machine-learning nowcaster.

Same mixed-frequency design as :class:`~MIDAS.sparse_midas.SparseMIDASNowcaster` (each monthly
predictor's high-frequency lags projected onto a Legendre dictionary, plus a target lag, with the
*direct* multi-step approach for ``h > 0``), but the map from features to the quarterly target is a
**histogram gradient boosting** regressor (:class:`sklearn.ensemble.HistGradientBoostingRegressor`)
instead of a sparse-group LASSO. This lets the model capture non-linearities and interactions the
linear models cannot, handles the ragged edge natively (trees split on missing values), and, with
``quantile`` set, emits quantile forecasts for density work.

Two honest caveats, stated up front:

* **Trees cannot extrapolate** beyond the training target range, so this is naturally immune to the
  linear blow-up the LASSO needs a stability guard for, but it can never predict a new record, a real
  limitation at turning points.
* With only ~80 to 125 quarterly observations, a boosting model must be **well regularised**
  (shallow trees, few leaves, modest iterations). Do not expect it to beat the sparse linear model on
  point RMSE, a robust small-sample result in macro; its value is non-linearity, interactions, native
  missing-data handling, and density (pair several ``quantile`` instances for a fan chart).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .align import stack_high_freq_lags
from .base import BaseNowcaster, InformationSet, NowcastResult
from .midas import legendre_dictionary


class BoostedMIDASNowcaster(BaseNowcaster):
    """Histogram gradient boosting on a Legendre-dictionary MIDAS design.

    Parameters
    ----------
    monthly_vars:
        Predictors to use; ``None`` uses the whole monthly panel (the ML use-case).
    n_lags, degree:
        MIDAS lag window and Legendre polynomial degree (the dictionary is ``degree + 1`` terms
        per predictor).
    quantile:
        ``None`` for a mean (squared-error) forecast, or a level in ``(0, 1)`` for a quantile
        forecast (pinball loss). Fit several to build a predictive interval.
    max_depth, learning_rate, max_iter, min_samples_leaf, l2_regularization:
        Boosting hyper-parameters. Defaults are deliberately conservative for the small sample.
    covid_quarters:
        Named outlier quarters dropped from estimation (same convention as the sparse model).
    """

    def __init__(
        self,
        monthly_vars: Sequence[str] | None = None,
        *,
        n_lags: int = 12,
        degree: int = 2,
        quantile: float | None = None,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        min_samples_leaf: int = 15,
        l2_regularization: float = 1.0,
        covid_quarters: Sequence[str] | None = None,
        random_state: int = 0,
        name: str | None = None,
    ) -> None:
        self.monthly_vars = None if monthly_vars is None else list(monthly_vars)
        self.n_lags = n_lags
        self.degree = degree
        self.quantile = quantile
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.covid_quarters = [pd.Timestamp(q) for q in (covid_quarters or [])]
        self.random_state = random_state
        self._name = name or ("GB-MIDAS" + (f"[q{quantile}]" if quantile is not None else ""))
        self._cached_key = None
        self._fit_state: dict | None = None

    # -------------------------------------------------------------- design construction
    def _dictionary_block(self, monthly: pd.DataFrame, index: pd.DatetimeIndex, var: str) -> np.ndarray:
        lags = stack_high_freq_lags(monthly[var].ffill(), index, n_lags=self.n_lags, end_month=3)
        return lags.to_numpy() @ legendre_dictionary(self.n_lags, self.degree)

    def _build_design(self, info: InformationSet, index: pd.DatetimeIndex):
        m_vars = self.monthly_vars or list(info.monthly.columns)
        blocks = [self._dictionary_block(info.monthly, index, v) for v in m_vars]
        Z = np.column_stack(blocks) if blocks else np.zeros((len(index), 0))
        return Z, m_vars

    def _feature_quarter(self, info: InformationSet) -> tuple[pd.Timestamp, int]:
        tq = pd.Period(pd.Timestamp(info.target_period), freq="Q")
        obs = info.monthly.dropna(how="all")
        last_q = pd.Period(obs.index[-1], freq="Q") if len(obs) else tq
        fq = min(tq, last_q)
        return pd.Timestamp(fq.end_time).normalize().replace(day=1), int((tq - fq).n)

    # ------------------------------------------------------------------- fit / nowcast
    def fit(self, info: InformationSet) -> "BoostedMIDASNowcaster":
        _, h = self._feature_quarter(info)
        key = (info.origin, h)
        if self._cached_key != key or self._fit_state is None:
            self._estimate(info, info.target, info.quarterly.index, h)
            self._cached_key = key
        self._target = info.target
        return self

    def _estimate(self, info: InformationSet, target: str, q_index: pd.DatetimeIndex, h: int) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        y_all = info.quarterly[target]
        Z, m_vars = self._build_design(info, q_index)
        # direct h-step training rows (same masking as the sparse model -> no look-ahead)
        y_shift = y_all.shift(-h)
        train = y_shift.notna().to_numpy() & y_all.shift(1).notna().to_numpy()
        if self.covid_quarters:
            tgt_q = pd.PeriodIndex(q_index, freq="Q") + h
            covid_p = [pd.Period(pd.Timestamp(q), freq="Q") for q in self.covid_quarters]
            train &= ~np.isin(tgt_q, covid_p)

        # trees are scale-invariant and split on NaN natively, so no standardisation or imputation:
        # the target lag plus the Legendre dictionary go in as-is.
        X = np.column_stack([y_all.shift(1).to_numpy(), Z])
        params = dict(
            loss="quantile" if self.quantile is not None else "squared_error",
            max_depth=self.max_depth, learning_rate=self.learning_rate, max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
            random_state=self.random_state, early_stopping=False,
        )
        if self.quantile is not None:
            params["quantile"] = self.quantile
        model = HistGradientBoostingRegressor(**params).fit(X[train], y_shift.to_numpy()[train])
        self._fit_state = dict(model=model, m_vars=m_vars, horizon=h, n_train=int(train.sum()))

    def nowcast(self, info: InformationSet) -> NowcastResult:
        st = self._fit_state
        if st is None:
            raise ValueError("fit before nowcast")
        fq_ts, h = self._feature_quarter(info)
        row = [self._dictionary_block(info.monthly, pd.DatetimeIndex([fq_ts]), v)[0] for v in st["m_vars"]]
        z = np.concatenate(row) if row else np.zeros(0)
        y_all = info.quarterly[info.target]
        prev_ts = pd.Timestamp((pd.Period(pd.Timestamp(fq_ts), freq="Q") - 1).end_time).normalize().replace(day=1)
        y_lag = y_all.get(prev_ts, np.nan)
        if not np.isfinite(y_lag):
            obs = y_all.dropna()
            y_lag = float(obs.iloc[-1]) if len(obs) else 0.0
        x = np.concatenate([[float(y_lag)], z])[None, :]
        return NowcastResult(mean=float(st["model"].predict(x)[0]), model=self.name,
                             extra={"horizon": h, "quantile": self.quantile, "n_train": st["n_train"]})
