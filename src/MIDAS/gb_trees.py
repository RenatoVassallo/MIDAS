"""Gradient-boosted trees on interpretable indicator features.

A non-linear nowcaster whose features are the indicators themselves, not a
MIDAS basis. Each monthly predictor is aggregated to the quarter (mean of its
within-quarter months, ragged-edge aware), and a histogram gradient-boosting
regressor maps those quarterly features plus the target's own lag to the
quarterly target. Because one feature corresponds to one indicator, the
feature-importance and SHAP tables read directly in terms of economic
variables, which is the point of using this over
:class:`~MIDAS.boosted_midas.BoostedMIDASNowcaster` (whose SHAP lands on
Legendre basis terms).

Design choices for the small macro sample:

* **Interpretable features.** One quarterly value per indicator (plus the
  target AR(1) lag). SHAP maps one-to-one to indicators.
* **Regularised boosting.** Shallow trees, small learning rate, a leaf-size
  floor, and L2 shrinkage. Trees split on missing values, so the ragged edge
  needs no imputation, and they cannot extrapolate, so there is no linear
  blow-up to guard against (but also no new record can be predicted).
* **Direct multi-step.** As in the sparse and MIDAS models, ``h`` is inferred
  from how much of the target quarter is observed, and a separate fit is trained
  for that horizon.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .align import stack_high_freq_lags
from .base import BaseNowcaster, InformationSet, NowcastResult


class GBTreesNowcaster(BaseNowcaster):
    """Histogram gradient boosting on quarterly-aggregated indicator features.

    Parameters
    ----------
    monthly_vars:
        Indicators to use; ``None`` uses the whole monthly panel. Pass the
        screened leading indicators for a compact, interpretable model.
    n_months:
        Within-quarter months averaged into each indicator's quarterly feature.
    max_depth, learning_rate, max_iter, min_samples_leaf, l2_regularization,
    early_stopping:
        Boosting hyper-parameters, deliberately conservative for the small
        sample.
    covid_quarters:
        Named outlier quarters dropped from estimation (unified crisis-in-
        estimation treatment, same convention as the other models).
    """

    def __init__(
        self,
        monthly_vars: Sequence[str] | None = None,
        *,
        n_months: int = 3,
        min_obs: int = 12,
        quantile: float | None = None,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        max_iter: int = 300,
        min_samples_leaf: int = 12,
        l2_regularization: float = 1.0,
        early_stopping: bool = True,
        covid_quarters: Sequence[str] | None = None,
        random_state: int = 0,
        name: str | None = None,
    ) -> None:
        self.monthly_vars = None if monthly_vars is None else list(monthly_vars)
        self.n_months = n_months
        self.min_obs = min_obs
        self.quantile = quantile
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.early_stopping = early_stopping
        self.covid_quarters = [pd.Timestamp(q) for q in (covid_quarters or [])]
        self.random_state = random_state
        self._name = name or ("GB-Trees" + (f"[q{quantile}]" if quantile is not None else ""))
        self._cached_key = None
        self._fit_state: dict | None = None

    # -------------------------------------------------------------- design construction
    def _quarterly_feature(self, monthly: pd.DataFrame, index: pd.DatetimeIndex, var: str) -> np.ndarray:
        """Ragged-edge quarterly mean of one indicator, indexed by ``index``."""

        lags = stack_high_freq_lags(monthly[var].ffill(), index, n_lags=self.n_months, end_month=3)
        return lags.to_numpy(dtype=float).mean(axis=1)

    def _build_design(self, info: InformationSet, index: pd.DatetimeIndex):
        m_vars = self.monthly_vars or list(info.monthly.columns)
        cols = [self._quarterly_feature(info.monthly, index, v) for v in m_vars]
        Z = np.column_stack(cols) if cols else np.zeros((len(index), 0))
        return Z, m_vars

    def _feature_quarter(self, info: InformationSet) -> tuple[pd.Timestamp, int]:
        tq = pd.Period(pd.Timestamp(info.target_period), freq="Q")
        obs = info.monthly.dropna(how="all")
        last_q = pd.Period(obs.index[-1], freq="Q") if len(obs) else tq
        fq = min(tq, last_q)
        return pd.Timestamp(fq.end_time).normalize().replace(day=1), int((tq - fq).n)

    # ------------------------------------------------------------------- fit / nowcast
    def fit(self, info: InformationSet) -> "GBTreesNowcaster":
        _, h = self._feature_quarter(info)
        key = (info.origin, h)
        if self._cached_key != key or self._fit_state is None:
            self._estimate(info, info.target, info.quarterly.index, h)
            self._cached_key = key
        return self

    def _estimate(self, info: InformationSet, target: str, q_index: pd.DatetimeIndex, h: int) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        y_all = info.quarterly[target]
        Z, m_vars = self._build_design(info, q_index)
        y_shift = y_all.shift(-h)
        train = y_shift.notna().to_numpy() & y_all.shift(1).notna().to_numpy()
        if self.covid_quarters:
            tgt_q = pd.PeriodIndex(q_index, freq="Q") + h
            covid_p = [pd.Period(pd.Timestamp(q), freq="Q") for q in self.covid_quarters]
            train &= ~np.isin(tgt_q, covid_p)

        # Drop indicators with too little history in the training window. Early
        # in an expanding window many series have not started, leaving near
        # all-NaN feature columns that break the boosting binner. This mirrors
        # the DFM's ``min_series_obs`` guard.
        if Z.shape[1]:
            valid = np.isfinite(Z[train]).sum(axis=0)
            keep = valid >= self.min_obs
            Z = Z[:, keep]
            m_vars = [v for v, k in zip(m_vars, keep) if k]

        feature_names = ["AR_lag"] + list(m_vars)
        X = np.column_stack([y_all.shift(1).to_numpy(), Z]) if Z.shape[1] else y_all.shift(1).to_numpy().reshape(-1, 1)
        params = dict(
            loss="quantile" if self.quantile is not None else "squared_error",
            max_depth=self.max_depth, learning_rate=self.learning_rate, max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            early_stopping=self.early_stopping and int(train.sum()) >= 40,
        )
        if self.quantile is not None:
            params["quantile"] = self.quantile
        model = HistGradientBoostingRegressor(**params).fit(X[train], y_shift.to_numpy()[train])
        self._fit_state = dict(
            model=model, m_vars=m_vars, feature_names=feature_names, horizon=h,
            n_train=int(train.sum()), X_train=X[train], y_train=y_shift.to_numpy()[train],
        )

    def nowcast(self, info: InformationSet) -> NowcastResult:
        st = self._fit_state
        if st is None:
            raise ValueError("fit before nowcast")
        fq_ts, h = self._feature_quarter(info)
        row = [self._quarterly_feature(info.monthly, pd.DatetimeIndex([fq_ts]), v)[0] for v in st["m_vars"]]
        y_all = info.quarterly[info.target]
        prev_ts = pd.Timestamp((pd.Period(pd.Timestamp(fq_ts), freq="Q") - 1).end_time).normalize().replace(day=1)
        y_lag = y_all.get(prev_ts, np.nan)
        if not np.isfinite(y_lag):
            obs = y_all.dropna()
            y_lag = float(obs.iloc[-1]) if len(obs) else 0.0
        x = np.concatenate([[float(y_lag)], np.asarray(row, dtype=float)])[None, :]
        return NowcastResult(mean=float(st["model"].predict(x)[0]), model=self.name,
                             extra={"horizon": h, "quantile": self.quantile, "n_train": st["n_train"]})

    # ------------------------------------------------------------------- interpretability
    def importance_frame(self) -> pd.DataFrame:
        """Permutation/native feature importance for the most recent fit."""

        from .ml import feature_importance_frame

        st = self._fit_state
        if st is None:
            raise ValueError("fit before requesting importance")
        frame, _ = feature_importance_frame(st["model"], st["X_train"], st["y_train"], st["feature_names"])
        return frame

    def shap_frame(self) -> pd.DataFrame:
        """Mean absolute SHAP value per indicator for the most recent fit."""

        from .ml import shap_importance_frame

        st = self._fit_state
        if st is None:
            raise ValueError("fit before requesting SHAP")
        return shap_importance_frame(st["model"], st["X_train"], st["feature_names"])
