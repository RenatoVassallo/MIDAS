"""Sparse-group LASSO MIDAS.

The machine-learning MIDAS regression of Babii, Ghysels and Striaukas (2022): each
monthly predictor's high-frequency lags are projected onto a Legendre dictionary
(:func:`~MIDAS.midas.legendre_dictionary`), so the predictor becomes a *group* of
coefficients, and a sparse-group LASSO selects predictors and prunes terms.

One file per model, as in :mod:`MIDAS.midas`: it holds both the estimator,
:class:`SparseGroupLasso` (a general grouped-sparse regression by FISTA), and the
thin :class:`~MIDAS.base.BaseNowcaster` wrapper, :class:`SparseMIDASNowcaster`,
which builds the MIDAS design from an :class:`~MIDAS.base.InformationSet` and
delegates the fit, exactly like :class:`~MIDAS.benchmarks.ADLMIDASNowcaster` wraps
``BetaMIDASRegressor`` and :class:`~MIDAS.dfm.DFMNowcaster` wraps ``DynamicFactorMQ``.

Design handling in the nowcaster:

* **Ragged edge**: each masked monthly series is forward-filled before its lags are
  stacked, so every predictor contributes its latest release.
* **Unbalanced panel**: dictionary features of not-yet-started series are mean-imputed
  (zero after standardisation), so long histories are never discarded.
* **Multi-horizon**: MIDAS cannot be iterated, so horizon ``h`` uses the *direct*
  approach, regressing ``y_{t+h}`` on features known at ``t``.
* **Robustness / breaks**: an optional Huber loss, plus ``covid_quarters`` to drop
  named outlier quarters from estimation and tuning (the outlier values are the
  caller's choice, so nothing pandemic-specific lives in the package).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .align import stack_high_freq_lags
from .base import BaseNowcaster, InformationSet, NowcastResult
from .midas import legendre_dictionary


# ============================================================ estimator: SparseGroupLasso
def _fista(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    penalized: np.ndarray,
    lam: float,
    alpha: float,
    *,
    loss: str,
    huber_delta: float,
    max_iter: int,
    tol: float,
    beta_init: np.ndarray | None,
    lipschitz: float,
) -> np.ndarray:
    """One FISTA solve of the sparse-group-LASSO objective at penalty ``lam``.

    Columns in ``penalized`` get the lasso soft-threshold; each array in ``groups``
    then gets a group soft-threshold. Unpenalised columns pass through untouched.
    """
    n, p = X.shape
    step = 1.0 / lipschitz
    beta = np.zeros(p) if beta_init is None else beta_init.copy()
    z = beta.copy()
    t = 1.0
    pen = np.where(penalized)[0]
    l1_thr = step * lam * alpha
    grp_thr = step * lam * (1.0 - alpha)
    for _ in range(max_iter):
        r = y - X @ z
        if loss == "squared":
            grad = -(X.T @ r) / n
        elif loss == "huber":
            grad = -(X.T @ np.clip(r, -huber_delta, huber_delta)) / n
        else:
            raise ValueError("loss must be 'squared' or 'huber'")
        b = z - step * grad
        beta_new = b.copy()
        beta_new[pen] = np.sign(b[pen]) * np.maximum(np.abs(b[pen]) - l1_thr, 0.0)  # lasso prox
        for idx in groups:                                                          # group prox
            norm = np.linalg.norm(beta_new[idx])
            if norm > 0:
                beta_new[idx] *= max(1.0 - grp_thr / norm, 0.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = beta_new + ((t - 1.0) / t_new) * (beta_new - beta)
        if np.linalg.norm(beta_new - beta) <= tol * (np.linalg.norm(beta) + 1e-12):
            return beta_new
        beta, t = beta_new, t_new
    return beta


def _lambda_path(X, y, penalized, alpha, n_lambda, eps) -> np.ndarray:
    """Descending penalty grid from a lambda that zeros every penalised coefficient."""
    unpen = np.where(~penalized)[0]
    if len(unpen):
        coef, *_ = np.linalg.lstsq(X[:, unpen], y, rcond=None)
        r = y - X[:, unpen] @ coef
    else:
        r = y - y.mean()
    corr = np.abs(X[:, penalized].T @ r) / len(y)
    lam_max = max(float(corr.max()) / max(alpha, 0.05), 1e-8)
    return np.exp(np.linspace(np.log(lam_max), np.log(lam_max * eps), n_lambda))


class SparseGroupLasso:
    """Sparse-group LASSO by FISTA, with an optional one-standard-error CV path.

    Minimises ``(1/2n) sum rho(y - X beta) + lambda[(1-alpha) sum_g sqrt(|g|)
    ||beta_g||_2 + alpha ||beta_pen||_1]`` where ``rho`` is the squared or Huber loss.
    Columns absent from every group are left unpenalised (intercept, controls).

    A general, estimator-only class (it takes a ready design and group index and
    returns coefficients), mirroring :class:`~MIDAS.midas.BetaMIDASRegressor`.

    Parameters
    ----------
    alpha:
        Mix between the lasso (``alpha``) and group (``1 - alpha``) penalties.
    loss:
        ``"squared"`` or robust ``"huber"``.
    lam:
        Fixed penalty. If ``None`` (default), chosen on a time-ordered holdout over a
        ``n_lambda``-point path.
    one_se:
        Use the one-standard-error rule (most parsimonious model within one s.e. of the
        best holdout error) rather than the raw minimum, which a single noisy holdout
        can push toward zero penalty.

    Attributes
    ----------
    coef_ : np.ndarray
        Fitted coefficients (all columns; unpenalised ones included).
    lambda_ : float
        The penalty used.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        loss: str = "squared",
        huber_delta: float = 1.35,
        lam: float | None = None,
        n_lambda: int = 50,
        eps: float = 1e-3,
        one_se: bool = True,
        val_frac: float = 0.25,
        max_iter: int = 2000,
        tol: float = 1e-7,
    ) -> None:
        self.alpha = alpha
        self.loss = loss
        self.huber_delta = huber_delta
        self.lam = lam
        self.n_lambda = n_lambda
        self.eps = eps
        self.one_se = one_se
        self.val_frac = val_frac
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _penalized_mask(p: int, groups: Sequence[np.ndarray]) -> np.ndarray:
        mask = np.zeros(p, dtype=bool)
        for g in groups:
            mask[np.asarray(g)] = True
        return mask

    def _solve(self, X, y, groups, penalized, lam) -> np.ndarray:
        lip = (np.linalg.norm(X, 2) ** 2) / len(y)
        return _fista(X, y, list(groups), penalized, lam, self.alpha, loss=self.loss,
                      huber_delta=self.huber_delta, max_iter=self.max_iter, tol=self.tol,
                      beta_init=None, lipschitz=lip)

    def _tune(self, X, y, groups, penalized, lambdas) -> float:
        n = len(y)
        n_val = max(8, int(n * self.val_frac))
        if n - n_val < 12:
            return float(lambdas[len(lambdas) // 2])
        Xtr, ytr, Xv, yv = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
        lip = (np.linalg.norm(Xtr, 2) ** 2) / len(ytr)
        beta, mse, sem = None, [], []
        for lam in lambdas:  # descending, warm-started along the path
            beta = _fista(Xtr, ytr, list(groups), penalized, lam, self.alpha, loss=self.loss,
                          huber_delta=self.huber_delta, max_iter=self.max_iter, tol=self.tol,
                          beta_init=beta, lipschitz=lip)
            sq = (yv - Xv @ beta) ** 2
            mse.append(float(sq.mean()))
            sem.append(float(sq.std(ddof=1) / np.sqrt(len(sq))) if len(sq) > 1 else 0.0)
        mse = np.asarray(mse)
        best = int(np.argmin(mse))
        if not self.one_se:
            return float(lambdas[best])
        within = np.where(mse <= mse[best] + sem[best])[0]  # largest lambda within 1 s.e.
        return float(lambdas[int(within[0])]) if len(within) else float(lambdas[best])

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Sequence[np.ndarray]) -> "SparseGroupLasso":
        """Fit on design ``X`` with penalised column ``groups`` (index arrays).

        Columns absent from every group are unpenalised.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = [np.asarray(g, dtype=int) for g in groups]
        penalized = self._penalized_mask(X.shape[1], groups)
        lambdas = _lambda_path(X, y, penalized, self.alpha, self.n_lambda, self.eps)
        self.lambda_ = float(self.lam) if self.lam is not None else self._tune(X, y, groups, penalized, lambdas)
        self.coef_ = self._solve(X, y, groups, penalized, self.lambda_)
        self.groups_ = groups
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_


# ============================================================ nowcaster: SparseMIDASNowcaster
class SparseMIDASNowcaster(BaseNowcaster):
    """Sparse-group LASSO MIDAS with a Legendre dictionary (Babii et al., 2022)."""

    def __init__(
        self,
        monthly_vars: Sequence[str] | None = None,
        *,
        n_lags: int = 12,
        degree: int = 2,
        alpha: float = 0.5,
        loss: str = "huber",
        huber_delta: float = 1.35,
        covid_quarters: Sequence[str] | None = None,
        n_lambda: int = 50,
        val_frac: float = 0.25,
        name: str | None = None,
    ) -> None:
        self.monthly_vars = None if monthly_vars is None else list(monthly_vars)
        self.n_lags = n_lags
        self.degree = degree
        self.alpha = alpha
        self.loss = loss
        self.huber_delta = huber_delta
        self.covid_quarters = [pd.Timestamp(q) for q in (covid_quarters or [])]
        self.n_lambda = n_lambda
        self.val_frac = val_frac
        self._name = name or "sg-LASSO"
        self._cached_key = None
        self._fit_state: dict | None = None

    # ------------------------------------------------------------- design construction
    def _dictionary_block(self, monthly: pd.DataFrame, index: pd.DatetimeIndex, var: str) -> np.ndarray:
        """Legendre dictionary features (len(index) x degree+1) for one predictor."""
        lags = stack_high_freq_lags(monthly[var].ffill(), index, n_lags=self.n_lags, end_month=3)
        return lags.to_numpy() @ legendre_dictionary(self.n_lags, self.degree)

    def _build_dictionary(self, info: InformationSet, index: pd.DatetimeIndex):
        m_vars = self.monthly_vars or list(info.monthly.columns)
        blocks, groups, col = [], [], 0
        for var in m_vars:
            Z = self._dictionary_block(info.monthly, index, var)
            blocks.append(Z)
            groups.append(np.arange(col, col + Z.shape[1]))
            col += Z.shape[1]
        Z = np.column_stack(blocks) if blocks else np.zeros((len(index), 0))
        return Z, groups, m_vars

    def _feature_quarter(self, info: InformationSet) -> tuple[pd.Timestamp, int]:
        """Quarter whose data feeds the design, and the implied forecast horizon.

        The feature quarter is the latest quarter with any monthly data at this origin,
        capped at the target: for a within-quarter nowcast it is the target itself
        (``h=0``); from the next quarter onward it is the quarter that just ended
        (``h`` grows with the target). Multi-step forecasts are therefore *direct*.
        """
        tq = pd.Period(pd.Timestamp(info.target_period), freq="Q")
        obs = info.monthly.dropna(how="all")
        last_q = pd.Period(obs.index[-1], freq="Q") if len(obs) else tq
        fq = min(tq, last_q)
        return pd.Timestamp(fq.end_time).normalize().replace(day=1), int((tq - fq).n)

    # ------------------------------------------------------------------- fit / nowcast
    def fit(self, info: InformationSet) -> "SparseMIDASNowcaster":
        _, h = self._feature_quarter(info)
        key = (info.origin, h)          # the design depends on the origin's data and the horizon
        if self._cached_key != key or self._fit_state is None:
            self._estimate(info, info.target, info.quarterly.index, h)
            self._cached_key = key
        self._target = info.target
        return self

    def _estimate(self, info: InformationSet, target: str, q_index: pd.DatetimeIndex, h: int) -> None:
        y_all = info.quarterly[target]
        Zdict, groups, m_vars = self._build_dictionary(info, q_index)

        # direct h-step training rows: row t carries features Z_t and realised target
        # y_{t+h}, and needs a released lag y_{t-1}. Both target series are masked, so
        # unreleased quarters drop out -> no look-ahead.
        y_shift = y_all.shift(-h)
        train = y_shift.notna().to_numpy() & y_all.shift(1).notna().to_numpy()
        if self.covid_quarters:
            # drop named outlier quarters from estimation and tuning: a dummy is not enough,
            # since it is all-zero in the tuning-train split when its quarter is in validation,
            # which then hijacks the penalty choice.
            tgt_q = pd.PeriodIndex(q_index, freq="Q") + h
            covid_p = [pd.Period(pd.Timestamp(q), freq="Q") for q in self.covid_quarters]
            train &= ~np.isin(tgt_q, covid_p)

        # Standardise dictionary columns on the TRAINING rows only, then mean-impute (-> 0).
        # Moments over the whole index would be dominated by the forward-filled tail of
        # not-yet-released quarters (every post-origin row repeats the last release), which
        # biases mu/sd toward the most recent observation and makes them origin-dependent.
        Ztr = Zdict[train]
        col_ok = np.isfinite(Ztr).any(axis=0)  # columns with at least one training value
        mu = np.zeros(Zdict.shape[1])
        sd = np.ones(Zdict.shape[1])
        if col_ok.any():
            mu[col_ok] = np.nanmean(Ztr[:, col_ok], axis=0)
            sd_ok = np.nanstd(Ztr[:, col_ok], axis=0)
            sd_ok[~np.isfinite(sd_ok) | (sd_ok == 0)] = 1.0
            sd[col_ok] = sd_ok
        Zstd = np.where(np.isfinite((Zdict - mu) / sd), (Zdict - mu) / sd, 0.0)

        # unpenalised covariates: intercept and the last released target value. The AR term
        # is y_{t-1}, not y_t: at h=0 the target IS y_t (a leak) and y_t is unpublished anyway.
        U = np.column_stack([np.ones(len(q_index)), y_all.shift(1).to_numpy()])
        n_unpen = U.shape[1]
        X = np.column_stack([U, Zstd])
        group_index = [g + n_unpen for g in groups]

        est = SparseGroupLasso(alpha=self.alpha, loss=self.loss, huber_delta=self.huber_delta,
                               n_lambda=self.n_lambda, val_frac=self.val_frac, one_se=True)
        est.fit(X[train], y_shift.to_numpy()[train], group_index)
        self._fit_state = dict(est=est, mu=mu, sd=sd, m_vars=m_vars, groups=groups,
                               n_unpen=n_unpen, horizon=h, n_train=int(train.sum()))

    def nowcast(self, info: InformationSet) -> NowcastResult:
        st = self._fit_state
        if st is None:
            raise ValueError("fit before nowcast")
        fq_ts, h = self._feature_quarter(info)
        row = [self._dictionary_block(info.monthly, pd.DatetimeIndex([fq_ts]), v)[0]
               for v in st["m_vars"]]
        z = np.concatenate(row) if row else np.zeros(0)
        z = np.where(np.isfinite((z - st["mu"]) / st["sd"]), (z - st["mu"]) / st["sd"], 0.0)

        y_all = info.quarterly[info.target]
        prev_ts = pd.Timestamp((pd.Period(pd.Timestamp(fq_ts), freq="Q") - 1).end_time).normalize().replace(day=1)
        y_lag = y_all.get(prev_ts, np.nan)
        if not np.isfinite(y_lag):                     # not released: use the last released value
            obs = y_all.dropna()
            y_lag = float(obs.iloc[-1]) if len(obs) else 0.0
        x = np.concatenate([np.asarray([1.0, float(y_lag)]), z])
        coef = st["est"].coef_
        raw = float(x @ coef)
        anchor = float(x[: st["n_unpen"]] @ coef[: st["n_unpen"]])  # unpenalised part: intercept + AR*y_lag

        # Stability guard (mirrors the DFM divergence guard). A noisy one-standard-error penalty can
        # pick too small a lambda on a volatile post-break quarter, so the dictionary block
        # extrapolates far from the AR baseline (observed: a 2022Q1 GDP nowcast of 24, and ~15 earlier
        # in the same quarter, vs a ~4 realised). We guard on the *dictionary contribution*
        # ``raw - anchor`` rather than the level: a genuine high-growth quarter has a high anchor too,
        # so its contribution is small and it is never clipped, whereas a blow-up is driven by the
        # dictionary alone. The scale is a robust MAD of the target (COVID outliers do not widen it).
        mean, guarded = raw, False
        y_obs = info.quarterly[info.target].dropna()
        if len(y_obs) >= 8:
            scale = max(float((y_obs - y_obs.median()).abs().median()) * 1.4826, 1e-6)
            if abs(raw - anchor) > 3.0 * scale:      # dictionary extrapolating far from the AR baseline
                mean, guarded = anchor, True
        return NowcastResult(mean=float(mean), model=self.name,
                             extra={"lambda": st["est"].lambda_, "horizon": h,
                                    "n_selected": self.n_selected(), "guarded": guarded,
                                    "raw_pred": raw, "anchor": anchor})

    # ------------------------------------------------------------------- interpretability
    def _dict_coef(self) -> np.ndarray:
        if self._fit_state is None:
            raise ValueError("fit before inspecting the model")
        return self._fit_state["est"].coef_[self._fit_state["n_unpen"]:]

    def selected_variables(self) -> list[str]:
        """Monthly predictors with at least one non-zero dictionary coefficient."""
        beta = self._dict_coef()
        return [v for v, g in zip(self._fit_state["m_vars"], self._fit_state["groups"])
                if np.any(beta[g] != 0)]

    def n_selected(self) -> int:
        return len(self.selected_variables())

    def coefficient_table(self) -> pd.DataFrame:
        """Non-zero group L2 norms per predictor (importance), largest first."""
        beta = self._dict_coef()
        rows = [{"variable": v, "l2_norm": float(np.linalg.norm(beta[g]))}
                for v, g in zip(self._fit_state["m_vars"], self._fit_state["groups"])]
        tab = pd.DataFrame(rows)
        return tab[tab.l2_norm > 0].sort_values("l2_norm", ascending=False).reset_index(drop=True)
