"""Robust ML helpers used in notebooks 04-06.

The teaching notebooks should run cleanly on student machines even when
optional compiled packages are unavailable. We therefore try XGBoost/SHAP once,
cache the result, and transparently fall back to fully portable scikit-learn
implementations.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

_XGBOOST_MODULE: Any | None = None
_XGBOOST_IMPORT_ERROR: Exception | None = None
_SHAP_MODULE: Any | None = None
_SHAP_IMPORT_ERROR: Exception | None = None
_WARNED_XGB_REGRESSOR = False
_WARNED_XGB_CLASSIFIER = False


def _load_xgboost():
    """Import xgboost at most once."""
    global _XGBOOST_MODULE, _XGBOOST_IMPORT_ERROR
    if _XGBOOST_MODULE is not None:
        return _XGBOOST_MODULE
    if _XGBOOST_IMPORT_ERROR is not None:
        raise _XGBOOST_IMPORT_ERROR
    try:
        import xgboost as xgb

        _XGBOOST_MODULE = xgb
        return xgb
    except Exception as exc:  # pragma: no cover - runtime environment specific
        _XGBOOST_IMPORT_ERROR = exc
        raise


def _load_shap():
    """Import shap at most once."""
    global _SHAP_MODULE, _SHAP_IMPORT_ERROR
    if _SHAP_MODULE is not None:
        return _SHAP_MODULE
    if _SHAP_IMPORT_ERROR is not None:
        raise _SHAP_IMPORT_ERROR
    try:
        import shap

        _SHAP_MODULE = shap
        return shap
    except Exception as exc:  # pragma: no cover - runtime environment specific
        _SHAP_IMPORT_ERROR = exc
        raise


def make_xgb_regressor(**kwargs):
    """Return XGBoost when available, otherwise a safe tree-boosting fallback."""
    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 0,
        "n_jobs": 1,
    }
    params.update(kwargs)
    global _WARNED_XGB_REGRESSOR
    try:
        xgb = _load_xgboost()

        model = xgb.XGBRegressor(**params)
        model._midas_display_name = "XGBoost"
        return model
    except Exception as exc:  # pragma: no cover - runtime environment specific
        if not _WARNED_XGB_REGRESSOR:
            warnings.warn(
                f"XGBoost is unavailable ({exc}). Falling back to HistGradientBoostingRegressor.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_XGB_REGRESSOR = True
        model = HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model._midas_display_name = "HistGradientBoosting fallback"
        return model


def make_xgb_classifier(**kwargs):
    """Return XGBoost classifier when available, otherwise a safe fallback."""
    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 0,
        "n_jobs": 1,
    }
    params.update(kwargs)
    global _WARNED_XGB_CLASSIFIER
    try:
        xgb = _load_xgboost()

        model = xgb.XGBClassifier(**params)
        model._midas_display_name = "XGBoost"
        return model
    except Exception as exc:  # pragma: no cover - runtime environment specific
        if not _WARNED_XGB_CLASSIFIER:
            warnings.warn(
                f"XGBoost is unavailable ({exc}). Falling back to HistGradientBoostingClassifier.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_XGB_CLASSIFIER = True
        model = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model._midas_display_name = "HistGradientBoosting fallback"
        return model


def feature_importance_frame(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    top_n: int = 12,
) -> tuple[pd.DataFrame, str]:
    """Return a tidy feature-importance table and the method used.

    Prefer SHAP for tree models; fall back to permutation importance when SHAP
    or the tree backend is unavailable.
    """
    try:
        shap = _load_shap()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        scores = np.abs(np.asarray(shap_values)).mean(axis=0)
        method = "SHAP"
    except Exception:  # pragma: no cover - runtime environment specific
        scores = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=0,
            n_jobs=1,
        ).importances_mean
        method = "Permutation importance"

    frame = (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return frame, method


def shap_importance_frame(
    model,
    X: np.ndarray,
    feature_names: list[str],
    *,
    top_n: int = 12,
    class_index: int | None = None,
) -> pd.DataFrame:
    """Return mean absolute SHAP values for tree-compatible models.

    For binary classifiers, the positive class is used by default.
    """
    shap = _load_shap()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    values = np.asarray(shap_values)

    if isinstance(shap_values, list):
        idx = 1 if class_index is None and len(shap_values) > 1 else (class_index or 0)
        values = np.asarray(shap_values[idx])
    elif values.ndim == 3:
        idx = 1 if class_index is None and values.shape[2] > 1 else (class_index or 0)
        values = values[:, :, idx]

    scores = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def make_boosted_regressor(**kwargs):
    """Alias with a teaching-friendly name."""
    return make_xgb_regressor(**kwargs)


def make_boosted_classifier(**kwargs):
    """Alias with a teaching-friendly name."""
    return make_xgb_classifier(**kwargs)
