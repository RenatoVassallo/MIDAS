"""Rolling/expanding-window forecast driver for panel data."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def rolling_forecast(
    panel: pd.DataFrame,
    target: str,
    features: list[str],
    fit_predict: Callable,
    *,
    eval_start: str,
    horizon: int = 1,
    window: str = "expanding",
    min_train: int = 24,
    by_country: str | None = "country",
) -> pd.DataFrame:
    """Generic rolling-forecast loop.

    panel: long-format DataFrame with a `date` column and (optional) `country` column.
    fit_predict: callable taking (X_train, y_train, X_test) -> (y_pred,).
                 May also return arrays for batch test sets.

    Returns a DataFrame with [country, date, y_true, y_hat].
    """
    if "date" not in panel.columns:
        raise ValueError("panel must have a 'date' column")
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    eval_start_ts = pd.to_datetime(eval_start)

    groups = df.groupby(by_country) if by_country else [(None, df)]
    out_rows = []
    for key, sub in groups:
        sub = sub.sort_values("date").reset_index(drop=True)
        eval_dates = sub.loc[sub["date"] >= eval_start_ts, "date"].tolist()
        for t in eval_dates:
            train_mask = sub["date"] < t
            train = sub.loc[train_mask].dropna(subset=[target] + features)
            test_row = sub.loc[sub["date"] == t]
            if len(train) < min_train or test_row.empty:
                continue
            if window == "rolling":
                train = train.tail(min_train * 4)
            X_tr = train[features].values
            y_tr = train[target].values
            X_te = test_row[features].values
            try:
                yhat = fit_predict(X_tr, y_tr, X_te)
            except Exception:
                yhat = np.array([np.nan])
            yhat = np.asarray(yhat).ravel()
            for j, row in enumerate(test_row.itertuples()):
                out_rows.append({
                    "country": key,
                    "date":    row.date,
                    "y_true":  getattr(row, target),
                    "y_hat":   float(yhat[j]) if j < len(yhat) else np.nan,
                })
    return pd.DataFrame(out_rows)


def make_lags(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int],
    by: str | None = "country",
) -> pd.DataFrame:
    """Append lagged versions of `cols`. Returns a copy."""
    out = df.copy()
    if by is not None:
        for col in cols:
            for k in lags:
                out[f"{col}_l{k}"] = out.groupby(by)[col].shift(k)
    else:
        for col in cols:
            for k in lags:
                out[f"{col}_l{k}"] = out[col].shift(k)
    return out
