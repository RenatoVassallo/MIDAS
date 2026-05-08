"""Mixed-frequency alignment utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_monthly_to_quarter(
    monthly: pd.DataFrame,
    method: str = "mean",
    end_month: int | None = None,
) -> pd.DataFrame:
    """Aggregate a monthly DataFrame (DatetimeIndex) to quarterly.

    method:
        "mean"  - simple average over the quarter
        "last"  - value of the last month available
        "sum"   - quarterly sum (flow variables)

    end_month:
        If given (1, 2, or 3), keep only data up to month-of-quarter <= end_month.
        Useful to mimic ragged-edge availability when nowcasting.
    """
    df = monthly.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("monthly index must be a DatetimeIndex")

    if end_month is not None:
        if end_month not in (1, 2, 3):
            raise ValueError("end_month must be 1, 2, or 3")
        keep = df.index.month.isin([3 * (q - 1) + m for q in (1, 2, 3, 4) for m in range(1, end_month + 1)])
        df = df.loc[keep]

    rule = "Q"
    if method == "mean":
        out = df.resample(rule).mean()
    elif method == "last":
        out = df.resample(rule).last()
    elif method == "sum":
        out = df.resample(rule).sum(min_count=1)
    else:
        raise ValueError(f"unknown method: {method}")
    return out


def ragged_edge_lattice(
    series_specs: dict[str, int],
    forecast_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build a 0/1 availability matrix illustrating the ragged edge.

    series_specs: maps series name -> publication-delay in days.
    Returns a DataFrame indexed by series, columns = the last 6 months.
    """
    months = pd.date_range(end=forecast_date, periods=6, freq="MS")
    out = pd.DataFrame(index=list(series_specs), columns=months, dtype=int)
    for name, delay in series_specs.items():
        for m in months:
            release = m + pd.offsets.MonthEnd(1) + pd.Timedelta(days=delay)
            out.loc[name, m] = int(release <= forecast_date)
    return out


def stack_high_freq_lags(
    high: pd.Series,
    target_index: pd.DatetimeIndex,
    n_lags: int,
    months_per_period: int = 3,
    end_month: int | None = None,
) -> pd.DataFrame:
    """For each quarterly date in target_index, stack the n_lags most recent
    monthly values of `high` (latest month first).

    end_month:
        None or 3 -> use the last month of the target quarter (M3).
        2         -> use the second month of the target quarter (M2).
        1         -> use the first month of the target quarter (M1).
        0         -> use only information through the previous quarter (L1Q).

    Returns a (len(target_index) x n_lags) DataFrame; columns named L0..L{n-1}.
    """
    out = pd.DataFrame(index=target_index, columns=[f"L{i}" for i in range(n_lags)], dtype=float)
    for q in target_index:
        quarter = pd.Timestamp(q).to_period("Q")
        quarter_start = quarter.start_time.normalize().replace(day=1)
        quarter_end = quarter.end_time.normalize().replace(day=1)
        if end_month is None or end_month == 3:
            anchor = quarter_end
        elif end_month == 0:
            anchor = quarter_start - pd.DateOffset(months=1)
        elif end_month in (1, 2):
            anchor = quarter_start + pd.DateOffset(months=end_month - 1)
        else:
            raise ValueError("end_month must be one of {0, 1, 2, 3, None}")
        idx = pd.date_range(end=anchor, periods=n_lags, freq="MS")[::-1]
        try:
            out.loc[q, :] = high.reindex(idx).values
        except Exception:
            out.loc[q, :] = np.nan
    return out
