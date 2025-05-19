import re
import pandas as pd

def prepare_data(lf_data, hf_data_list, xlag_list, ylag, horizon, start_date=None, end_date=None):
    """
    Prepare data for mixed-frequency regression with one or multiple high-frequency datasets.

    Args:
        lf_data (Series): Low-frequency time series
        hf_data_list (list of Series): List of high-frequency time series
        xlag_list (list of int or str): List of high-frequency lags for each dataset
        ylag (int or str): Number of low-frequency lags
        horizon (int): Forecast horizon
        start_date (date, optional): Start date for data preparation
        end_date (date, optional): End date for data preparation

    Returns:
        tuple: Prepared data for regression and forecasting
    """
    num_hf = len(hf_data_list)
    assert len(xlag_list) == num_hf, "Number of xlags must match number of high-frequency datasets"

    ylag = calculate_lags(ylag, lf_data)
    xlag_list = [calculate_lags(xlag, hf_data) for xlag, hf_data in zip(xlag_list, hf_data_list)]

    min_date_y = lf_data.index[ylag]
    min_date_x = max([hf_data.index[xlag + horizon] for hf_data, xlag in zip(hf_data_list, xlag_list)])

    if min_date_y < min_date_x:
        min_date_y = next(d for d in lf_data.index if d > min_date_x)

    if (start_date is None) or (start_date < min_date_y):
        start_date = min_date_y

    if end_date is None:
        end_date = lf_data.index[-2]

    max_date = lf_data.index[-1]
    max_hf_end_date = min([hf_data.index[-1] for hf_data in hf_data_list])

    if max_date > max_hf_end_date:
        max_date = next(d for d in reversed(lf_data.index) if d < max_hf_end_date)

    if end_date > max_date:
        end_date = max_date

    forecast_start_date = lf_data.index[lf_data.index.get_loc(end_date) + 1]

    ylags = None
    if ylag > 0:
        ylags = pd.concat([lf_data.shift(l) for l in range(1, ylag + 1)], axis=1)

    x_data = [[] for _ in range(num_hf)]

    for lfdate in lf_data.loc[start_date:max_date].index:
        for i, (hf_data, xlag) in enumerate(zip(hf_data_list, xlag_list)):
            start_hf = hf_data.index.get_indexer([lfdate], method='bfill')[0]
            x_data[i].append(hf_data.iloc[start_hf - horizon: start_hf - xlag - horizon: -1].values)

    x_frames = [pd.DataFrame(data=x_rows, index=lf_data.loc[start_date:max_date].index) for x_rows in x_data]

    return (
        lf_data.loc[start_date:end_date],
        ylags.loc[start_date:end_date] if ylag > 0 else None,
        *[x.loc[start_date:end_date] for x in x_frames],
        lf_data[forecast_start_date:max_date],
        ylags[forecast_start_date:max_date] if ylag > 0 else None,
        *[x.loc[forecast_start_date:] for x in x_frames]
    )

def calculate_lags(lag, time_series):
    if isinstance(lag, str):
        return parse_lag_string(lag, data_freq(time_series)[0])
    return lag


def data_freq(time_series):
    try:
        freq = time_series.index.freq
        return freq.freqstr or pd.infer_freq(time_series.index)
    except AttributeError:
        return pd.infer_freq(time_series.index)


def parse_lag_string(lag_string, freq):
    freq_map = {
        'd': {'m': 30, 'd': 1},
        'b': {'m': 22, 'b': 1},
        'm': {'q': 3, 'm': 1},
        'q': {'y': 4},
        'a': {'y': 1}
    }
    m = re.match(r'(\d+)(\w)', lag_string)
    duration = int(m.group(1))
    period = m.group(2).lower()
    return duration * freq_map[freq.lower()][period]
