import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.optimize import least_squares
from .polynomials import polynomial_weights
from .optim_func import ssr, jacobian
from .preprocess import prepare_data
from statsmodels.tsa.ar_model import AutoReg

class MIDAS:
    def __init__(self, low_freq_series, high_freq_series_list, hf_lags_list, lf_lags):
        """
        Initialize the MIDAS model.

        Args:
            low_freq_series (pd.Series): Low-frequency target variable (e.g., GDP)
            high_freq_series_list (list of pd.Series): List of high-frequency regressors
            hf_lags_list (list of int or str): Lags for each high-frequency regressor
            lf_lags (int): Number of lags of the low-frequency variable (AR terms)
        """
        self.low_freq_series = self._validate_series(low_freq_series, "low_freq_series").dropna()
        self.high_freq_series_list = [self._validate_series(s, f"high_freq_series_list[{i}]")
                                      for i, s in enumerate(high_freq_series_list)]
        self.hf_lags_list = hf_lags_list
        self.lf_lags = lf_lags
        self.res = None
        self.poly_list = None
        self.current_end_date = None

    def _validate_series(self, series, name):
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex.")
        if pd.infer_freq(series.index) is None:
            raise ValueError(f"Frequency of {name}'s index could not be inferred. Ensure regular time steps.")
        return series.asfreq(pd.infer_freq(series.index))

    def prepare_data(self, forecast_horizon, start_date, end_date):
        if len(self.high_freq_series_list) == 0:
            return None  # AR-only case
        result = prepare_data(
            self.low_freq_series,
            self.high_freq_series_list,
            self.hf_lags_list,
            self.lf_lags,
            forecast_horizon,
            start_date,
            end_date
        )

        return {
            "y_train": result[0],
            "y_train_lags": result[1],
            "x_train_list": result[2:2 + len(self.high_freq_series_list)],
            "y_forecast_target": result[2 + len(self.high_freq_series_list)],
            "y_forecast_lags": result[3 + len(self.high_freq_series_list)],
            "x_forecast_list": result[4 + len(self.high_freq_series_list):]
        }

    def fit(self, data_dict, poly_list=None):
        if data_dict is None:
            # AR-only model training: use data up to current rolling end date
            end_idx = self.low_freq_series.index.get_loc(self.current_end_date)
            y = self.low_freq_series.iloc[:end_idx + 1].dropna()

            if len(y) <= self.lf_lags:
                raise ValueError(f"Not enough observations for AR({self.lf_lags}) model at {self.current_end_date}.")

            self.ar_model = AutoReg(y, lags=self.lf_lags, old_names=False).fit()
            return self.ar_model

        # ---------- MIDAS model training ----------
        x_train_list = data_dict["x_train_list"]
        y_train = data_dict["y_train"]
        y_train_lags = data_dict["y_train_lags"]
        self.poly_list = poly_list or ['beta'] * len(x_train_list)

        assert len(self.poly_list) == len(x_train_list), "Mismatch in number of regressors and weight methods."

        weight_methods = [polynomial_weights(poly) for poly in self.poly_list]
        x_weighted = []
        init_weight_params = []

        for x, method in zip(x_train_list, weight_methods):
            params = method.init_params()
            xw, _ = method.x_weighted(x, params)
            x_weighted.append(xw)
            init_weight_params.extend(params)

        if y_train_lags is not None:
            design_matrix = np.concatenate([
                np.ones((len(y_train), 1)),
                *[xw.reshape(-1, 1) for xw in x_weighted],
                y_train_lags
            ], axis=1)
        else:
            design_matrix = np.concatenate([
                np.ones((len(y_train), 1)),
                *[xw.reshape(-1, 1) for xw in x_weighted]
            ], axis=1)

        num_regressors = len(x_train_list)
        ols_params = np.linalg.lstsq(design_matrix, y_train, rcond=None)[0]
        ar_params = ols_params[num_regressors + 1:] if y_train_lags is not None else np.array([])

        init_params = np.concatenate([
            ols_params[:num_regressors + 1],
            init_weight_params,
            ar_params
        ])

        def ssr_func(v):
            return ssr(v, x_train_list, y_train.values,
                    y_train_lags.values if y_train_lags is not None else None,
                    weight_methods)

        def jac_func(v):
            return jacobian(v, x_train_list, y_train.values,
                            y_train_lags.values if y_train_lags is not None else None,
                            weight_methods)

        self.res = least_squares(ssr_func, init_params, jac_func,
                                xtol=1e-9, ftol=1e-9, max_nfev=5000, verbose=0)
        return self.res

    def predict(self, data_dict):
        if data_dict is None:
            # AR-only forecast
            idx_pos = self.low_freq_series.index.get_loc(self.current_end_date) + 1
            if idx_pos >= len(self.low_freq_series):
                raise IndexError(f"Cannot forecast beyond available index at {self.current_end_date}.")
            forecast_idx = self.low_freq_series.index[idx_pos]

            forecast = self.ar_model.predict(start=self.ar_model.nobs, end=self.ar_model.nobs)
            return pd.DataFrame(forecast.values.reshape(-1, 1), index=[forecast_idx], columns=["y_forecast"])

        # ---------- MIDAS model forecast ----------
        if self.res is None:
            raise ValueError("Model is not fitted. Call `.fit()` first.")

        x_forecast_list = data_dict["x_forecast_list"]
        y_forecast_lags = data_dict["y_forecast_lags"]

        num_regressors = len(x_forecast_list)
        weight_methods = [polynomial_weights(p) for p in self.poly_list]
        a = self.res.x
        intercept = a[0]
        betas = a[1:1 + num_regressors]

        offset = 1 + num_regressors
        xw_list = []
        for i in range(num_regressors):
            method = weight_methods[i]
            theta = a[offset:offset + method.num_params]
            xfc = x_forecast_list[i].values if isinstance(x_forecast_list[i], pd.DataFrame) else x_forecast_list[i]
            xw, _ = method.x_weighted(xfc, theta)
            xw_list.append(xw)
            offset += method.num_params

        yf = intercept + sum(b * xw for b, xw in zip(betas, xw_list))

        if y_forecast_lags is not None:
            ar_params = a[offset:]
            ylags = y_forecast_lags.values if isinstance(y_forecast_lags, pd.DataFrame) else y_forecast_lags
            for i in range(len(ar_params)):
                yf += ar_params[i] * ylags[:, i]

        return pd.DataFrame(yf, index=x_forecast_list[0].index, columns=["y_forecast"])

    def rolling_forecast(self, start_date, end_date, forecast_horizon=1, poly_list=None, verbose=False):
        """
        Perform rolling forecast using an expanding window.

        Args:
            start_date (datetime): Start of training window.
            end_date (datetime): First forecast origin.
            forecast_horizon (int): h-step-ahead forecast.
            poly_list (list of str): Polynomial types per regressor (only for MIDAS).
            verbose (bool): If True, print progress and diagnostics.

        Returns:
            (pd.DataFrame, float): DataFrame with predictions and targets, and RMSE.
        """
        preds, targets, dates = [], [], []

        forecast_start_loc = self.low_freq_series.index.get_loc(end_date)
        max_idx = len(self.low_freq_series) - forecast_horizon
        model_end_dates = self.low_freq_series.index[forecast_start_loc:max_idx]

        if len(model_end_dates) == 0:
            raise ValueError(
                f"No valid forecast window: `end_date` ({end_date.strftime('%Y-%m-%d')}) "
                f"is too late relative to data length ({len(self.low_freq_series)} samples) "
                f"and forecast_horizon={forecast_horizon}."
            )

        for estimate_end in model_end_dates:
            self.current_end_date = estimate_end  # For AR-only slicing logic

            if verbose:
                print(f"Rolling window end = {estimate_end} | Forecast horizon = {forecast_horizon}")

            try:
                # Handle AR-only case by passing None
                data_dict = self.prepare_data(
                    forecast_horizon=forecast_horizon,
                    start_date=start_date,
                    end_date=estimate_end
                )

                if data_dict is not None and len(data_dict["y_forecast_target"]) < forecast_horizon:
                    if verbose:
                        print(f"Skipping {estimate_end} due to insufficient forecast horizon.")
                    continue

                self.fit(data_dict, poly_list=poly_list)
                forecast = self.predict(data_dict)

                if data_dict is None:
                    # AR-only: get true value manually from original series
                    idx_pos = self.low_freq_series.index.get_loc(estimate_end) + forecast_horizon
                    if idx_pos >= len(self.low_freq_series):
                        if verbose:
                            print(f"Skipping {estimate_end}: no true target available.")
                        continue
                    true = self.low_freq_series.iloc[idx_pos]
                    dt = self.low_freq_series.index[idx_pos]

                    if pd.isna(true):
                        if verbose:
                            print(f"Skipping {dt} due to missing target.")
                        continue
                else:
                    true = data_dict["y_forecast_target"].iloc[forecast_horizon - 1]
                    dt = data_dict["y_forecast_target"].index[forecast_horizon - 1]

                pred = forecast.iloc[forecast_horizon - 1, 0]
                if pd.isna(pred) or pd.isna(true):
                    if verbose:
                        print(f"Skipping {dt} due to NaN in prediction or target.")
                    continue

                preds.append(pred)
                targets.append(true)
                dates.append(dt)

            except Exception as e:
                if verbose:
                    print(f"⚠ Skipping {estimate_end} due to error: {e}")
                continue

        preds = np.array(preds)
        targets = np.array(targets)

        if len(preds) == 0 or len(targets) == 0:
            raise ValueError("No valid forecasts generated.")

        rmse = sqrt(mean_squared_error(targets, preds))
        results_df = pd.DataFrame({"preds": preds, "targets": targets}, index=pd.DatetimeIndex(dates))

        if verbose:
            print(f"\n✅ Final RMSE: {rmse:.4f}")

        return results_df, rmse



def midas_compare(low_freq_series, model_specs, hf_lags, lf_lags, forecast_horizon,
                  start_date, end_date, plot_forecasts=False):
    """
    Compare multiple MIDAS models and plot RMSE + pseudo out-of-sample forecasts.

    Args:
        low_freq_series (pd.Series): The low-frequency target variable.
        model_specs (list of dict): Each dict must have keys:
            - 'name': label for the model
            - 'high_freq_series': list of high-frequency regressors
            - 'polys': list of polynomial weight function names
            - 'hf_lags' (optional): list of lag specs (e.g., ["3m"])
            - 'lf_lags' (optional): int, number of low-frequency lags
        hf_lags (list of str): Default HF lags
        lf_lags (int): Default LF lags
        forecast_horizon (int): Step-ahead forecast
        start_date (datetime)
        end_date (datetime)
        plot_forecasts (bool): Whether to plot forecast vs target for each model.

    Returns:
        dict: Keys = model names, values = (forecast_df, rmse)
    """
    rmse_data = []
    forecast_results = {}

    for spec in model_specs:
        name = spec["name"]
        hf_series = spec.get("high_freq_series", [])
        polys = spec.get("polys", [])

        model_hf_lags = spec.get("hf_lags", hf_lags[:len(hf_series)])
        model_lf_lags = spec.get("lf_lags", lf_lags)

        print(f"Fitting model: {name}")

        model = MIDAS(low_freq_series,
                      high_freq_series_list=hf_series,
                      hf_lags_list=model_hf_lags,
                      lf_lags=model_lf_lags)
        
        df_forecast, rmse = model.rolling_forecast(
            start_date=start_date,
            end_date=end_date,
            forecast_horizon=forecast_horizon,
            poly_list=polys,
            verbose=False
        )

        rmse_data.append((name, rmse))
        forecast_results[name] = (df_forecast, rmse)

    # RMSE bar plot
    rmse_df = pd.DataFrame(rmse_data, columns=["Model", "RMSE"]).set_index("Model")
    ax = rmse_df.plot(kind='bar', figsize=(6, 4), legend=False, color='steelblue')

    # Annotate each bar with its RMSE value
    for i, v in enumerate(rmse_df["RMSE"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

    plt.title("RMSE Comparison of MIDAS Models")
    plt.ylabel("RMSE")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    if plot_forecasts:
        plt.figure(figsize=(12, 5))
        reference_model = next(iter(forecast_results.values()))
        actual_series = reference_model[0]["targets"]

        plt.plot(actual_series.index, actual_series.values,
                label="Actual", color='black', linewidth=2, marker='o')

        styles = ['--', '-.', ':', (0, (3, 1, 1, 1))]
        colors = plt.cm.tab10.colors

        for idx, (name, (df_model, rmse)) in enumerate(forecast_results.items()):
            plt.plot(
                df_model.index,
                df_model["preds"],
                label=f"{name} (RMSE={rmse:.2f})",
                linestyle=styles[idx % len(styles)],
                color=colors[idx % len(colors)],
                marker='o'
            )

        plt.title("MIDAS Model Forecasts vs Actual")
        plt.xlabel("Date")
        plt.ylabel("GDP YoY")
        plt.legend()
        plt.grid(True)

        import matplotlib.dates as mdates
        quarter_end_months = [3, 6, 9, 12]
        locator = mdates.MonthLocator(bymonth=quarter_end_months)
        formatter = mdates.DateFormatter('%Y-%m')
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.show()

    return forecast_results