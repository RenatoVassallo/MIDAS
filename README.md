# MIDAS

Teaching-friendly Python tools for mixed-frequency nowcasting and forecasting, built around a single coherent workflow:

1. motivate mixed-frequency problems visually,
2. benchmark simple baselines,
3. estimate restricted MIDAS models,
4. compare modern ML alternatives,
5. augment forecasts with text features and evaluate decisions.

## Library highlights

- `MIDAS.PanelBuilder`: builds the processed monthly and quarterly teaching panels
- `MIDAS.BetaMIDASRegressor`: restricted Beta-MIDAS estimator with optional quarterly controls
- `MIDAS.rolling_forecast`: generic expanding/rolling backtest helper for panel data
- `MIDAS.align_monthly_to_quarter` and `MIDAS.stack_high_freq_lags`: mixed-frequency alignment helpers
- `MIDAS.rmse`, `MIDAS.mae`, `MIDAS.dm_test`: lightweight forecast evaluation tools
- `MIDAS.use_aer_style` and `MIDAS.PALETTE`: plotting style shared across all notebooks


## Installation

Python `3.11` is required.

With `uv`:

```bash
uv sync --group dev
```

Without `uv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install ipykernel jupyter nbclient nbconvert nbformat
```

Optional ML extras:

If `xgboost` or `shap` are unavailable, the notebooks fall back automatically
to scikit-learn boosted trees and permutation importance.


## References

- Andreou, Ghysels, and Kourtellos (2010), *Regression Models with Mixed Sampling Frequencies*
- Foroni, Marcellino, and Schumacher (2015), *U-MIDAS: MIDAS Regressions with Unrestricted Lag Polynomials*
- Ghysels, Kvedaras, and Zemlys (2016), *Mixed Frequency Data Sampling Regression Models: The R Package midasr*