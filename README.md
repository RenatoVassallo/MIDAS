# MIDAS

Mixed-frequency **nowcasting and forecasting** tools for Python. `MIDAS` turns a panel of
monthly indicators and quarterly targets into a full pseudo-real-time workflow: it
reconstructs exactly what was knowable at any past origin (respecting publication
delays), runs a family of models through one uniform interface, backtests them without
look-ahead, and reports point forecasts, bootstrapped bands, evaluation tables and
publication-quality plots. It also ships the teaching materials it grew out of.

## Quick install

Latest release (`0.2.1`), no build step:

```bash
pip install https://github.com/RenatoVassallo/MIDAS/releases/download/0.2.1/midas-0.2.1-py3-none-any.whl
```

With `uv`:

```bash
uv add https://github.com/RenatoVassallo/MIDAS/releases/download/0.2.1/midas-0.2.1-py3-none-any.whl
```

Python `3.11` is required. Installing from source is described at the bottom.

## What is inside

**One contract, many models.** Every model implements `BaseNowcaster`:
`fit(info) -> nowcast(info) -> NowcastResult`, where `info` is an `InformationSet` (the
masked panel knowable at one origin). So benchmarks, a factor model and a machine-learning
model are all driven the same way, and combining or backtesting them is uniform.

- **Data and real time**
  - `MetadataPanel`: monthly + quarterly frames with per-column metadata (frequency,
    economic group, publication delay).
  - `RealtimeEngine`: reconstructs the no-look-ahead information set at any origin; begin,
    middle and end-of-month origins select genuinely different data as releases clear.

- **Models**
  - Benchmarks: `RandomWalkNowcaster`, `HistoricalMeanNowcaster`, `ARNowcaster`,
    `BridgeNowcaster`, `ADLMIDASNowcaster`.
  - `DFMNowcaster`: dynamic factor model (wraps `statsmodels` `DynamicFactorMQ`), with
    optional group-block factors, native ragged-edge handling, a news decomposition and a
    COVID-as-missing option.
  - `SparseMIDASNowcaster`: sparse-group LASSO MIDAS (Babii, Ghysels and Striaukas), built
    on the standalone `SparseGroupLasso` estimator and a Legendre dictionary.
  - `CombinationNowcaster`: equal or performance-weighted ensembles.

- **Backtesting and evaluation**
  - `run_backtest` (intra-quarter origins) and `run_horizon_backtest` (h = 0, 1, ...),
    both pseudo-real-time and expanding-window.
  - `horizon_bands` / `bootstrap_quantiles`: bootstrapped predictive bands from a model's
    own realised errors (no normality assumed).
  - `horizon_rmse_table` and `horizon_table_latex`: matched-sample RMSE tables with
    Diebold-Mariano stars; `rmse`, `mae`, `dm_test`, `evaluation_table`.
  - `nowcast_plots`: vintage tracking (GDPNow style), fan charts, nowcast-evolution and
    model-comparison figures.

- **MIDAS estimators and weights**
  - `BetaMIDASRegressor`, `beta_weights`, `almon_weights`, `legendre_dictionary`,
    `align_monthly_to_quarter`, `stack_high_freq_lags`.

- **Teaching helpers**
  - `PanelBuilder` (processed teaching panels), `use_aer_style` and `PALETTE`.

## Minimal example

```python
import pandas as pd
from MIDAS import (MetadataPanel, VariableMeta, RealtimeEngine,
                   DFMNowcaster, RandomWalkNowcaster, run_backtest)

# 1. Wrap your data: date-indexed monthly and quarterly frames, one VariableMeta per column
#    (monthly rows dated at the first of the month; quarterly at the first day of the end month).
panel = MetadataPanel.from_frames(
    monthly, quarterly,
    [VariableMeta(column="ip",  frequency="M", group="activity", publication_delay_days=45),
     VariableMeta(column="gdp", frequency="Q", group="activity", publication_delay_days=60)],
)

# 2. Reconstruct exactly what was known on 15 May 2019 (nothing released later is visible)
engine = RealtimeEngine(panel)
info = engine.information_set("2019-05-15", target="gdp", target_period="2019-06-01")

# 3. Fit any model and read a nowcast (point + optional density)
res = DFMNowcaster(factors=2).fit(info).nowcast(info)
print(res.mean, res.std)

# 4. Backtest a set of models, pseudo-real-time, expanding window
results = run_backtest(
    panel, "gdp",
    {"RW": RandomWalkNowcaster(), "DFM": DFMNowcaster(factors=2)},
    eval_start="2015-01-01",
)
```

## Install from source

With `uv` (the `dev` group, notebook tooling and tests, installs by default):

```bash
uv sync
```

Without `uv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install ipykernel jupyter nbclient nbconvert nbformat
```

If `xgboost` or `shap` are unavailable, the ML helpers fall back automatically to
scikit-learn boosted trees and permutation importance.

## References

- Ghysels, Santa-Clara and Valkanov (2004), *The MIDAS touch: mixed data sampling regressions*
- Andreou, Ghysels and Kourtellos (2010), *Regression models with mixed sampling frequencies*
- Mariano and Murasawa (2003), *A new coincident index of business cycles*
- Giannone, Reichlin and Small (2008), *Nowcasting: the real-time informational content of macroeconomic data*
- Banbura and Modugno (2014), *Maximum likelihood estimation of factor models on data sets with arbitrary pattern of missing data*
- Babii, Ghysels and Striaukas (2022), *Machine learning time series regressions with an application to nowcasting*, JBES
- Foroni, Marcellino and Schumacher (2015), *U-MIDAS: MIDAS regressions with unrestricted lag polynomials*
- Diebold and Mariano (1995); Harvey, Leybourne and Newbold (1997), *Tests of equal predictive accuracy*
