# MIDAS: Mixed Data Sampling for Time-Series Nowcasting and Forecasting

A generalized and extensible Python package for estimating MIDAS regressions — designed for both **nowcasting** and **forecasting** using low- and high-frequency time series data.

This package builds upon and extends the ideas in the original `midaspy` package, offering:

- A flexible `MIDAS` class for end-to-end workflows
- Support for **exp-Almon** and **Beta** polynomial lag weighting
- Rolling forecast routines for pseudo real-time evaluation
- Compatibility with arbitrary combinations of high-frequency regressors

---

## 📚 Tutorials and Resources

For detailed tutorials and examples, refer to the following notebooks:

* [`tutorial/MIDAS_Tutorial.ipynb`](tutorial/MIDAS_Tutorial.ipynb): Demonstrates quarterly GDP nowcasting using a monthly leading indicator.

For more comprehensive documentation, slides, and research materials, visit the [BSE Forecast NLP repository](https://github.com/RenatoVassallo/BSE-ForecastNLP).

---

## 🚀 Features

- ✅ Generalized `MIDAS` class with:
  - `prepare_data`: frequency alignment + lag stacking
  - `fit`: nonlinear optimization of weighted regressors
  - `predict`: nowcast or forecast output
  - `rolling_forecast`: expanding-window backtesting

- ✅ Supports:
  - **Multiple frequency combinations**: Daily, Business Daily, Monthly, Quarterly and Annual
  - Arbitrary number of high-frequency regressors
  - **Nowcasting** (`alignment_lag=0`)
  - **Forecasting** (`alignment_lag ≥ 1`)

- ✅ Polynomial lag weighting:
  - `expalmon`: exponential Almon lag structure
  - `beta`: standard beta functional form

- ✅ Easy model comparison with `midas_compare` utility:
  - Runs and evaluates multiple specifications
  - Returns RMSEs and forecasts for side-by-side comparison

---

## 📦 Installation

Install the latest version:

```bash
pip install https://github.com/RenatoVassallo/MIDAS/releases/download/0.1.0/midas-0.1.0-py3-none-any.whl
```

## 📚 References

Ghysels, Santa-Clara & Valkanov (2006), Predicting Volatility: MIDAS Regressions.

## 📝 License

This project is licensed under the MIT License.