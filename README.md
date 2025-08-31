# Analytics-Portfolio

This repository contains my personal collection of analytics projects, notebooks, and data.

## Note on Python Environment

Please use the system Python environment instead of a virtual environment.

## Project-local Python launcher

A lightweight launcher is provided at the repository root as `./python`. It delegates
to the Codespace-managed Python interpreter and is useful when you want a single,
reproducible interpreter for scripts and notebooks in this Codespace:

```bash
# Run Python using the project launcher
./python -V
./python -m pip install -r requirements.txt
```

If you prefer to call the interpreter by name from anywhere, add the repo root to your PATH:

```bash
export PATH="$(pwd):$PATH"
# then simply run
python -V
```
## Contents

- **Notebooks** – Example Jupyter notebooks demonstrating data analysis and modeling steps.
  - `Master.ipynb` and `Showcase.ipynb` include exploration, model training, and evaluation routines.
- **Scripts** – Utility Python scripts.
  - `intradaydata.py` downloads intraday stock data using the `yfinance` library and stores it under `raw_data/intraday/`.
  - `patient_readmission_model.py` creates a synthetic healthcare dataset, cleans it, applies simple data augmentation to balance classes, and trains a model to predict hospital readmission risk.
  - `stock_market_analysis.py` loads sample NVDA prices and fits a regression model to forecast the next day's closing value.
  - `marketing_analytics.py` generates noisy customer data, performs RFM-based clustering, and evaluates an A/B test on conversion rates.
  - `supply_chain_optimization.py` creates synthetic inventory records, cleans them, forecasts demand, and computes EOQ-driven reorder suggestions under budget constraints.
- **website** – TypeScript and CSS code for a small portfolio site.
- **raw_data** – Sample datasets of daily and intraday stock prices used for experimentation.
  - `synthetic_patient_readmissions.csv` is the generated dataset used by `patient_readmission_model.py`.
  - `synthetic_marketing_data.csv` supports the marketing segmentation and testing example.
  - `synthetic_supply_chain_data.csv` contains the inventory and demand figures for the operations optimization demo.
  - `NVDA.csv`, `MRVL.csv`, and `INOD.csv` provide daily stock prices for the financial examples.
- **SQL** – A set of SQL queries targeting the Classic Models database for practice with selection and aggregation.
- **python.py** – A single script with many machine learning snippets including model fitting, evaluation metrics, and visualization.
- **Do Hoang Hai Long - Provisional Transcript - 20030239 - 5.pdf** – Academic transcript included for completeness.

Run the notebooks with Jupyter or execute the scripts directly to reproduce the analyses.

## Portfolio Website

The `website` folder contains a simple portfolio front end written in TypeScript with separate CSS styling.

### Run locally

1. `cd website`
2. Install dependencies with `npm install` (needed once).
3. Compile the TypeScript source using `npx tsc` (or run `npx --prefix website tsc -p website` from the repository root).
4. Open `index.html` in a browser to view the portfolio.

## Portfolio Website Plan
See [Portfolio_Site_Plan.md](Portfolio_Site_Plan.md) for guidance on presenting these projects in a web portfolio.
