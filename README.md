# Analytics-Portfolio

This repository contains my personal collection of analytics projects, notebooks, and data.

## Contents

- **Notebooks** – Example Jupyter notebooks demonstrating data analysis and modeling steps.
  - `Master.ipynb` and `Showcase.ipynb` include exploration, model training, and evaluation routines.
- **Scripts** – Utility Python scripts.
  - `intradaydata.py` downloads intraday stock data using the `yfinance` library and stores it under `raw_data/intraday/`.
  - `patient_readmission_model.py` creates a synthetic healthcare dataset, cleans it, applies simple data augmentation to balance classes, and trains a model to predict hospital readmission risk.
  - `stock_market_analysis.py` loads sample NVDA prices and fits a regression model to forecast the next day's closing value.
  - `marketing_analytics.py` generates noisy customer data, performs RFM-based clustering, and evaluates an A/B test on conversion rates.
  - `supply_chain_optimization.py` creates synthetic inventory records, cleans them, forecasts demand, and computes EOQ-driven reorder suggestions under budget constraints.
- **raw_data** – Sample datasets of daily and intraday stock prices used for experimentation.
  - `synthetic_patient_readmissions.csv` is the generated dataset used by `patient_readmission_model.py`.
  - `synthetic_marketing_data.csv` supports the marketing segmentation and testing example.
  - `synthetic_supply_chain_data.csv` contains the inventory and demand figures for the operations optimization demo.
  - `NVDA.csv`, `MRVL.csv`, and `INOD.csv` provide daily stock prices for the financial examples.
- **SQL** – A set of SQL queries targeting the Classic Models database for practice with selection and aggregation.
- **python.py** – A single script with many machine learning snippets including model fitting, evaluation metrics, and visualization.
- **Do Hoang Hai Long - Provisional Transcript - 20030239 - 5.pdf** – Academic transcript included for completeness.

Run the notebooks with Jupyter or execute the scripts directly to reproduce the analyses.

## Portfolio Website Plan
See [Portfolio_Site_Plan.md](Portfolio_Site_Plan.md) for guidance on presenting these projects in a web portfolio.
