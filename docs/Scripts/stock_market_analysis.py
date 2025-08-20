#!/usr/bin/env python
"""Simple stock analysis and forecast example.

This script loads sample NVDA stock prices, engineers lag features, and fits a
linear regression model to predict the next day's closing price.  It showcases
basic time-series preparation and evaluation steps for financial data.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def load_prices(path: Path) -> pd.DataFrame:
    """Read NVDA prices and compute daily returns."""
    df = pd.read_csv(path, parse_dates=["Date"], thousands=",")
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = (
        df[numeric_cols].replace({",": ""}, regex=True).astype(float)
    )
    df = df.sort_values("Date").reset_index(drop=True)
    df["Return"] = df["Close"].pct_change()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged closing-price features for forecasting."""
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    return df.dropna().reset_index(drop=True)


def forecast(df: pd.DataFrame) -> None:
    """Fit and evaluate a one-step-ahead regression model."""
    features = [f"lag_{i}" for i in range(1, 6)]
    X = df[features]
    y = df["Close"]
    # Use the last 30 days as a hold-out set
    X_train, X_test = X[:-30], X[-30:]
    y_train, y_test = y[:-30], y[-30:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Mean absolute error: {mae:.2f}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "raw_data" / "NVDA.csv"
    df = load_prices(data_path)
    df = build_features(df)
    forecast(df)


if __name__ == "__main__":
    main()
