#!/usr/bin/env python
"""Synthetic marketing analytics with segmentation and A/B testing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.stats.proportion import proportions_ztest

RNG = np.random.default_rng(seed=42)


def generate_data(path: Path, n: int = 1000) -> pd.DataFrame:
    """Generate a synthetic marketing dataset with noisy entries."""
    customers = np.arange(1, n + 1)
    ages = RNG.integers(18, 70, size=n)
    genders = RNG.choice(["M", "F", "male", "female"], size=n)
    recency = RNG.integers(1, 365, size=n).astype(float)
    frequency = RNG.integers(1, 20, size=n).astype(float)
    monetary = RNG.gamma(2.0, 100.0, size=n)

    group = RNG.choice(["A", "B"], size=n)
    conv_prob = {"A": 0.05, "B": 0.08}
    purchase = [RNG.random() < conv_prob[g] for g in group]

    data = pd.DataFrame(
        {
            "customer_id": customers,
            "age": ages,
            "gender": genders,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "group": group,
            "purchase": purchase,
        }
    )

    # Inject missing values and outliers
    miss_idx = RNG.choice(n, size=20, replace=False)
    data.loc[miss_idx, "recency"] = -1  # invalid recency
    neg_idx = RNG.choice(n, size=20, replace=False)
    data.loc[neg_idx, "monetary"] *= -1  # negative spend

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load data and fix basic quality issues."""
    df = pd.read_csv(path)
    df["recency"] = df["recency"].apply(lambda x: np.nan if x < 0 else x)
    df["recency"].fillna(df["recency"].median(), inplace=True)
    df["monetary"] = df["monetary"].abs()
    df["gender"] = df["gender"].str.upper().str[0]
    return df


def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Perform KMeans clustering on RFM features."""
    rfm = df[["recency", "frequency", "monetary"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["segment"] = kmeans.fit_predict(rfm)
    return df


def ab_test(df: pd.DataFrame) -> None:
    """Run a z-test on conversion rates between groups."""
    summary = df.groupby("group")["purchase"].agg(["sum", "count"])
    successes = summary["sum"].to_numpy()
    trials = summary["count"].to_numpy()
    stat, pval = proportions_ztest(successes, trials)
    rates = successes / trials
    print("Conversion rates:\n", rates)
    print(f"Z-statistic: {stat:.3f}, p-value: {pval:.3f}")


if __name__ == "__main__":
    DATA_PATH = Path("raw_data/synthetic_marketing_data.csv")
    df = generate_data(DATA_PATH)
    df = load_and_clean(DATA_PATH)
    df = segment_customers(df)
    print(df["segment"].value_counts())
    ab_test(df)
