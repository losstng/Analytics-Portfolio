#!/usr/bin/env python
"""Synthetic patient readmission model.

This script generates a noisy healthcare dataset, cleans it, and trains a
logistic regression model to predict whether a patient will be readmitted
within 30 days of discharge.  It demonstrates a simple workflow for data
cleaning, feature engineering, and model evaluation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RNG = np.random.default_rng(seed=42)


def generate_data(path: Path, n: int = 1000) -> pd.DataFrame:
    """Create a synthetic patient dataset with intentional issues."""
    data = pd.DataFrame(
        {
            "patient_id": np.arange(1, n + 1),
            "age": RNG.integers(20, 90, size=n).astype(float),
            "gender": RNG.choice(
                ["Male", "Female", "Femle", "male ", "F", "M"], size=n
            ),
            "length_of_stay": RNG.integers(1, 30, size=n).astype(float),
            "num_lab_procedures": RNG.integers(1, 100, size=n).astype(float),
            "num_medications": RNG.integers(1, 50, size=n),
            "num_diagnoses": RNG.integers(1, 10, size=n),
            "previous_admissions": RNG.poisson(1, size=n),
            "glucose_level": RNG.integers(70, 200, size=n).astype(float),
            "readmitted": RNG.binomial(1, 0.2, size=n),
        }
    )

    # Inject missing values
    for col in ["age", "length_of_stay", "num_lab_procedures", "glucose_level"]:
        data.loc[data.sample(frac=0.1, random_state=RNG.integers(0, 1e6)).index, col] = np.nan

    # Introduce negative and zero stays
    neg_idx = data.sample(frac=0.05, random_state=RNG.integers(0, 1e6)).index
    data.loc[neg_idx, "length_of_stay"] *= -1
    zero_idx = data.sample(frac=0.02, random_state=RNG.integers(0, 1e6)).index
    data.loc[zero_idx, "length_of_stay"] = 0

    # Mix glucose units: some values recorded in mmol/L instead of mg/dL
    mmol_idx = data.sample(frac=0.2, random_state=RNG.integers(0, 1e6)).index
    data.loc[mmol_idx, "glucose_level"] = data.loc[mmol_idx, "glucose_level"] / 18

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load dataset, fix common data quality issues."""
    df = pd.read_csv(path)

    # Fix gender typos and casing
    df["gender"] = df["gender"].str.strip().str.upper()
    df["gender"] = df["gender"].replace({"FEMLE": "FEMALE", "F": "FEMALE", "M": "MALE"})
    df["gender"] = df["gender"].str.title()

    # Length of stay: make positive and replace zeros with median
    df["length_of_stay"] = df["length_of_stay"].abs()
    median_stay = df.loc[df["length_of_stay"] > 0, "length_of_stay"].median()
    df.loc[df["length_of_stay"] == 0, "length_of_stay"] = median_stay

    # Convert glucose values recorded in mmol/L to mg/dL
    df.loc[df["glucose_level"] < 25, "glucose_level"] = (
        df.loc[df["glucose_level"] < 25, "glucose_level"] * 18
    )

    return df


def augment_training_data(
    X: pd.DataFrame, y: pd.Series, numeric_features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority class by jittering numeric features."""
    train_df = X.copy()
    train_df["target"] = y.values
    minority = train_df[train_df["target"] == 1]
    majority = train_df[train_df["target"] == 0]
    n_needed = len(majority) - len(minority)
    if n_needed <= 0:
        return X, y

    samples = minority.sample(
        n=n_needed, replace=True, random_state=RNG.integers(0, 1e6)
    ).copy()
    for col in numeric_features:
        std = train_df[col].std()
        samples[col] += RNG.normal(0, 0.05 * std, size=len(samples))

    augmented = pd.concat([train_df, samples], ignore_index=True)
    X_aug = augmented.drop(columns="target")
    y_aug = augmented["target"]
    return X_aug, y_aug


def build_model(df: pd.DataFrame) -> None:
    """Train and evaluate a logistic regression model."""
    y = df["readmitted"]
    X = df.drop(["readmitted", "patient_id"], axis=1)

    numeric_features = [
        "age",
        "length_of_stay",
        "num_lab_procedures",
        "num_medications",
        "num_diagnoses",
        "previous_admissions",
        "glucose_level",
    ]
    categorical_features = ["gender"]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Augment the training set to balance classes
    X_train, y_train = augment_training_data(X_train, y_train, numeric_features)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"ROC AUC: {auc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, preds))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "raw_data" / "synthetic_patient_readmissions.csv"

    if not data_path.exists():
        generate_data(data_path)

    df = load_and_clean(data_path)
    build_model(df)


if __name__ == "__main__":
    main()
