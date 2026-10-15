"""Data loading, cleaning, and feature engineering for telco churn.

The feature pipeline is a scikit-learn ColumnTransformer so the exact same
transformations run in the notebooks, in training, and in the Streamlit app.
Raw categorical columns are one-hot encoded, numeric columns are passed
through (and scaled only for the logistic regression model, which gets its
own pipeline in train.py).
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_month"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw Telco CSV and apply the documented cleaning steps.

    Cleaning steps:
    - TotalCharges ships as text with blank strings for brand new customers
      (tenure 0). Convert to numeric; blanks become NaN and are imputed later
      in the pipeline.
    - SeniorCitizen ships as 0/1; cast to a string category so it is one-hot
      encoded with the other categorical fields.
    - Churn is mapped to 1/0.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Currently one: charges_per_month.

    charges_per_month = TotalCharges / max(tenure, 1). For a customer whose
    monthly bill has crept up over time this runs higher than MonthlyCharges,
    which is a rough proxy for price increases over the account life.
    """
    out = df.copy()
    safe_tenure = out["tenure"].clip(lower=1)
    out["charges_per_month"] = out["TotalCharges"] / safe_tenure
    return out


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into the feature matrix and target series."""
    df = add_features(df)
    return df[ALL_FEATURES].copy(), df[TARGET].copy()


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Return the shared preprocessing transformer.

    scale_numeric=True adds median imputation plus standard scaling for the
    numeric block (used by logistic regression). The tree model uses
    scale_numeric=False: median imputation only, raw numeric values.
    """
    if scale_numeric:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate and customer count per level of a column, sorted by rate."""
    grouped = (
        df.groupby(column)[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "customers"})
        .sort_values("churn_rate", ascending=False)
    )
    grouped["churn_rate"] = grouped["churn_rate"].round(4)
    return grouped


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide with a zero guard, used for segment lift calculations."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator
