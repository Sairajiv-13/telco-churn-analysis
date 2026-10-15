"""Tests for the feature pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    load_data,
    make_xy,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip("dataset not downloaded; run `python data/get_data.py`")
    return load_data(DATA_PATH)


def test_load_data_row_count_and_target(raw):
    assert len(raw) == 7043
    assert raw["Churn"].isin([0, 1]).all()
    assert 0.20 < raw["Churn"].mean() < 0.35


def test_total_charges_is_numeric(raw):
    assert pd.api.types.is_numeric_dtype(raw["TotalCharges"])
    # Blank TotalCharges only occur for tenure 0 customers (11 known rows).
    assert raw["TotalCharges"].isna().sum() == 11


def test_feature_pipeline_shape_and_dtypes(raw):
    X, y = make_xy(raw)
    assert list(X.columns) == ALL_FEATURES
    assert len(X) == len(y) == len(raw)
    prep = build_preprocessor(scale_numeric=False)
    Xt = prep.fit_transform(X)
    n_expected = len(NUMERIC_FEATURES) + sum(
        raw[c].nunique() for c in CATEGORICAL_FEATURES
    )
    assert Xt.shape == (len(raw), n_expected)
    assert np.isfinite(np.asarray(Xt).astype(float)).all()


def test_scaled_pipeline_output_is_finite(raw):
    X, _ = make_xy(raw)
    prep = build_preprocessor(scale_numeric=True)
    Xt = prep.fit_transform(X.head(500))
    assert np.isfinite(np.asarray(Xt).astype(float)).all()
