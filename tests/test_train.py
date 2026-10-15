"""Smoke tests for training and evaluation on a small slice of the data."""

from pathlib import Path

import numpy as np
import pytest

from churn.evaluate import metrics_at_threshold
from churn.features import load_data, make_xy
from churn.train import build_models, split_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


@pytest.fixture(scope="module")
def small_xy():
    if not DATA_PATH.exists():
        pytest.skip("dataset not downloaded; run `python data/get_data.py`")
    df = load_data(DATA_PATH).head(800)
    return make_xy(df)


def test_train_smoke_and_metric_bounds(small_xy):
    X, y = small_xy
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)

    pipe = build_models()["logistic_regression"]
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    assert prob.shape == (len(X_test),)
    assert ((prob >= 0) & (prob <= 1)).all()

    m = metrics_at_threshold(y_test, prob, threshold=0.5)
    # Sanity bounds on a small slice: better than chance, not suspiciously perfect.
    assert 0.55 < m["roc_auc"] < 1.0
    assert 0.5 < m["accuracy"] < 1.0


def test_random_forest_trains(small_xy):
    X, y = small_xy
    X_train, X_test, y_train, _ = split_data(X, y)
    pipe = build_models()["random_forest"]
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    assert np.isfinite(prob).all()
