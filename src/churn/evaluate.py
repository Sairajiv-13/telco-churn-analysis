"""Evaluation helpers: metrics at a threshold, threshold sweep, importance."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Binary classification metrics for a fixed decision threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def threshold_sweep(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Precision and recall across thresholds, for the trade-off discussion."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    rows = [metrics_at_threshold(y_true, y_prob, t) for t in thresholds]
    return pd.DataFrame(rows)


def grouped_permutation_importance(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance grouped back to the original input columns.

    The pipeline one-hot encodes categorical columns, so raw permutation
    importance on the input frame is the honest view: permute one original
    column at a time and measure the drop in ROC AUC.
    """
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )
    out = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return out.reset_index(drop=True)
