"""Train churn models and persist the best one plus a metrics report.

Two models are trained on the same split:
- Logistic regression with class_weight="balanced" (scaled numeric features).
- Random forest with class_weight="balanced_subsample" (raw numeric features).

Class imbalance (about 27 percent churners) is handled with class weights in
both models rather than resampling, so the training set stays the real
customer mix and predicted probabilities stay comparable across models.

The model with the higher test ROC AUC is saved to models/churn_model.joblib
together with its name, and full metrics for both models go to
reports/metrics.json.

Run from the repo root:

    python -m churn.train --data data/raw/telco_churn.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn.evaluate import (
    grouped_permutation_importance,
    metrics_at_threshold,
    threshold_sweep,
)
from churn.features import build_preprocessor, churn_rate_by, load_data, make_xy

SEGMENT_COLUMNS = [
    "Contract",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "PaymentMethod",
    "PaperlessBilling",
    "tenure_band",
]

RANDOM_STATE = 42
TEST_SIZE = 0.25


def build_models() -> dict[str, Pipeline]:
    """Return the two model pipelines keyed by display name."""
    logreg = Pipeline(
        steps=[
            ("prep", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    forest = Pipeline(
        steps=[
            ("prep", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {"logistic_regression": logreg, "random_forest": forest}


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with a fixed seed."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_and_evaluate(data_path: str | Path) -> dict:
    """Train both models, evaluate on the held-out test set, return metrics."""
    df = load_data(data_path)
    X, y = make_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = metrics_at_threshold(y_test, prob, threshold=0.5)
        fitted[name] = pipe

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipe = fitted[best_name]
    best_prob = best_pipe.predict_proba(X_test)[:, 1]

    report = {
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "overall_churn_rate": float(y.mean()),
        "test_churn_rate": float(y_test.mean()),
        "models": results,
        "best_model": best_name,
        "best_model_threshold_sweep": threshold_sweep(y_test, best_prob).to_dict(
            orient="records"
        ),
    }

    drivers = grouped_permutation_importance(best_pipe, X_test, y_test)
    report["best_model_drivers"] = drivers.to_dict(orient="records")

    seg_df = df.copy()
    seg_df["tenure_band"] = pd.cut(
        seg_df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    segment_rates = {
        col: churn_rate_by(seg_df, col).reset_index().to_dict(orient="records")
        for col in SEGMENT_COLUMNS
    }

    return {
        "report": report,
        "best_pipeline": best_pipe,
        "best_name": best_name,
        "drivers": drivers,
        "segment_rates": segment_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train telco churn models.")
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--model-out", default="models/churn_model.joblib")
    parser.add_argument("--metrics-out", default="reports/metrics.json")
    args = parser.parse_args()

    outcome = train_and_evaluate(args.data)
    report = outcome["report"]

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": outcome["best_pipeline"],
            "model_name": outcome["best_name"],
            "drivers": outcome["drivers"],
            "segment_rates": outcome["segment_rates"],
            "overall_churn_rate": report["overall_churn_rate"],
        },
        model_path,
    )

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2))

    print(f"Rows: {report['n_rows']} (train {report['n_train']}, test {report['n_test']})")
    for name, m in report["models"].items():
        print(
            f"{name}: AUC {m['roc_auc']:.3f}  accuracy {m['accuracy']:.3f}  "
            f"precision {m['precision']:.3f}  recall {m['recall']:.3f}"
        )
    print(f"Best model: {report['best_model']} -> {model_path}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
