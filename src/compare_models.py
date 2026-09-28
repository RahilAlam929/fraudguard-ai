import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def evaluate_model(model_path, model_name):
    model = joblib.load(model_path)

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv",
        dtype=np.float64
    )
    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    )["Class"]

    X_test_np = X_test.to_numpy(dtype=np.float64)

    y_pred = model.predict(X_test_np)
    y_prob = model.predict_proba(X_test_np)[:, 1] \
        if hasattr(model, "predict_proba") \
        else None

    return {
        "model": model_name,
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, y_prob
        ),
        "pr_auc": average_precision_score(
            y_test, y_prob
        ),
        "false_positives": int(
            ((y_test == 0) & (y_pred == 1)).sum()
        ),
        "false_negatives": int(
            ((y_test == 1) & (y_pred == 0)).sum()
        ),
    }


def main():
    print("Evaluating models...")

    logistic = evaluate_model(
        "models/logistic_regression.joblib",
        "Logistic Regression"
    )

    random_forest = evaluate_model(
        "models/random_forest.joblib",
        "Random Forest"
    )

    comparison = pd.DataFrame([
        logistic,
        random_forest,
    ])

    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False
    )

    print("\nReport saved:")
    print("reports/model_comparison.csv")


if __name__ == "__main__":
    main()
