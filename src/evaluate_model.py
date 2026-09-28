import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def main():
    print("Loading model and test data...")

    model = joblib.load("models/logistic_regression.joblib")

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv",
        dtype=np.float64
    )
    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    )["Class"]

    X_test_np = X_test.to_numpy(dtype=np.float64)

    y_pred = model.predict(X_test_np)
    scores = model.decision_function(X_test_np)

    probabilities = np.empty_like(scores)

    positive = scores >= 0
    probabilities[positive] = 1.0 / (
        1.0 + np.exp(-scores[positive])
    )

    negative_scores = scores[~positive]
    exp_scores = np.exp(negative_scores)

    probabilities[~positive] = (
        exp_scores / (1.0 + exp_scores)
    )

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    print("=== Confusion Matrix ===")

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print("\nTrue Negatives :", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives :", tp)

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\n=== Ranking Metrics ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame({
        "metric": [
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            "roc_auc",
            "pr_auc",
        ],
        "value": [
            tn,
            fp,
            fn,
            tp,
            roc_auc,
            pr_auc,
        ],
    })

    metrics.to_csv(
        REPORTS_DIR / "model_evaluation.csv",
        index=False
    )

    print("\nReport saved:")
    print("reports/model_evaluation.csv")


if __name__ == "__main__":
    main()
