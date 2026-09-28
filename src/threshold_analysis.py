import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def sigmoid(scores):
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

    return probabilities


def main():
    model = joblib.load("models/logistic_regression.joblib")

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv",
        dtype=np.float64
    )
    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    )["Class"]

    scores = model.decision_function(
        X_test.to_numpy(dtype=np.float64)
    )

    probabilities = sigmoid(scores)

    thresholds = np.arange(0.10, 1.00, 0.05)

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append({
            "threshold": round(threshold, 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fraud_predictions": int(predictions.sum()),
        })

    results_df = pd.DataFrame(results)

    print("=== Threshold Analysis ===")
    print(results_df.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        REPORTS_DIR / "threshold_analysis.csv",
        index=False
    )

    print("\nReport saved:")
    print("reports/threshold_analysis.csv")


if __name__ == "__main__":
    main()
