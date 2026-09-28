import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def main():
    model = joblib.load("models/random_forest.joblib")

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv",
        dtype=np.float64
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    )["Class"]

    probabilities = model.predict_proba(
        X_test.to_numpy(dtype=np.float64)
    )[:, 1]

    thresholds = np.arange(0.10, 1.00, 0.05)

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

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

    print("=== Random Forest Threshold Analysis ===")
    print(results_df.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        REPORTS_DIR / "random_forest_threshold_analysis.csv",
        index=False
    )

    print("\nReport saved:")
    print(
        "reports/random_forest_threshold_analysis.csv"
    )


if __name__ == "__main__":
    main()
