import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def main():
    print("Loading prepared data...")

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv",
        dtype=np.float64
    )
    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv",
        dtype=np.float64
    )

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Class"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Class"]

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train.to_numpy(dtype=np.float64), y_train)

    print("Training complete.")

    X_test_np = X_test.to_numpy(dtype=np.float64)

    y_pred = model.predict(X_test_np)

    # Use the stable decision scores directly.
    decision_scores = model.decision_function(X_test_np)

    # Convert decision scores to probabilities safely.
    y_prob = np.empty_like(decision_scores)

    positive = decision_scores >= 0
    y_prob[positive] = 1.0 / (
        1.0 + np.exp(-decision_scores[positive])
    )

    negative_scores = decision_scores[~positive]
    exp_scores = np.exp(negative_scores)

    y_prob[~positive] = exp_scores / (1.0 + exp_scores)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "logistic_regression.joblib"
    )

    print("\nModel saved:")
    print("models/logistic_regression.joblib")


if __name__ == "__main__":
    main()
