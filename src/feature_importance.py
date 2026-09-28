import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def main():
    print("Loading Random Forest model...")

    model = joblib.load(
        "models/random_forest.joblib"
    )

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\n=== Top 15 Features ===")
    print(
        feature_importance.head(15).to_string(
            index=False
        )
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_importance.to_csv(
        REPORTS_DIR / "feature_importance.csv",
        index=False
    )

    top_features = feature_importance.head(15)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "feature_importance.png",
        dpi=150
    )

    plt.close()

    print("\nReports saved:")
    print("reports/feature_importance.csv")
    print("reports/feature_importance.png")


if __name__ == "__main__":
    main()
