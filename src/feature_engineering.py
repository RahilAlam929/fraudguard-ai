import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")


def main():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Class"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Class"]

    print("=== Feature Engineering Check ===")
    print(f"Training features: {X_train.shape}")
    print(f"Testing features:  {X_test.shape}")

    print("\n=== Feature Columns ===")
    print(X_train.columns.tolist())

    print("\n=== Training Class Distribution ===")
    print(y_train.value_counts())

    print("\n=== Testing Class Distribution ===")
    print(y_test.value_counts())

    print("\n=== Feature Integrity ===")
    print("Training missing values:", X_train.isnull().sum().sum())
    print("Testing missing values:", X_test.isnull().sum().sum())

    print("\n=== Class Weight Baseline ===")

    normal_count = (y_train == 0).sum()
    fraud_count = (y_train == 1).sum()
    total = len(y_train)

    class_weight_normal = total / (2 * normal_count)
    class_weight_fraud = total / (2 * fraud_count)

    print(f"Normal class weight: {class_weight_normal:.6f}")
    print(f"Fraud class weight:  {class_weight_fraud:.6f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feature_stats = pd.DataFrame({
        "feature": X_train.columns,
        "train_mean": X_train.mean().values,
        "train_std": X_train.std().values,
        "train_min": X_train.min().values,
        "train_max": X_train.max().values,
    })

    feature_stats.to_csv(
        REPORT_DIR / "feature_statistics.csv",
        index=False
    )

    summary = pd.DataFrame({
        "metric": [
            "training_samples",
            "testing_samples",
            "features",
            "training_normal",
            "training_fraud",
            "testing_normal",
            "testing_fraud",
            "normal_class_weight",
            "fraud_class_weight",
        ],
        "value": [
            len(X_train),
            len(X_test),
            X_train.shape[1],
            normal_count,
            fraud_count,
            (y_test == 0).sum(),
            (y_test == 1).sum(),
            class_weight_normal,
            class_weight_fraud,
        ],
    })

    summary.to_csv(
        REPORT_DIR / "phase3_summary.csv",
        index=False
    )

    print("\nReports saved:")
    print("reports/feature_statistics.csv")
    print("reports/phase3_summary.csv")

    print("\nFeature engineering check complete.")


if __name__ == "__main__":
    main()
