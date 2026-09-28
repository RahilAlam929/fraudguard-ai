import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


RAW_DATA = Path("data/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def main():
    print("Loading dataset...")
    df = pd.read_csv(RAW_DATA)

    print(f"Original shape: {df.shape}")

    # Remove duplicate transactions
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Removed duplicates: {duplicate_count}")
    print(f"Shape after deduplication: {df.shape}")

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified split keeps fraud ratio consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale only Time and Amount
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )

    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    # Create output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save fitted scaler for inference
    joblib.dump(
        scaler,
        MODELS_DIR / "feature_scaler.joblib"
    )

    print("Scaler saved: models/feature_scaler.joblib")

    # Save prepared datasets
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("\nData preparation complete.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())


if __name__ == "__main__":
    main()
