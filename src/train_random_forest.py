import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def main():
    print("Loading prepared data...")

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv",
        dtype=np.float64
    )
    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    )["Class"]

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train.to_numpy(dtype=np.float64),
        y_train
    )

    print("Training complete.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "random_forest.joblib"
    )

    print("\nModel saved:")
    print("models/random_forest.joblib")


if __name__ == "__main__":
    main()
