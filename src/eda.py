import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum().sum())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== CLASS DISTRIBUTION ===")
print(df["Class"].value_counts())

print("\n=== CLASS PERCENTAGE ===")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

print("\n=== TRANSACTION AMOUNT ===")
print(df["Amount"].describe())

print("\n=== FRAUD TRANSACTION AMOUNT ===")
print(df[df["Class"] == 1]["Amount"].describe())

print("\n=== LEGITIMATE TRANSACTION AMOUNT ===")
print(df[df["Class"] == 0]["Amount"].describe())

df["Class"].value_counts().plot(kind="bar")

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()

plt.savefig("reports/class_distribution.png")
plt.close()

print("\nEDA completed.")
print("Chart saved: reports/class_distribution.png")
