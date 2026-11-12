import pandas as pd

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

print("Total reviews:", len(df))

print("\nFit distribution:")
print(df["fit"].value_counts())

print("\nFit percentages:")
print(df["fit"].value_counts(normalize=True) * 100)

print("\nAverage quality:")
print(df["quality"].mean())

print("\nQuality summary:")
print(df["quality"].describe())

print("\nTop categories:")
print(df["category"].value_counts().head(10))