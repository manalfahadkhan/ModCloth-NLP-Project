
import pandas as pd

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

print("Dataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFit categories:")
print(df["fit"].value_counts())

print("\nQuality ratings:")
print(df["quality"].value_counts())
