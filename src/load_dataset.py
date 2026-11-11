
import pandas as pd

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())