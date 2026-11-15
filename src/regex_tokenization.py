import pandas as pd
import re

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

text = df["review_text"].dropna().iloc[0]

tokens = re.findall(r"\b\w+\b", text)

print("Original review:")
print(text)

print("\nRegex tokens:")
print(tokens)