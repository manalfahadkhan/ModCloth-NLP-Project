import pandas as pd
from nltk.tokenize import TreebankWordTokenizer

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

text = df["review_text"].dropna().iloc[0]

tokenizer = TreebankWordTokenizer()

print("Original review:")
print(text)

print("\nNLTK tokens:")
print(tokenizer.tokenize(text))