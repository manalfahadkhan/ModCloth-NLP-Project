import pandas as pd
from collections import Counter

df = pd.read_csv(
    "data/processed/manal_cleaned.csv",
    low_memory=False
)

df["clean_text"] = df["clean_text"].fillna("").astype(str)
df["fit"] = df["fit"].fillna("").astype(str).str.lower().str.strip()

categories = ["small", "fit", "large"]

for category in categories:

    print("\nFit category:", category)

    data = df[df["fit"] == category]

    words = []

    for text in data["clean_text"].dropna():
        words.extend(text.split())

    print("Total reviews:", len(data))

    print("Most common words:")
    print(Counter(words).most_common(15))

    print("-" * 50)
