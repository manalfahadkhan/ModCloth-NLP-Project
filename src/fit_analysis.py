import pandas as pd
from collections import Counter

df = pd.read_csv(
    "data/processed/manal_cleaned.csv"
)

for category in ["small", "fit", "large"]:

    print("\nFit category:", category)

    data = df[df["fit"] == category]

    words = []

    for text in data["clean_text"].dropna():
        words.extend(text.split())

    print("Total reviews:", len(data))

    print("Common words:")
    print(Counter(words).most_common(15))