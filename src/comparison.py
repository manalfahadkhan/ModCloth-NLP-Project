
import pandas as pd

df = pd.read_csv(
    "data/processed/manal_cleaned.csv",
    low_memory=False
)

df["review_text"] = df["review_text"].fillna("")
df["clean_text"] = df["clean_text"].fillna("")

valid_reviews = df[
    (df["review_text"].str.strip() != "") &
    (df["clean_text"].str.strip() != "")
]

print("Total valid reviews:", len(valid_reviews))

for i in range(min(5, len(valid_reviews))):

    print("\nOriginal Review:")
    print(valid_reviews["review_text"].iloc[i])

    print("\nCleaned Review:")
    print(valid_reviews["clean_text"].iloc[i])

    print("-" * 50)
