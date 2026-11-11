
import pandas as pd
import re

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

print("Total reviews:", len(df))

df["review_text"] = df["review_text"].fillna("").astype(str)

valid_reviews = df[
    df["review_text"].str.strip() != ""
]

print("Reviews containing actual text:", len(valid_reviews))

if len(valid_reviews) == 0:
    print("No customer review text found in this dataset.")
else:

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^a-z\s']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df["clean_text"] = df["review_text"].apply(clean_text)

    original = valid_reviews.iloc[0]["review_text"]
    cleaned = clean_text(original)

    print("\nOriginal review:")
    print(original)

    print("\nCleaned review:")
    print(cleaned)

    df.to_csv(
        "data/processed/manal_cleaned.csv",
        index=False
    )

    print("\nFile saved successfully!")
