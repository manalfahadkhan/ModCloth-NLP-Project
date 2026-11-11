
import pandas as pd
from nltk.stem import PorterStemmer

df = pd.read_csv(
    "data/processed/manal_cleaned.csv",
    low_memory=False
)

df["clean_text"] = df["clean_text"].fillna("").astype(str)

valid_reviews = df[df["clean_text"].str.strip() != ""]

if len(valid_reviews) == 0:
    print("No valid reviews found!")
else:

    stemmer = PorterStemmer()

    text = valid_reviews["clean_text"].iloc[0]

    words = text.split()

    stemmed_words = [
        stemmer.stem(word) for word in words
    ]

    print("Original cleaned words:")
    print(words)

    print("\nStemmed words:")
    print(stemmed_words)
