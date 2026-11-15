import pandas as pd
import spacy

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

nlp = spacy.load("en_core_web_sm")

text = df["review_text"].dropna().iloc[0]

doc = nlp(text.lower())

stop_words = nlp.Defaults.stop_words - {
    "no", "not", "never"
}

words = [token.text for token in doc if token.is_alpha]

filtered_words = [
    word for word in words
    if word not in stop_words
]

print("Original words:")
print(words)

print("\nAfter stop-word removal:")
print(filtered_words)