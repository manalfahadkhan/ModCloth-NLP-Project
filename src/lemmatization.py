import pandas as pd
import spacy

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

nlp = spacy.load("en_core_web_sm")

text = df["review_text"].dropna().iloc[0]

doc = nlp(text.lower())

lemmas = [
    token.lemma_
    for token in doc
    if token.is_alpha
]

print("Original review:")
print(text)

print("\nLemmatized words:")
print(lemmas)