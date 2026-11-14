
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "manal_cleaned.csv"
MODEL_DIR = ROOT / "models" / "count_based"
REPORT_DIR = ROOT / "reports"


def load_documents():

    df = pd.read_csv(DATA_PATH, low_memory=False)

    documents = df["clean_text"].fillna("").astype(str)
    documents = documents.str.strip()

    documents = documents[documents != ""]

    return documents.tolist()


def calculate_sparsity(matrix):

    total = matrix.shape[0] * matrix.shape[1]

    if total == 0:
        return 0.0

    zeros = total - matrix.nnz

    return (zeros / total) * 100


def run():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents()

    print("Total documents:", len(documents))

    results = {}

    # PART A1: BAG OF WORDS

    print("\nTraining Bag of Words...")

    bow = CountVectorizer(
        dtype=np.int32
    )

    bow_matrix = bow.fit_transform(documents)

    joblib.dump(
        bow,
        MODEL_DIR / "bow_vectorizer.pkl"
    )

    results["bow"] = {
        "shape": list(bow_matrix.shape),
        "vocabulary_size": len(bow.vocabulary_),
        "sparsity": calculate_sparsity(bow_matrix)
    }

    print("BoW completed!")
    print("Shape:", bow_matrix.shape)

    del bow_matrix

    # PART A2: TF-IDF

    print("\nTraining TF-IDF...")

    tfidf = TfidfVectorizer(
        dtype=np.float32
    )

    tfidf_matrix = tfidf.fit_transform(documents)

    joblib.dump(
        tfidf,
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    features = np.array(tfidf.get_feature_names_out())

    weights = np.asarray(
        tfidf_matrix.mean(axis=0)
    ).ravel()

    highest = weights.argsort()[-10:][::-1]

    lowest = weights.argsort()[:10]

    results["tfidf"] = {
        "shape": list(tfidf_matrix.shape),
        "vocabulary_size": len(tfidf.vocabulary_),
        "sparsity": calculate_sparsity(tfidf_matrix),

        "highest_weight_words": [
            {
                "word": str(features[i]),
                "weight": float(weights[i])
            }
            for i in highest
        ],

        "lowest_weight_words": [
            {
                "word": str(features[i]),
                "weight": float(weights[i])
            }
            for i in lowest
        ]
    }

    print("TF-IDF completed!")
    print("Shape:", tfidf_matrix.shape)

    # PART A3: N-GRAM ANALYSIS

    print("\nStarting N-gram analysis...")

    ngram_results = {}

    for n, name in [
        (1, "unigram"),
        (2, "bigram"),
        (3, "trigram")
    ]:

        print(f"\nProcessing {name}...")

        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            min_df=2,
            max_features=100000,
            dtype=np.int32
        )

        matrix = vectorizer.fit_transform(documents)

        ngram_results[name] = {
            "shape": list(matrix.shape),
            "features": len(vectorizer.vocabulary_),
            "sparsity": calculate_sparsity(matrix)
        }

        print(name, "completed!")
        print("Features:", len(vectorizer.vocabulary_))
        print(
            "Sparsity:",
            calculate_sparsity(matrix)
        )

        del matrix
        del vectorizer

    results["ngrams"] = ngram_results

    # PART A4: EXAMPLE REVIEW

    print("\nCalculating example review TF-IDF weights...")

    sample_index = next(
        (
            i for i, doc in enumerate(documents)
            if len(doc.split()) >= 8
        ),
        None
    )

    if sample_index is not None:

        row = tfidf_matrix.getrow(sample_index)

        order = np.argsort(row.data)[::-1][:8]

        sample_words = [
            {
                "word": str(features[row.indices[j]]),
                "weight": float(row.data[j])
            }
            for j in order
        ]

        results["example_review"] = documents[sample_index]

        results["example_review_weights"] = sample_words

    else:

        results["example_review"] = "No suitable review found"
        results["example_review_weights"] = []

    # SAVE RESULTS

    with open(
        REPORT_DIR / "count_based_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nAll Count-Based experiments completed!")

    print("\nFinal Results:")

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    run()
