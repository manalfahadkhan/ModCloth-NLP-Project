from pathlib import Path
import argparse
import json
import pandas as pd
from gensim.models import Word2Vec, FastText

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "manal_cleaned.csv"
MODEL_DIR = ROOT / "models" / "embeddings"
REPORT_DIR = ROOT / "reports"


def load_sentences():

    df = pd.read_csv(DATA_PATH, low_memory=False)

    documents = df["clean_text"].fillna("").astype(str)
    documents = documents.str.strip()

    documents = documents[documents != ""].tolist()

    return [doc.split() for doc in documents if doc.split()]


def train_model(model_type, sentences):

    common = {
        "vector_size": 100,
        "window": 5,
        "min_count": 5,
        "workers": 4,
        "epochs": 3,
        "seed": 42
    }

    if model_type == "cbow":

        model = Word2Vec(
            sentences=sentences,
            sg=0,
            **common
        )

        model.save(str(MODEL_DIR / "word2vec_cbow.model"))

    elif model_type == "skipgram":

        model = Word2Vec(
            sentences=sentences,
            sg=1,
            **common
        )

        model.save(str(MODEL_DIR / "word2vec_skipgram.model"))

    elif model_type == "fasttext":

        model = FastText(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=5,
            workers=4,
            sg=0,
            epochs=3,
            seed=42,
            min_n=3,
            max_n=6,
            bucket=50000
        )

        model.save(str(MODEL_DIR / "fasttext.model"))

    return model


def run():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["cbow", "skipgram", "fasttext", "all"],
        default="all"
    )

    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences()

    if args.model == "all":
        models = ["cbow", "skipgram", "fasttext"]
    else:
        models = [args.model]

    results = {}

    for model_type in models:

        model = train_model(model_type, sentences)

        results[model_type] = {
            "vocabulary_size": len(model.wv),
            "vector_size": model.vector_size
        }

        print(model_type, len(model.wv))

    results["documents"] = len(sentences)

    with open(
        REPORT_DIR / "embedding_training_results.json",
        "w"
    ) as file:

        json.dump(results, file, indent=4)


if __name__ == "__main__":
    run()