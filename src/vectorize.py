
from pathlib import Path
import joblib
import numpy as np
from gensim.models import Word2Vec, FastText

ROOT = Path(__file__).resolve().parents[1]

COUNT_DIR = ROOT / "models" / "count_based"
EMBED_DIR = ROOT / "models" / "embeddings"


def vectorize_document(raw_document, representation="tfidf"):

    text = str(raw_document).strip()
    tokens = text.split()

    if representation in ["bow", "tfidf"]:

        if representation == "bow":
            filename = "bow_vectorizer.pkl"
        else:
            filename = "tfidf_vectorizer.pkl"

        vectorizer = joblib.load(COUNT_DIR / filename)

        return vectorizer.transform([text])

    if representation == "cbow":

        model = Word2Vec.load(
            str(EMBED_DIR / "word2vec_cbow.model")
        )

    elif representation == "skipgram":

        model = Word2Vec.load(
            str(EMBED_DIR / "word2vec_skipgram.model")
        )

    elif representation == "fasttext":

        model = FastText.load(
            str(EMBED_DIR / "fasttext.model")
        )

    else:

        raise ValueError("Invalid representation")

    vectors = [
        model.wv[token]
        for token in tokens
        if token in model.wv.key_to_index
        or representation == "fasttext"
    ]

    if not vectors:

        return np.zeros(
            (1, model.vector_size),
            dtype=np.float32
        )

    return np.mean(vectors, axis=0, keepdims=True)


if __name__ == "__main__":

    example = "comfortable fabric beautiful dress"

    print("Testing TF-IDF:")
    print(vectorize_document(example, "tfidf").shape)

    print("Testing BoW:")
    print(vectorize_document(example, "bow").shape)
