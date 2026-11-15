from pathlib import Path
import json
from gensim.models import Word2Vec, FastText

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "embeddings"
REPORT_DIR = ROOT / "reports"

models = {
    "cbow": Word2Vec.load(
        str(MODEL_DIR / "word2vec_cbow.model")
    ),

    "skipgram": Word2Vec.load(
        str(MODEL_DIR / "word2vec_skipgram.model")
    ),

    "fasttext": FastText.load(
        str(MODEL_DIR / "fasttext.model")
    )
}


def similar_words(model, word):

    try:
        return [
            {"word": w, "similarity": float(s)}
            for w, s in model.wv.most_similar(word, topn=8)
        ]

    except KeyError:
        return []


def cosine(model, word1, word2):

    try:
        return float(model.wv.similarity(word1, word2))

    except KeyError:
        return None


results = {
    "similar_words": {},
    "cosine_similarity": {},
    "analogy": {}
}

queries = [
    "dress",
    "fit",
    "fabric",
    "comfortable",
    "small"
]

for name, model in models.items():

    results["similar_words"][name] = {}

    for word in queries:

        results["similar_words"][name][word] = (
            similar_words(model, word)
        )

    try:

        results["analogy"][name] = [
            {"word": w, "similarity": float(s)}
            for w, s in model.wv.most_similar(
                positive=["dress", "small"],
                negative=["large"],
                topn=5
            )
        ]

    except KeyError:

        results["analogy"][name] = "Required word unavailable"


pairs = {
    "related": ["dress", "skirt"],
    "unrelated": ["dress", "shipping"]
}

for label, pair in pairs.items():

    results["cosine_similarity"][label] = {}

    for name, model in models.items():

        results["cosine_similarity"][label][name] = (
            cosine(model, pair[0], pair[1])
        )


with open(
    REPORT_DIR / "embedding_probes.json",
    "w"
) as file:

    json.dump(results, file, indent=4)

print(json.dumps(results, indent=4))