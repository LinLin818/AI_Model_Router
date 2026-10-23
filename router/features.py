"""Turn a prompt string into a feature vector.

Two backends:
  * "tfidf"       - word n-grams + a few handcrafted signals. No downloads, fast, good baseline.
  * "embeddings"  - sentence-transformers embeddings + the same handcrafted signals.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

_CODE = re.compile(r"```|\bdef \b|\bclass \b|\bfunction\b|\bimport \b|#include|\bSELECT\b|=>|[{}]")
_REASON = re.compile(
    r"\b(prove|derive|step[- ]by[- ]step|why|optimi[sz]e|trade-?offs?|design|analy[sz]e|"
    r"compare|debug|refactor|algorithm|complexity|justify|generali[sz]e)\b",
    re.I,
)


class HandFeatures(BaseEstimator, TransformerMixin):
    """Cheap structural signals: length, code, math/reasoning keywords, etc."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []

        for t in X:
            n_chars = max(len(t), 1)
            rows.append(
                [
                    np.log1p(len(t)),
                    np.log1p(len(t.split())),
                    t.count("?"),
                    t.count("\n"),
                    sum(ch.isdigit() for ch in t) / n_chars,
                    float(bool(_CODE.search(t))),
                    len(_REASON.findall(t)),
                ]
            )
        return np.asarray(rows, dtype=float)


class EmbeddingFeatures(BaseEstimator, TransformerMixin):
    """Sentence-transformers embeddings (model is loaded lazily, never pickled)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def _get_model(self):
        if getattr(self, "_model", None) is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


    def transform(self, X):
        return self._get_model().encode(list(X), normalize_embeddings=True, show_progress_bar=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state


def build_featurizer(backend: str = "tfidf", embedding_model: str = "all-MiniLM-L6-v2"):
    hand = Pipeline([("raw", HandFeatures()), ("scale", StandardScaler())])
    if backend == "tfidf":
        text = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
    elif backend == "embeddings":
        text = EmbeddingFeatures(embedding_model)
    else:
        raise ValueError(f"unknown backend: {backend!r}")
    return FeatureUnion([("text", text), ("hand", hand)])