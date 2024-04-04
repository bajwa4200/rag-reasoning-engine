"""Local embedding backends (no external APIs)."""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Protocol


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class EmbeddingModel(Protocol):
    def fit(self, documents: list[str]) -> None: ...
    def embed(self, text: str) -> list[float]: ...


class BaseEmbedding(ABC):
    @abstractmethod
    def fit(self, documents: list[str]) -> None: ...

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class TfidfEmbedding(BaseEmbedding):
    """Simple bag-of-words TF-IDF vectors."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []

    def fit(self, documents: list[str]) -> None:
        tokens_per_doc = [_tokenize(d) for d in documents]
        df: Counter[str] = Counter()
        for tokens in tokens_per_doc:
            df.update(set(tokens))

        vocab = sorted(df.keys())
        self._vocab = {term: i for i, term in enumerate(vocab)}
        n = max(1, len(documents))
        self._idf = [math.log((1 + n) / (1 + df[term])) + 1.0 for term in vocab]

    def embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        if not self._vocab:
            return []
        tf = Counter(tokens)
        vec = [0.0] * len(self._vocab)
        for term, count in tf.items():
            if term in self._vocab:
                idx = self._vocab[term]
                vec[idx] = (count / max(1, len(tokens))) * self._idf[idx]
        return vec


class HashEmbedding(BaseEmbedding):
    """Feature hashing into a fixed-dimensional vector."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def fit(self, documents: list[str]) -> None:
        del documents  # stateless

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self.dim
            sign = 1.0 if int(digest[:2], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
