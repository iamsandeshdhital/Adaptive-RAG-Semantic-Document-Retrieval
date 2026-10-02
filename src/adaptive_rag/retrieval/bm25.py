"""Okapi BM25, implemented directly rather than taken from a library.

For a query ``q`` and a chunk ``d`` the score is

    score(q, d) = sum over terms t in q of
                  IDF(t) * f(t, d) * (k1 + 1)
                  / ( f(t, d) + k1 * (1 - b + b * |d| / avgdl) )

where ``f(t, d)`` is the frequency of ``t`` in ``d``, ``|d|`` is the length of
``d`` in tokens and ``avgdl`` is the mean chunk length. The IDF uses the
standard probabilistic form with the ``+1`` inside the logarithm, which keeps
the value positive even for a term that appears in every chunk:

    IDF(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

Scoring walks an inverted index, so only chunks that share at least one term
with the query are touched.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

from ..config import BM25Config
from ..preprocess import preprocess
from .base import Retriever, Retrieved, rank_top_k


class BM25Retriever(Retriever):
    """Sparse keyword retrieval over chunk text."""

    name = "bm25"

    def __init__(self, config: BM25Config | None = None) -> None:
        super().__init__()
        self.config = config or BM25Config()
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._vocabulary_size = 0

    # -- indexing ---------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        return preprocess(
            text,
            remove_stopwords=self.config.remove_stopwords,
            apply_stemming=self.config.stem,
        )

    def _build(self) -> None:
        tokenized = [self._tokenize(chunk.text) for chunk in self._chunks]
        self._doc_lengths = [len(tokens) for tokens in tokenized]
        total_tokens = sum(self._doc_lengths)
        self._avgdl = total_tokens / len(tokenized) if tokenized else 0.0

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_index, tokens in enumerate(tokenized):
            for term, frequency in Counter(tokens).items():
                postings[term].append((doc_index, frequency))

        self._postings = dict(postings)
        self._vocabulary_size = len(self._postings)

        n_docs = len(tokenized)
        self._idf = {}
        for term, entries in self._postings.items():
            doc_frequency = len(entries)
            self._idf[term] = math.log(
                1.0 + (n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )

    # -- querying ---------------------------------------------------------
    def score_query(self, query: str) -> list[float]:
        """Return the BM25 score of every indexed chunk for ``query``."""
        self._require_index()
        scores = [0.0] * len(self._chunks)
        k1 = self.config.k1
        b = self.config.b
        avgdl = self._avgdl or 1.0

        for term in self._tokenize(query):
            entries = self._postings.get(term)
            if not entries:
                continue
            idf = self._idf[term]
            for doc_index, frequency in entries:
                length_norm = 1.0 - b + b * (self._doc_lengths[doc_index] / avgdl)
                denominator = frequency + k1 * length_norm
                if denominator <= 0.0:
                    continue
                scores[doc_index] += idf * frequency * (k1 + 1.0) / denominator
        return scores

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        return rank_top_k(self.score_query(query), self._chunks, k)

    # -- introspection ----------------------------------------------------
    def describe(self) -> dict[str, object]:
        summary = super().describe()
        summary.update(
            {
                "k1": self.config.k1,
                "b": self.config.b,
                "stopwords_removed": self.config.remove_stopwords,
                "stemming": self.config.stem,
                "vocabulary": self._vocabulary_size,
                "avg_chunk_tokens": round(self._avgdl, 1),
            }
        )
        return summary

    def term_idf(self, term: str) -> float:
        """IDF of a single raw term, used in the report to illustrate weighting."""
        self._require_index()
        tokens = self._tokenize(term)
        if not tokens:
            return 0.0
        return self._idf.get(tokens[0], 0.0)

    def explain(self, query: str, chunk_index: int) -> list[tuple[str, float]]:
        """Per-term contributions to the score of one chunk.

        Used for the error analysis in the discussion section: it shows which
        query terms actually drove a retrieval decision.
        """
        self._require_index()
        k1, b = self.config.k1, self.config.b
        avgdl = self._avgdl or 1.0
        length_norm = 1.0 - b + b * (self._doc_lengths[chunk_index] / avgdl)

        contributions: list[tuple[str, float]] = []
        for term in dict.fromkeys(self._tokenize(query)):
            entries = self._postings.get(term)
            if not entries:
                continue
            frequency = next((f for i, f in entries if i == chunk_index), 0)
            if frequency == 0:
                continue
            value = self._idf[term] * frequency * (k1 + 1.0) / (frequency + k1 * length_norm)
            contributions.append((term, round(value, 4)))
        contributions.sort(key=lambda pair: -pair[1])
        return contributions


def build_bm25(chunks: Sequence, config: BM25Config | None = None) -> BM25Retriever:
    """Convenience constructor that indexes in one call."""
    retriever = BM25Retriever(config)
    retriever.index(chunks)
    return retriever
