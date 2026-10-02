"""Hybrid retrieval: combining the sparse and dense rankings.

Two fusion rules are implemented.

**Weighted score fusion** (``fusion="score"``). BM25 scores are unbounded while
cosine similarities lie in [-1, 1], so the two cannot be added directly. Each
score vector is first min-max normalised onto [0, 1] across the corpus, then
combined as

    combined = alpha * dense_norm + (1 - alpha) * bm25_norm

so ``alpha=0`` reproduces BM25, ``alpha=1`` reproduces dense retrieval, and
``alpha=0.5`` weights them equally.

**Reciprocal rank fusion** (``fusion="rrf"``). Scores are discarded and only
positions are used, which makes the rule scale-free and removes the need for
normalisation:

    combined = sum over retrievers of 1 / (rrf_k + rank)

Each retriever contributes its top ``candidate_k`` chunks. Score fusion is the
default because it can express a preference between the two signals; RRF is
included because it is the more robust choice when score distributions are not
comparable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import BM25Config, DenseConfig, HybridConfig
from .base import Retriever, Retrieved, rank_top_k
from .bm25 import BM25Retriever
from .dense import DenseRetriever


def min_max_normalise(scores: np.ndarray) -> np.ndarray:
    """Map scores onto [0, 1]. A constant vector becomes all zeros."""
    scores = np.asarray(scores, dtype=np.float64)
    lowest = float(scores.min())
    highest = float(scores.max())
    spread = highest - lowest
    if spread <= 1e-12:
        return np.zeros_like(scores)
    return (scores - lowest) / spread


class HybridRetriever(Retriever):
    """Fuses a :class:`BM25Retriever` and a :class:`DenseRetriever`."""

    name = "hybrid"

    def __init__(
        self,
        config: HybridConfig | None = None,
        *,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        bm25_config: BM25Config | None = None,
        dense_config: DenseConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or HybridConfig()
        if not 0.0 <= self.config.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if self.config.fusion not in {"score", "rrf"}:
            raise ValueError(
                f"unknown fusion rule {self.config.fusion!r}; expected 'score' or 'rrf'"
            )
        self.bm25 = bm25 or BM25Retriever(bm25_config)
        self.dense = dense or DenseRetriever(dense_config)

    def _build(self) -> None:
        # Re-indexing a component that already holds these chunks is wasteful,
        # and re-encoding the corpus is the expensive part, so it is skipped.
        if not self.bm25.is_indexed or self.bm25.chunks != self._chunks:
            self.bm25.index(self._chunks)
        if not self.dense.is_indexed or self.dense.chunks != self._chunks:
            self.dense.index(self._chunks)

    # -- fusion rules -----------------------------------------------------
    def _fuse_scores(self, query: str, alpha: float) -> np.ndarray:
        bm25_scores = np.asarray(self.bm25.score_query(query), dtype=np.float64)
        dense_scores = np.asarray(self.dense.score_query(query), dtype=np.float64)
        return alpha * min_max_normalise(dense_scores) + (
            1.0 - alpha
        ) * min_max_normalise(bm25_scores)

    def _fuse_ranks(self, query: str) -> np.ndarray:
        candidate_k = min(self.config.candidate_k, len(self._chunks))
        rrf_k = self.config.rrf_k
        position_of = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}
        combined = np.zeros(len(self._chunks), dtype=np.float64)

        for retriever in (self.bm25, self.dense):
            for result in retriever.search(query, candidate_k):
                combined[position_of[result.chunk_id]] += 1.0 / (rrf_k + result.rank)
        return combined

    def score_query(self, query: str, alpha: float | None = None) -> np.ndarray:
        """Fused score for every chunk."""
        self._require_index()
        if self.config.fusion == "rrf":
            return self._fuse_ranks(query)
        return self._fuse_scores(
            query, self.config.alpha if alpha is None else float(alpha)
        )

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        scores = self.score_query(query)
        return rank_top_k(scores.tolist(), self._chunks, k, drop_zero=False)

    def search_with_alpha(self, query: str, k: int, alpha: float) -> list[Retrieved]:
        """Search with a one-off ``alpha``, used by the sensitivity sweep."""
        self._require_index()
        if self.config.fusion == "rrf":
            raise ValueError("alpha has no effect under reciprocal rank fusion")
        scores = self._fuse_scores(query, alpha)
        return rank_top_k(scores.tolist(), self._chunks, k, drop_zero=False)

    def describe(self) -> dict[str, object]:
        summary = super().describe()
        summary.update(
            {
                "fusion": self.config.fusion,
                "alpha": self.config.alpha,
                "candidate_k": self.config.candidate_k,
                "rrf_k": self.config.rrf_k,
                "components": [self.bm25.describe(), self.dense.describe()],
            }
        )
        return summary


def build_hybrid(
    chunks: Sequence,
    config: HybridConfig | None = None,
    *,
    bm25: BM25Retriever | None = None,
    dense: DenseRetriever | None = None,
) -> HybridRetriever:
    """Convenience constructor that indexes in one call."""
    retriever = HybridRetriever(config, bm25=bm25, dense=dense)
    retriever.index(chunks)
    return retriever
