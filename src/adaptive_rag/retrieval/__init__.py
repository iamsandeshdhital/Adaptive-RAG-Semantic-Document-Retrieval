"""Retrieval strategies: BM25, dense and hybrid."""

from __future__ import annotations

from typing import Sequence

from ..chunking import Chunk
from ..config import RAGConfig
from .base import Retrieved, Retriever, rank_top_k
from .bm25 import BM25Retriever, build_bm25
from .dense import DenseRetriever, build_dense, faiss_available
from .embeddings import (
    EmbeddingBackend,
    LsaBackend,
    SentenceTransformerBackend,
    get_embedding_backend,
    sentence_transformers_available,
)
from .hybrid import HybridRetriever, build_hybrid, min_max_normalise

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "EmbeddingBackend",
    "HybridRetriever",
    "LsaBackend",
    "Retrieved",
    "Retriever",
    "SentenceTransformerBackend",
    "build_bm25",
    "build_dense",
    "build_hybrid",
    "build_retriever",
    "faiss_available",
    "get_embedding_backend",
    "get_retriever",
    "min_max_normalise",
    "rank_top_k",
    "sentence_transformers_available",
]


def get_retriever(
    name: str,
    config: RAGConfig | None = None,
    *,
    shared_bm25: BM25Retriever | None = None,
    shared_dense: DenseRetriever | None = None,
) -> Retriever:
    """Build an unindexed retriever by name.

    ``shared_bm25`` and ``shared_dense`` let the experiment runner reuse an
    already-indexed component, which avoids re-encoding the corpus when the
    hybrid retriever is built after its two parts.
    """
    config = config or RAGConfig()
    key = name.strip().lower()
    if key == "bm25":
        return shared_bm25 or BM25Retriever(config.bm25)
    if key == "dense":
        return shared_dense or DenseRetriever(config.dense)
    if key == "hybrid":
        return HybridRetriever(
            config.hybrid,
            bm25=shared_bm25,
            dense=shared_dense,
            bm25_config=config.bm25,
            dense_config=config.dense,
        )
    raise ValueError(
        f"unknown retriever {name!r}; expected 'bm25', 'dense' or 'hybrid'"
    )


def build_retriever(
    name: str,
    chunks: Sequence[Chunk],
    config: RAGConfig | None = None,
    **kwargs: object,
) -> Retriever:
    """Build a retriever by name and index it over ``chunks``."""
    retriever = get_retriever(name, config, **kwargs)  # type: ignore[arg-type]
    if not retriever.is_indexed or retriever.chunks != list(chunks):
        retriever.index(chunks)
    return retriever
