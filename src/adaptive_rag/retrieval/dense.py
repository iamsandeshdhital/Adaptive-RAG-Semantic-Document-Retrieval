"""Dense retrieval by cosine similarity over sentence embeddings.

Because every vector is L2-normalised at encoding time, cosine similarity is
just an inner product. Two search paths are available and return the same
ranking:

* a NumPy matrix product, which is exact and entirely adequate for a corpus of
  a few hundred chunks;
* a FAISS ``IndexFlatIP``, used when the library is installed. This is also an
  exact index, so it is a demonstration of the FAISS API rather than an
  approximation; the approximate FAISS index types only pay off at a corpus
  size far beyond this project.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import DenseConfig
from .base import Retriever, Retrieved, rank_top_k
from .embeddings import EmbeddingBackend, get_embedding_backend


def faiss_available() -> bool:
    """True when FAISS can be imported."""
    try:  # pragma: no cover - environment dependent
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


class DenseRetriever(Retriever):
    """Embedding-based retrieval with an exact inner-product search."""

    name = "dense"

    def __init__(
        self,
        config: DenseConfig | None = None,
        *,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DenseConfig()
        self._backend = backend
        self._matrix: np.ndarray | None = None
        self._faiss_index = None
        self._using_faiss = False

    @property
    def backend(self) -> EmbeddingBackend:
        """The embedding backend, constructed lazily from the config."""
        if self._backend is None:
            self._backend = get_embedding_backend(
                self.config.backend,
                model_name=self.config.model_name,
                lsa_dim=self.config.lsa_dim,
                batch_size=self.config.batch_size,
            )
        return self._backend

    @property
    def using_faiss(self) -> bool:
        return self._using_faiss

    @property
    def matrix(self) -> np.ndarray:
        """The ``(n_chunks, dim)`` matrix of chunk embeddings."""
        self._require_index()
        assert self._matrix is not None
        return self._matrix

    def _build(self) -> None:
        texts = [chunk.text for chunk in self._chunks]
        backend = self.backend
        if backend.requires_fit:
            backend.fit(texts)
        self._matrix = np.ascontiguousarray(backend.encode(texts), dtype=np.float32)

        self._using_faiss = False
        self._faiss_index = None
        if self.config.use_faiss and faiss_available():
            import faiss

            index = faiss.IndexFlatIP(self._matrix.shape[1])
            index.add(self._matrix)
            self._faiss_index = index
            self._using_faiss = True

    def score_query(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` against every chunk."""
        self._require_index()
        vector = self.backend.encode_query(query).astype(np.float32)
        assert self._matrix is not None
        return self._matrix @ vector[0]

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        self._require_index()
        assert self._matrix is not None
        k = min(k, len(self._chunks))
        if k <= 0:
            return []

        if self._using_faiss and self._faiss_index is not None:
            vector = np.ascontiguousarray(
                self.backend.encode_query(query), dtype=np.float32
            )
            scores, indices = self._faiss_index.search(vector, k)
            results: list[Retrieved] = []
            for rank, (position, score) in enumerate(
                zip(indices[0], scores[0]), start=1
            ):
                if position < 0:
                    continue
                results.append(
                    Retrieved(
                        chunk=self._chunks[int(position)],
                        score=float(score),
                        rank=rank,
                    )
                )
            return results

        # Cosine similarity can legitimately be negative, so zero scores are
        # not dropped here the way they are for BM25.
        return rank_top_k(
            self.score_query(query).tolist(), self._chunks, k, drop_zero=False
        )

    def describe(self) -> dict[str, object]:
        summary = super().describe()
        summary.update(self.backend.describe())
        summary["index"] = "faiss:IndexFlatIP" if self._using_faiss else "numpy:matmul"
        return summary


def build_dense(
    chunks: Sequence,
    config: DenseConfig | None = None,
    *,
    backend: EmbeddingBackend | None = None,
) -> DenseRetriever:
    """Convenience constructor that indexes in one call."""
    retriever = DenseRetriever(config, backend=backend)
    retriever.index(chunks)
    return retriever
