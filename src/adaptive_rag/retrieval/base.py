"""Common interface shared by the three retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from ..chunking import Chunk


@dataclass(frozen=True)
class Retrieved:
    """One scored chunk returned by a retriever."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class Retriever(ABC):
    """A retriever maps a query to a ranked list of chunks.

    Implementations must be indexed before use. ``search`` is expected to be
    cheap enough to call once per query in a timing loop, so any expensive
    preparation belongs in ``index``.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._indexed = False

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    def index(self, chunks: Sequence[Chunk]) -> "Retriever":
        """Build whatever structures ``search`` needs. Returns ``self``."""
        if not chunks:
            raise ValueError("cannot index an empty chunk collection")
        self._chunks = list(chunks)
        self._build()
        self._indexed = True
        return self

    @abstractmethod
    def _build(self) -> None:
        """Subclass hook called by :meth:`index`."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        """Return the ``k`` highest scoring chunks for ``query``."""

    def _require_index(self) -> None:
        if not self._indexed:
            raise RuntimeError(f"{type(self).__name__} must be indexed before searching")

    def describe(self) -> dict[str, object]:
        """Configuration summary recorded with experiment results."""
        return {"retriever": self.name, "chunks": len(self._chunks)}


def rank_top_k(
    scores: Sequence[float],
    chunks: Sequence[Chunk],
    k: int,
    *,
    drop_zero: bool = True,
) -> list[Retrieved]:
    """Turn a score array into the top ``k`` results.

    Ties are broken by chunk order so that repeated runs of the same query
    return an identical ranking, which the evaluation relies on.
    """
    if k <= 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    results: list[Retrieved] = []
    for rank, position in enumerate(order[:k], start=1):
        score = float(scores[position])
        if drop_zero and score <= 0.0:
            continue
        results.append(Retrieved(chunk=chunks[position], score=score, rank=rank))
    return results
