"""The end-to-end pipeline: load, chunk, index, retrieve, answer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from .chunking import Chunk, Chunker, chunk_statistics, get_chunker
from .config import RAGConfig
from .generation import Answer, Generator, get_generator
from .loaders import Document, corpus_statistics, load_corpus
from .retrieval import Retrieved, Retriever, build_retriever
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever


@dataclass
class BuildReport:
    """Timings and counts recorded while the pipeline was built."""

    documents: int
    chunks: int
    load_seconds: float
    chunk_seconds: float
    index_seconds: float
    corpus_stats: dict[str, float | int]
    chunk_stats: dict[str, float | int]

    @property
    def total_seconds(self) -> float:
        return self.load_seconds + self.chunk_seconds + self.index_seconds


class RAGPipeline:
    """Holds one configuration of the pipeline and answers questions with it."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        documents: Sequence[Document] | None = None,
        chunker: Chunker | None = None,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self._documents = list(documents) if documents is not None else None
        self._chunker = chunker
        self._retriever = retriever
        self._generator = generator
        self._chunks: list[Chunk] = []
        self.report: BuildReport | None = None

    # -- construction -----------------------------------------------------
    def build(self) -> "RAGPipeline":
        """Load the corpus, chunk it and index it. Returns ``self``."""
        started = time.perf_counter()
        if self._documents is None:
            self._documents = load_corpus(self.config.documents_dir)
        load_seconds = time.perf_counter() - started

        if self._chunker is None:
            self._chunker = get_chunker(self.config.chunking, self.config.chunking_params)

        started = time.perf_counter()
        self._chunks = self._chunker.chunk_corpus(self._documents)
        chunk_seconds = time.perf_counter() - started

        started = time.perf_counter()
        if self._retriever is None:
            self._retriever = build_retriever(
                self.config.retriever, self._chunks, self.config
            )
        elif not self._retriever.is_indexed:
            self._retriever.index(self._chunks)
        index_seconds = time.perf_counter() - started

        if self._generator is None:
            self._generator = get_generator(
                self.config.generation.backend, self.config.generation
            )

        self.report = BuildReport(
            documents=len(self._documents),
            chunks=len(self._chunks),
            load_seconds=load_seconds,
            chunk_seconds=chunk_seconds,
            index_seconds=index_seconds,
            corpus_stats=corpus_statistics(self._documents),
            chunk_stats=chunk_statistics(self._chunks),
        )
        return self

    # -- accessors --------------------------------------------------------
    @property
    def documents(self) -> list[Document]:
        if self._documents is None:
            raise RuntimeError("pipeline has not been built; call build() first")
        return self._documents

    @property
    def chunks(self) -> list[Chunk]:
        if not self._chunks:
            raise RuntimeError("pipeline has not been built; call build() first")
        return self._chunks

    @property
    def chunker(self) -> Chunker:
        if self._chunker is None:
            raise RuntimeError("pipeline has not been built; call build() first")
        return self._chunker

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            raise RuntimeError("pipeline has not been built; call build() first")
        return self._retriever

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            raise RuntimeError("pipeline has not been built; call build() first")
        return self._generator

    # -- use --------------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[Retrieved]:
        """Retrieve the top chunks for ``query``."""
        return self.retriever.search(query, k or self.config.top_k)

    def answer(self, query: str, k: int | None = None) -> tuple[Answer, list[Retrieved]]:
        """Retrieve and then generate. Returns the answer and its context."""
        contexts = self.retrieve(query, k)
        return self.generator.generate(query, contexts), contexts

    def describe(self) -> dict[str, object]:
        """A record of this configuration, saved alongside experiment results."""
        return {
            "chunking": self.config.chunking,
            "chunker": repr(self.chunker),
            "retriever_config": self.retriever.describe(),
            "generator": self.generator.name,
            "top_k": self.config.top_k,
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "chunk_stats": self.report.chunk_stats if self.report else {},
        }


def build_pipeline(config: RAGConfig | None = None, **kwargs: object) -> RAGPipeline:
    """Build and return a ready-to-use pipeline."""
    return RAGPipeline(config, **kwargs).build()  # type: ignore[arg-type]


class SharedCorpus:
    """Caches documents, chunks and indexes across a grid of configurations.

    The experiment runs six combinations (two chunking strategies by three
    retrievers). Encoding the corpus with a transformer is by far the most
    expensive step, so each chunking strategy is encoded once and the resulting
    dense index is shared by the dense and hybrid runs. Without this the
    experiment would encode the corpus four times instead of twice.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._documents = load_corpus(config.documents_dir)
        self._chunks: dict[str, list[Chunk]] = {}
        self._bm25: dict[str, BM25Retriever] = {}
        self._dense: dict[str, DenseRetriever] = {}

    @property
    def documents(self) -> list[Document]:
        return self._documents

    def chunks(self, strategy: str) -> list[Chunk]:
        if strategy not in self._chunks:
            chunker = get_chunker(strategy, self.config.chunking_params)
            self._chunks[strategy] = chunker.chunk_corpus(self._documents)
        return self._chunks[strategy]

    def bm25(self, strategy: str) -> BM25Retriever:
        if strategy not in self._bm25:
            retriever = BM25Retriever(self.config.bm25)
            retriever.index(self.chunks(strategy))
            self._bm25[strategy] = retriever
        return self._bm25[strategy]

    def dense(self, strategy: str) -> DenseRetriever:
        if strategy not in self._dense:
            retriever = DenseRetriever(self.config.dense)
            retriever.index(self.chunks(strategy))
            self._dense[strategy] = retriever
        return self._dense[strategy]

    def pipeline(self, strategy: str, retriever_name: str) -> RAGPipeline:
        """A built pipeline that reuses the cached indexes."""
        config = self.config.with_(chunking=strategy, retriever=retriever_name)
        retriever = build_retriever(
            retriever_name,
            self.chunks(strategy),
            config,
            shared_bm25=self.bm25(strategy) if retriever_name in {"bm25", "hybrid"} else None,
            shared_dense=self.dense(strategy) if retriever_name in {"dense", "hybrid"} else None,
        )
        return RAGPipeline(
            config,
            documents=self._documents,
            chunker=get_chunker(strategy, config.chunking_params),
            retriever=retriever,
        ).build()
