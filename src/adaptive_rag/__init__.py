"""Adaptive-RAG: semantic document retrieval.

A compact retrieval-augmented generation pipeline built to compare retrieval
strategies (BM25, dense and hybrid) and chunking strategies (fixed-size and
paragraph-based) on the same document collection.

Authors: Anurag Jha and Sandesh Dhital.
"""

from .chunking import (
    Chunk,
    Chunker,
    FixedSizeChunker,
    ParagraphChunker,
    chunk_statistics,
    get_chunker,
)
from .config import (
    BM25Config,
    ChunkingConfig,
    DenseConfig,
    GenerationConfig,
    HybridConfig,
    RAGConfig,
)
from .generation import Answer, ExtractiveGenerator, Generator, get_generator
from .loaders import Document, corpus_statistics, load_corpus, load_document
from .pipeline import RAGPipeline, SharedCorpus, build_pipeline
from .retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retrieved,
    Retriever,
    build_retriever,
)

__version__ = "1.0.0"
__authors__ = ("Anurag Jha", "Sandesh Dhital")

__all__ = [
    "Answer",
    "BM25Config",
    "BM25Retriever",
    "Chunk",
    "Chunker",
    "ChunkingConfig",
    "DenseConfig",
    "DenseRetriever",
    "Document",
    "ExtractiveGenerator",
    "FixedSizeChunker",
    "GenerationConfig",
    "Generator",
    "HybridConfig",
    "HybridRetriever",
    "ParagraphChunker",
    "RAGConfig",
    "RAGPipeline",
    "Retrieved",
    "Retriever",
    "SharedCorpus",
    "__authors__",
    "__version__",
    "build_pipeline",
    "build_retriever",
    "chunk_statistics",
    "corpus_statistics",
    "get_chunker",
    "get_generator",
    "load_corpus",
    "load_document",
]
