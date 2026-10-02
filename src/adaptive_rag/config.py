"""Central configuration for the Adaptive-RAG pipeline.

Every tunable number used anywhere in the project is declared here so that an
experiment can be described by a single object and reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# Repository layout ---------------------------------------------------------
# config.py lives at <root>/src/adaptive_rag/config.py, so the project root is
# three parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EVAL_PATH = DATA_DIR / "eval" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "results"

#: Sentence-embedding model used for dense retrieval. Small enough to run on a
#: laptop CPU (22M parameters, 384 dimensions).
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Fallback backend used when sentence-transformers is not installed. It is a
#: genuine dense vector space (TF-IDF followed by truncated SVD, i.e. latent
#: semantic analysis) but it is fitted on the corpus rather than pretrained.
FALLBACK_EMBEDDING_BACKEND = "lsa"


@dataclass(frozen=True)
class ChunkingConfig:
    """Parameters for both chunking strategies.

    The fixed-size defaults follow the project brief: 300-word chunks with a
    50-word overlap.
    """

    #: Words per chunk for the fixed-size strategy.
    chunk_size: int = 300
    #: Words repeated between consecutive fixed-size chunks.
    overlap: int = 50
    #: Paragraph chunks shorter than this are merged with the following one.
    min_paragraph_words: int = 80
    #: Paragraph chunks longer than this are split at sentence boundaries.
    max_paragraph_words: int = 350

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        if self.min_paragraph_words > self.max_paragraph_words:
            raise ValueError("min_paragraph_words must not exceed max_paragraph_words")


@dataclass(frozen=True)
class BM25Config:
    """Okapi BM25 parameters.

    ``k1`` controls how quickly the contribution of a term saturates as it
    repeats; ``b`` controls how strongly scores are normalised by chunk length.
    The defaults are the values in common use.
    """

    k1: float = 1.5
    b: float = 0.75
    #: Remove stopwords from queries and chunks before indexing.
    remove_stopwords: bool = True
    #: Apply the light suffix-stripping stemmer.
    stem: bool = True


@dataclass(frozen=True)
class DenseConfig:
    """Dense retrieval parameters."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    #: "auto" prefers sentence-transformers and falls back to "lsa"; the other
    #: accepted values are "sentence-transformers" and "lsa".
    backend: str = "auto"
    #: Number of dimensions retained by the LSA fallback.
    lsa_dim: int = 256
    #: Use a FAISS inner-product index when the library is available.
    use_faiss: bool = True
    #: Texts encoded per forward pass.
    batch_size: int = 32


@dataclass(frozen=True)
class HybridConfig:
    """Hybrid fusion parameters.

    ``alpha`` is the weight given to the dense score under score fusion, so
    ``alpha=0.0`` reproduces BM25 and ``alpha=1.0`` reproduces dense retrieval.
    """

    #: "score" for weighted min-max score fusion, "rrf" for reciprocal rank fusion.
    fusion: str = "score"
    alpha: float = 0.5
    #: Candidates drawn from each retriever before fusion.
    candidate_k: int = 20
    #: Smoothing constant of reciprocal rank fusion.
    rrf_k: int = 60


@dataclass(frozen=True)
class GenerationConfig:
    """Answer generation parameters.

    The default backend is extractive and needs no network access or API key,
    which keeps the whole pipeline runnable offline. Setting ``backend`` to
    "claude" uses the Anthropic API instead.
    """

    #: "extractive", "claude", or "auto" (Claude when an API key is present).
    backend: str = "extractive"
    #: Sampling parameters are not sent: the current Claude models reject
    #: temperature and top_p, and the answer should be as close to
    #: deterministic as the API allows in any case.
    model: str = "claude-opus-5"
    max_tokens: int = 1024
    #: Sentences kept by the extractive generator.
    max_sentences: int = 3


@dataclass(frozen=True)
class RAGConfig:
    """The complete description of one pipeline instance."""

    documents_dir: Path = DOCUMENTS_DIR
    #: "fixed" or "paragraph".
    chunking: str = "fixed"
    #: "bm25", "dense" or "hybrid".
    retriever: str = "bm25"
    #: Chunks passed to the generator.
    top_k: int = 5

    chunking_params: ChunkingConfig = field(default_factory=ChunkingConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def with_(self, **changes: object) -> "RAGConfig":
        """Return a copy with the given top-level fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]


#: Strategy names used by the experiment grid.
CHUNKING_STRATEGIES = ("fixed", "paragraph")
RETRIEVER_NAMES = ("bm25", "dense", "hybrid")
