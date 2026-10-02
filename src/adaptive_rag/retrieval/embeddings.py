"""Embedding backends for dense retrieval.

Two backends are provided behind one interface:

``SentenceTransformerBackend``
    A pretrained sentence-embedding model (``all-MiniLM-L6-v2`` by default).
    This is the backend used for the reported results.

``LsaBackend``
    Latent semantic analysis: TF-IDF followed by truncated SVD, fitted on the
    corpus being indexed. It needs only NumPy and scikit-learn, so the pipeline
    still runs where PyTorch cannot be installed. It is a real dense vector
    space, but it is corpus-fitted rather than pretrained, and it scores lower.

All backends return L2-normalised vectors, so an inner product is a cosine
similarity and the two search paths (NumPy and FAISS) agree.
"""

from __future__ import annotations

import importlib
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows untouched."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _quiet_model_loader() -> None:
    """Silence the model loader.

    Weight-loading progress bars and hub notices are noise for a corpus of a
    few dozen chunks, and they interleave with the experiment output. Each step
    is optional and guarded, because the logging entry points move between
    library versions.
    """
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for module_name in ("transformers.utils.logging", "huggingface_hub.utils.logging"):
        try:  # pragma: no cover - depends on the installed versions
            module = importlib.import_module(module_name)
            if hasattr(module, "set_verbosity_error"):
                module.set_verbosity_error()
            if hasattr(module, "disable_progress_bar"):
                module.disable_progress_bar()
        except Exception:
            pass
    try:  # pragma: no cover - depends on the installed versions
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    except Exception:
        pass


class EmbeddingBackend(ABC):
    """Turns text into unit-length dense vectors."""

    name: str = "base"
    #: True when the backend must see the corpus before it can encode.
    requires_fit: bool = False

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the output vectors."""

    def fit(self, corpus: Sequence[str]) -> "EmbeddingBackend":
        """Fit on the corpus. A no-op for pretrained backends."""
        return self

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts into a ``(len(texts), dim)`` array."""

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query into a ``(1, dim)`` array."""
        return self.encode([text])

    def describe(self) -> dict[str, object]:
        return {"embedding_backend": self.name, "dim": self.dim}


class SentenceTransformerBackend(EmbeddingBackend):
    """Pretrained transformer sentence encoder."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        _quiet_model_loader()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "sentence-transformers is not installed; install it with "
                "'pip install -r requirements-dense.txt' or pass "
                "--embeddings lsa to use the NumPy fallback"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = SentenceTransformer(model_name, device=device)
        # The accessor was renamed in sentence-transformers 6; support both.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        ) or self._model.get_sentence_embedding_dimension
        self._dim = int(get_dim())
        self.name = f"sentence-transformers:{model_name.split('/')[-1]}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def describe(self) -> dict[str, object]:
        summary = super().describe()
        summary["model_name"] = self.model_name
        return summary


class LsaBackend(EmbeddingBackend):
    """TF-IDF plus truncated SVD, fitted on the indexed corpus."""

    name = "lsa"
    requires_fit = True

    def __init__(self, n_components: int = 256, random_state: int = 0) -> None:
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("the lsa backend requires scikit-learn") from exc

        self._svd_class = TruncatedSVD
        self._requested_components = n_components
        self._random_state = random_state
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._svd = None
        self._dim = 0

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: Sequence[str]) -> "LsaBackend":
        tfidf = self._vectorizer.fit_transform(list(corpus))
        # SVD cannot keep more components than the smaller matrix dimension.
        n_components = max(2, min(self._requested_components, min(tfidf.shape) - 1))
        self._svd = self._svd_class(
            n_components=n_components, random_state=self._random_state
        )
        self._svd.fit(tfidf)
        self._dim = n_components
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("LsaBackend.fit must be called before encode")
        tfidf = self._vectorizer.transform(list(texts))
        return l2_normalise(self._svd.transform(tfidf))

    def describe(self) -> dict[str, object]:
        summary = super().describe()
        summary["requested_components"] = self._requested_components
        if self._svd is not None:
            summary["explained_variance"] = round(
                float(self._svd.explained_variance_ratio_.sum()), 4
            )
        return summary


def sentence_transformers_available() -> bool:
    """True when the optional transformer dependency can be imported."""
    try:  # pragma: no cover - environment dependent
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def get_embedding_backend(
    backend: str = "auto",
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    lsa_dim: int = 256,
    batch_size: int = 32,
) -> EmbeddingBackend:
    """Build an embedding backend by name.

    ``"auto"`` prefers the pretrained model and falls back to LSA if
    sentence-transformers is unavailable, so a fresh checkout runs either way.
    """
    key = backend.strip().lower()
    if key in {"st", "sentence-transformers", "sentence_transformers", "transformer"}:
        return SentenceTransformerBackend(model_name, batch_size=batch_size)
    if key == "lsa":
        return LsaBackend(n_components=lsa_dim)
    if key == "auto":
        if sentence_transformers_available():
            return SentenceTransformerBackend(model_name, batch_size=batch_size)
        return LsaBackend(n_components=lsa_dim)
    raise ValueError(
        f"unknown embedding backend {backend!r}; expected 'auto', "
        f"'sentence-transformers' or 'lsa'"
    )
