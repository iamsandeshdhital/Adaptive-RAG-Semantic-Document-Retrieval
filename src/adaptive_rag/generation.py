"""The generation stage: turning retrieved chunks into an answer.

Two backends are available.

``ExtractiveGenerator`` (the default)
    Selects the sentences from the retrieved chunks that best match the query
    and returns them with citations. It needs no API key and no network, so the
    whole pipeline can be run and graded offline. Retrieval quality is what the
    experiments measure, and an extractive reader keeps the answer traceable to
    the retrieved text.

``ClaudeGenerator`` (optional)
    Sends the retrieved chunks to the Claude API as grounded context. Enable it
    with ``--generator claude`` once ``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from .config import GenerationConfig
from .preprocess import STOPWORDS, split_sentences, stem, tokenize
from .retrieval.base import Retrieved

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages supplied by "
    "the user. Cite the passages you use with bracketed numbers such as [1]. "
    "If the passages do not contain the answer, reply exactly: "
    "'The provided documents do not answer this question.' "
    "Answer in no more than four sentences and do not add information from "
    "outside the passages."
)

NO_CONTEXT_MESSAGE = "No relevant passage was retrieved for this question."
NOT_ANSWERABLE_MESSAGE = "The provided documents do not answer this question."

#: Shortest sentence the extractive generator will quote. Fragments below this
#: are headings, list labels or captions rather than statements, and the length
#: normalisation would otherwise rank them above real sentences.
MIN_ANSWER_TOKENS = 6


def _normalise_for_comparison(text: str) -> str:
    """Lowercase, collapse whitespace and drop punctuation, for equality tests."""
    return " ".join(tokenize(text))


@dataclass(frozen=True)
class Answer:
    """A generated answer together with the evidence it was built from."""

    text: str
    backend: str
    citations: list[str] = field(default_factory=list)
    contexts: list[Retrieved] = field(default_factory=list)
    #: Populated when a backend falls back to another one.
    note: str = ""

    def format_citations(self) -> str:
        return "\n".join(
            f"[{i}] {item.chunk.display_source()} ({item.chunk.chunk_id})"
            for i, item in enumerate(self.contexts, start=1)
        )


def build_context_block(contexts: Sequence[Retrieved]) -> str:
    """Render retrieved chunks as the numbered passage list sent to a model."""
    parts = []
    for i, item in enumerate(contexts, start=1):
        parts.append(
            f"[{i}] Source: {item.chunk.display_source()}\n{item.chunk.text}"
        )
    return "\n\n".join(parts)


class Generator(ABC):
    """Produces an :class:`Answer` from a query and retrieved context."""

    name: str = "base"

    @abstractmethod
    def generate(self, query: str, contexts: Sequence[Retrieved]) -> Answer:
        """Answer ``query`` using only ``contexts``."""


class ExtractiveGenerator(Generator):
    """Selects the best-matching sentences from the retrieved chunks.

    Each sentence is scored by the weight of the query terms it contains,
    divided by the square root of its length so that long sentences are not
    favoured simply for containing more words. A sentence is only used if it
    matches at least one non-stopword query term, which is what allows the
    generator to say that the documents do not answer the question.
    """

    name = "extractive"

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    def generate(self, query: str, contexts: Sequence[Retrieved]) -> Answer:
        if not contexts:
            return Answer(text=NO_CONTEXT_MESSAGE, backend=self.name, contexts=[])

        query_terms = {
            stem(token) for token in tokenize(query) if token not in STOPWORDS
        }
        if not query_terms:
            query_terms = {stem(token) for token in tokenize(query)}

        scored: list[tuple[float, int, int, str]] = []
        for context_index, item in enumerate(contexts):
            heading = _normalise_for_comparison(item.chunk.section)
            for sentence_index, sentence in enumerate(split_sentences(item.chunk.text)):
                # The paragraph chunker prefixes each chunk with its section
                # heading so the heading words stay searchable. That heading is
                # then a short "sentence" packed with topical terms, which the
                # length normalisation below would rank above the sentence that
                # actually answers the question. It is a label, not a statement,
                # so it is not a candidate answer.
                if heading and _normalise_for_comparison(sentence) == heading:
                    continue
                tokens = [stem(t) for t in tokenize(sentence)]
                if len(tokens) < MIN_ANSWER_TOKENS:
                    continue
                overlap = query_terms.intersection(tokens)
                if not overlap:
                    continue
                # Rank contexts earlier in the list slightly higher, since the
                # retriever already judged them more relevant.
                retrieval_bonus = 1.0 / (1.0 + context_index)
                score = len(overlap) / (len(tokens) ** 0.5) + 0.1 * retrieval_bonus
                scored.append((score, context_index, sentence_index, sentence))

        if not scored:
            return Answer(
                text=NOT_ANSWERABLE_MESSAGE,
                backend=self.name,
                contexts=list(contexts),
            )

        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        chosen = scored[: self.config.max_sentences]
        # Present the sentences in reading order rather than score order.
        chosen.sort(key=lambda row: (row[1], row[2]))

        used = sorted({row[1] for row in chosen})
        citation_labels = [f"[{index + 1}]" for index in used]
        sentences = [
            f"{row[3]} [{row[1] + 1}]" for row in chosen
        ]
        return Answer(
            text=" ".join(sentences),
            backend=self.name,
            citations=citation_labels,
            contexts=list(contexts),
        )


class ClaudeGenerator(Generator):
    """Grounded answer generation through the Claude Messages API."""

    name = "claude"

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "the claude generator requires the anthropic package; install it "
                "with 'pip install anthropic' or use --generator extractive"
            ) from exc
        self._client = anthropic.Anthropic()
        self._fallback = ExtractiveGenerator(self.config)

    def generate(self, query: str, contexts: Sequence[Retrieved]) -> Answer:
        if not contexts:
            return Answer(text=NO_CONTEXT_MESSAGE, backend=self.name, contexts=[])

        prompt = (
            f"Context passages:\n\n{build_context_block(contexts)}\n\n"
            f"Question: {query}"
        )
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # A safety classifier can decline a request; the answer is then taken
        # from the extractive backend so the pipeline still returns something.
        if getattr(response, "stop_reason", None) == "refusal":
            answer = self._fallback.generate(query, contexts)
            return Answer(
                text=answer.text,
                backend=f"{self.name}->extractive",
                citations=answer.citations,
                contexts=list(contexts),
                note="the API declined the request; the extractive backend was used",
            )

        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        return Answer(
            text=text or NOT_ANSWERABLE_MESSAGE,
            backend=self.name,
            citations=sorted(set(re.findall(r"\[\d+\]", text))),
            contexts=list(contexts),
        )


def api_key_present() -> bool:
    """True when an Anthropic credential is visible in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def get_generator(
    backend: str = "extractive",
    config: GenerationConfig | None = None,
) -> Generator:
    """Build a generator by name.

    ``"auto"`` uses Claude when a credential is present and the extractive
    backend otherwise, so that a checkout without an API key still runs.
    """
    config = config or GenerationConfig()
    key = backend.strip().lower()
    if key == "extractive":
        return ExtractiveGenerator(config)
    if key == "claude":
        return ClaudeGenerator(config)
    if key == "auto":
        if api_key_present():
            try:
                return ClaudeGenerator(config)
            except ImportError:
                return ExtractiveGenerator(config)
        return ExtractiveGenerator(config)
    raise ValueError(
        f"unknown generator {backend!r}; expected 'extractive', 'claude' or 'auto'"
    )
