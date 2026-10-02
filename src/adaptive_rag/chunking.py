"""Chunking strategies.

Two strategies are compared in the experiments:

``FixedSizeChunker``
    A sliding window of a fixed number of words with a fixed overlap. It
    ignores the structure of the document entirely.

``ParagraphChunker``
    The project brief calls this the semantic strategy. Rather than segmenting
    by embedding similarity, it respects boundaries the author already put in
    the text: Markdown section headings and blank lines between paragraphs.
    Very short paragraphs are merged with their neighbours so that a chunk
    still carries enough context to answer a question, and very long ones are
    split at sentence boundaries so no chunk is unusably large.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .config import ChunkingConfig
from .loaders import Document
from .preprocess import split_sentences

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of text.

    ``section`` records the heading the chunk was taken from when one is
    available. It is used only for display and for error analysis; retrieval
    scores are computed from ``text`` alone.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    index: int
    strategy: str
    section: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def display_source(self) -> str:
        """A short human-readable provenance string for citations."""
        if self.section:
            return f"{self.doc_title} / {self.section}"
        return self.doc_title


class Chunker(ABC):
    """Base class for chunking strategies."""

    #: Short identifier used in result tables and chunk identifiers.
    name: str = "base"

    @abstractmethod
    def split(self, document: Document) -> list[str]:
        """Return the raw text of each chunk for one document."""

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Wrap the output of :meth:`split` in :class:`Chunk` objects."""
        chunks: list[Chunk] = []
        for index, text in enumerate(self.split(document)):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{self.name}::{index:03d}",
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    text=_strip_headings(text),
                    index=index,
                    strategy=self.name,
                    section=_leading_heading(text),
                )
            )
        return chunks

    def chunk_corpus(self, documents: Sequence[Document]) -> list[Chunk]:
        """Chunk every document in order."""
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks


def _leading_heading(text: str) -> str:
    """The most recent heading at the start of a chunk, if any."""
    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match is not None:
            return match.group(2).strip()
    return ""


def _strip_headings(text: str) -> str:
    """Remove the ``#`` markers but keep the heading words as searchable text.

    A heading such as "Bleaching and Thermal Stress" is a strong lexical signal,
    so it is retained in the chunk body; only the Markdown syntax is dropped.
    """
    lines = []
    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        lines.append(match.group(2).strip() + "." if match else line)
    return " ".join(" ".join(lines).split())


class FixedSizeChunker(Chunker):
    """Fixed-size sliding window over the word sequence of a document."""

    name = "fixed"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    @property
    def stride(self) -> int:
        """Words advanced between consecutive windows."""
        return self.config.chunk_size - self.config.overlap

    def split(self, document: Document) -> list[str]:
        words = document.text.split()
        size = self.config.chunk_size
        if not words:
            return []
        if len(words) <= size:
            return [" ".join(words)]

        windows: list[str] = []
        start = 0
        while start < len(words):
            window = words[start : start + size]
            windows.append(" ".join(window))
            if start + size >= len(words):
                break
            start += self.stride

        # A trailing window shorter than the overlap adds nothing that the
        # previous window does not already contain, so drop it.
        if len(windows) > 1 and len(windows[-1].split()) <= self.config.overlap:
            windows.pop()
        return windows

    def __repr__(self) -> str:
        return (
            f"FixedSizeChunker(chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.overlap})"
        )


class ParagraphChunker(Chunker):
    """Structure-aware chunker using headings and paragraph boundaries."""

    name = "paragraph"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def split(self, document: Document) -> list[str]:
        blocks = self._blocks_with_headings(document.text)
        merged = self._merge_short(blocks)
        chunks: list[str] = []
        for block in merged:
            chunks.extend(self._split_long(block))
        return chunks

    def _blocks_with_headings(self, text: str) -> list[str]:
        """Split on blank lines, attaching each heading to the text below it.

        A heading on its own is not a useful chunk, so it is prefixed to the
        first paragraph of its section and repeated as the ``section`` label of
        every later paragraph in that section.
        """
        blocks: list[str] = []
        pending_heading = ""
        current_section = ""

        for raw_block in text.split("\n\n"):
            block = raw_block.strip()
            if not block:
                continue

            lines = block.splitlines()
            heading_match = _HEADING.match(lines[0].strip())
            if heading_match is not None and len(lines) == 1:
                pending_heading = lines[0].strip()
                current_section = heading_match.group(2).strip()
                continue
            if heading_match is not None:
                pending_heading = lines[0].strip()
                current_section = heading_match.group(2).strip()
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue

            if pending_heading:
                blocks.append(f"{pending_heading}\n{block}")
                pending_heading = ""
            elif current_section:
                blocks.append(f"### {current_section}\n{block}")
            else:
                blocks.append(block)

        return blocks

    def _merge_short(self, blocks: Sequence[str]) -> list[str]:
        """Merge blocks below the minimum word count into the following block."""
        minimum = self.config.min_paragraph_words
        merged: list[str] = []
        buffer = ""

        for block in blocks:
            candidate = f"{buffer}\n{block}".strip() if buffer else block
            if len(candidate.split()) < minimum:
                buffer = candidate
                continue
            merged.append(candidate)
            buffer = ""

        if buffer:
            # Nothing left to merge forward into, so attach to the last chunk
            # instead of emitting an undersized one.
            if merged:
                merged[-1] = f"{merged[-1]}\n{buffer}".strip()
            else:
                merged.append(buffer)
        return merged

    def _split_long(self, block: str) -> list[str]:
        """Split an oversized block at sentence boundaries."""
        maximum = self.config.max_paragraph_words
        if len(block.split()) <= maximum:
            return [block]

        heading = ""
        body = block
        first_line = block.splitlines()[0].strip()
        if _HEADING.match(first_line):
            heading = first_line
            body = "\n".join(block.splitlines()[1:]).strip()

        pieces: list[str] = []
        current: list[str] = []
        current_words = 0
        for sentence in split_sentences(body):
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > maximum:
                pieces.append(" ".join(current))
                current, current_words = [], 0
            current.append(sentence)
            current_words += sentence_words
        if current:
            pieces.append(" ".join(current))

        if heading:
            pieces = [f"{heading}\n{piece}" for piece in pieces]
        return pieces or [block]

    def __repr__(self) -> str:
        return (
            f"ParagraphChunker(min_words={self.config.min_paragraph_words}, "
            f"max_words={self.config.max_paragraph_words})"
        )


def get_chunker(name: str, config: ChunkingConfig | None = None) -> Chunker:
    """Build a chunker by name (``"fixed"`` or ``"paragraph"``)."""
    key = name.strip().lower()
    if key in {"fixed", "fixed-size", "fixed_size"}:
        return FixedSizeChunker(config)
    if key in {"paragraph", "semantic", "section"}:
        return ParagraphChunker(config)
    raise ValueError(f"unknown chunking strategy {name!r}; expected 'fixed' or 'paragraph'")


def chunk_statistics(chunks: Iterable[Chunk]) -> dict[str, float | int]:
    """Chunk counts and length distribution, reported alongside every run."""
    counts = [chunk.word_count for chunk in chunks]
    if not counts:
        return {"chunks": 0, "mean_words": 0.0, "min_words": 0, "max_words": 0,
                "total_words": 0}
    counts.sort()
    return {
        "chunks": len(counts),
        "total_words": sum(counts),
        "mean_words": round(sum(counts) / len(counts), 1),
        "median_words": counts[len(counts) // 2],
        "min_words": counts[0],
        "max_words": counts[-1],
    }
