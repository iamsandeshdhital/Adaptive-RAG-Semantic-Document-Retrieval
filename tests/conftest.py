"""Shared fixtures.

The tests use a tiny synthetic corpus rather than the real one wherever the
behaviour under test does not depend on the real documents. That keeps them
fast and keeps the assertions readable, since the expected chunk boundaries and
term counts can be worked out by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_rag.chunking import FixedSizeChunker, ParagraphChunker
from adaptive_rag.config import DOCUMENTS_DIR, EVAL_PATH, ChunkingConfig
from adaptive_rag.loaders import Document, load_corpus

TINY_DOCS = {
    "alpha.md": """---
title: Sourdough Starter
doc_id: sourdough
topic: cooking
---

## Feeding

A sourdough starter is a culture of flour and water that hosts wild yeast and
lactic acid bacteria. Feed it once a day with equal weights of flour and water
to keep the culture active and predictable.

## Storage

Refrigerate the jar to slow fermentation when you are not baking. Cold storage
means the starter needs feeding only once a week instead of every day.
""",
    "beta.md": """---
title: Bicycle Gears
doc_id: bicycle
topic: mechanics
---

## Ratios

A bicycle drivetrain trades force for distance. A small chainring paired with a
large sprocket gives a low gear that climbs steep hills slowly but with little
effort from the rider.

## Maintenance

Keep the chain clean and lubricated. A worn chain wears the sprockets quickly
and will skip under load.
""",
}


@pytest.fixture
def tiny_corpus_dir(tmp_path: Path) -> Path:
    """A two-document corpus written to a temporary directory."""
    directory = tmp_path / "documents"
    directory.mkdir()
    for name, content in TINY_DOCS.items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


@pytest.fixture
def tiny_documents(tiny_corpus_dir: Path) -> list[Document]:
    return load_corpus(tiny_corpus_dir)


@pytest.fixture
def tiny_chunks(tiny_documents: list[Document]):
    return ParagraphChunker(ChunkingConfig(min_paragraph_words=10)).chunk_corpus(
        tiny_documents
    )


@pytest.fixture
def numbered_document() -> Document:
    """A document whose words are "w0 w1 w2 ..." so windows are checkable."""
    text = " ".join(f"w{i}" for i in range(1000))
    return Document(
        doc_id="numbered",
        title="Numbered",
        text=text,
        path=Path("numbered.txt"),
    )


@pytest.fixture(scope="session")
def real_documents() -> list[Document]:
    """The project corpus, skipped if it is not present."""
    if not DOCUMENTS_DIR.is_dir():
        pytest.skip(f"corpus directory missing: {DOCUMENTS_DIR}")
    return load_corpus(DOCUMENTS_DIR)


@pytest.fixture(scope="session")
def eval_path() -> Path:
    if not EVAL_PATH.is_file():
        pytest.skip(f"evaluation set missing: {EVAL_PATH}")
    return EVAL_PATH


@pytest.fixture
def fixed_chunker() -> FixedSizeChunker:
    return FixedSizeChunker(ChunkingConfig(chunk_size=300, overlap=50))
