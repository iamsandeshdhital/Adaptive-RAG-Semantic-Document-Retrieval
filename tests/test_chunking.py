"""Tests for the two chunking strategies."""

from __future__ import annotations

import pytest

from adaptive_rag.chunking import (
    FixedSizeChunker,
    ParagraphChunker,
    chunk_statistics,
    get_chunker,
)
from adaptive_rag.config import ChunkingConfig


class TestFixedSizeChunker:
    def test_window_size_and_stride(self, numbered_document):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=300, overlap=50))
        chunks = chunker.chunk_document(numbered_document)

        assert chunker.stride == 250
        assert all(chunk.word_count <= 300 for chunk in chunks)
        # 1000 words with a stride of 250 starts windows at 0, 250, 500, 750.
        assert len(chunks) == 4
        assert chunks[0].text.split()[0] == "w0"
        assert chunks[1].text.split()[0] == "w250"

    def test_consecutive_chunks_share_the_overlap(self, numbered_document):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=100, overlap=20))
        chunks = chunker.chunk_document(numbered_document)

        for earlier, later in zip(chunks, chunks[1:]):
            tail = earlier.text.split()[-20:]
            head = later.text.split()[:20]
            assert tail == head

    def test_zero_overlap_partitions_the_document(self, numbered_document):
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=250, overlap=0))
        chunks = chunker.chunk_document(numbered_document)

        recombined = " ".join(chunk.text for chunk in chunks).split()
        assert recombined == numbered_document.text.split()

    def test_short_document_is_a_single_chunk(self):
        from adaptive_rag.loaders import Document
        from pathlib import Path

        document = Document(
            doc_id="short", title="Short", text="only a handful of words here",
            path=Path("short.txt"),
        )
        chunks = FixedSizeChunker().chunk_document(document)
        assert len(chunks) == 1
        assert chunks[0].text == "only a handful of words here"

    def test_trailing_sliver_is_dropped(self):
        """A final window no longer than the overlap is already covered."""
        from adaptive_rag.loaders import Document
        from pathlib import Path

        # 105 words, chunk 100, overlap 20 -> windows at 0 and 80; the window at
        # 160 would not exist, but a naive implementation emits a 5-word tail.
        document = Document(
            doc_id="sliver", title="Sliver",
            text=" ".join(f"w{i}" for i in range(105)), path=Path("s.txt"),
        )
        chunks = FixedSizeChunker(
            ChunkingConfig(chunk_size=100, overlap=20)
        ).chunk_document(document)
        assert all(chunk.word_count > 20 for chunk in chunks)

    def test_invalid_overlap_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            ChunkingConfig(chunk_size=100, overlap=100)

    def test_invalid_chunk_size_rejected(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkingConfig(chunk_size=0)


class TestParagraphChunker:
    def test_respects_paragraph_boundaries(self, tiny_documents):
        chunker = ParagraphChunker(ChunkingConfig(min_paragraph_words=10))
        chunks = chunker.chunk_document(tiny_documents[0])

        # The sourdough document has two sections with one paragraph each.
        assert len(chunks) == 2
        assert "Feeding" in chunks[0].section
        assert "Storage" in chunks[1].section

    def test_heading_text_is_kept_in_the_body(self, tiny_documents):
        chunks = ParagraphChunker(
            ChunkingConfig(min_paragraph_words=10)
        ).chunk_document(tiny_documents[0])

        # The heading words remain searchable, without the Markdown syntax.
        assert chunks[0].text.startswith("Feeding.")
        assert "#" not in chunks[0].text

    def test_short_paragraphs_are_merged(self, tmp_path):
        from adaptive_rag.loaders import load_document

        path = tmp_path / "short_paras.md"
        path.write_text(
            "---\ndoc_id: tiny\ntitle: Tiny\n---\n\n"
            "One short line here.\n\nAnother short line here.\n\n"
            "A third short line here.\n",
            encoding="utf-8",
        )
        document = load_document(path)
        chunks = ParagraphChunker(
            ChunkingConfig(min_paragraph_words=12)
        ).chunk_document(document)

        assert len(chunks) == 1
        assert chunks[0].word_count >= 12

    def test_long_paragraph_is_split_at_sentences(self, tmp_path):
        from adaptive_rag.loaders import load_document

        sentence = "This sentence has exactly eight words in it. "
        path = tmp_path / "long_para.md"
        path.write_text(
            "---\ndoc_id: long\ntitle: Long\n---\n\n" + sentence * 20 + "\n",
            encoding="utf-8",
        )
        document = load_document(path)
        chunks = ParagraphChunker(
            ChunkingConfig(min_paragraph_words=10, max_paragraph_words=40)
        ).chunk_document(document)

        assert len(chunks) > 1
        assert all(chunk.word_count <= 48 for chunk in chunks)
        # Splitting at sentence boundaries means no chunk ends mid-sentence.
        assert all(chunk.text.rstrip().endswith(".") for chunk in chunks)

    def test_document_without_headings(self, tmp_path):
        from adaptive_rag.loaders import load_document

        path = tmp_path / "plain.txt"
        path.write_text(
            "First paragraph with a reasonable number of words in it here.\n\n"
            "Second paragraph with a reasonable number of words in it too.\n",
            encoding="utf-8",
        )
        document = load_document(path)
        chunks = ParagraphChunker(
            ChunkingConfig(min_paragraph_words=8)
        ).chunk_document(document)

        assert len(chunks) == 2
        assert all(chunk.section == "" for chunk in chunks)


class TestChunkIdentity:
    def test_chunk_ids_are_unique_and_stable(self, real_documents):
        for strategy in ("fixed", "paragraph"):
            chunks = get_chunker(strategy).chunk_corpus(real_documents)
            ids = [chunk.chunk_id for chunk in chunks]
            assert len(ids) == len(set(ids))
            # Re-chunking the same documents reproduces the same identifiers.
            again = get_chunker(strategy).chunk_corpus(real_documents)
            assert [c.chunk_id for c in again] == ids

    def test_every_document_is_represented(self, real_documents):
        for strategy in ("fixed", "paragraph"):
            chunks = get_chunker(strategy).chunk_corpus(real_documents)
            assert {c.doc_id for c in chunks} == {d.doc_id for d in real_documents}

    def test_no_chunk_is_empty(self, real_documents):
        for strategy in ("fixed", "paragraph"):
            chunks = get_chunker(strategy).chunk_corpus(real_documents)
            assert all(chunk.text.strip() for chunk in chunks)
            assert all(chunk.word_count > 0 for chunk in chunks)

    def test_paragraph_chunking_adds_no_redundancy(self, real_documents):
        """Fixed-size chunking duplicates the overlap; paragraphs should not."""
        corpus_words = sum(d.word_count for d in real_documents)
        paragraph_words = chunk_statistics(
            get_chunker("paragraph").chunk_corpus(real_documents)
        )["total_words"]
        fixed_words = chunk_statistics(
            get_chunker("fixed").chunk_corpus(real_documents)
        )["total_words"]

        assert paragraph_words <= corpus_words * 1.05
        assert fixed_words > corpus_words


def test_get_chunker_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        get_chunker("embedding-similarity")


def test_chunk_statistics_of_empty_input():
    stats = chunk_statistics([])
    assert stats["chunks"] == 0
