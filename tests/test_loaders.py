"""Tests for document loading, text extraction and preprocessing."""

from __future__ import annotations

import pytest

from adaptive_rag.loaders import (
    DocumentLoadError,
    corpus_statistics,
    load_corpus,
    load_document,
    normalise_text,
)
from adaptive_rag.preprocess import (
    STOPWORDS,
    preprocess,
    split_sentences,
    stem,
    tokenize,
)


class TestLoading:
    def test_reads_front_matter(self, tiny_corpus_dir):
        document = load_document(tiny_corpus_dir / "alpha.md")
        assert document.doc_id == "sourdough"
        assert document.title == "Sourdough Starter"
        assert document.topic == "cooking"

    def test_front_matter_is_not_part_of_the_body(self, tiny_corpus_dir):
        document = load_document(tiny_corpus_dir / "alpha.md")
        assert "doc_id" not in document.text
        assert "topic:" not in document.text

    def test_doc_id_derived_from_filename_when_absent(self, tmp_path):
        path = tmp_path / "07_solar_power.md"
        path.write_text("Some text about solar power collection.\n", encoding="utf-8")
        document = load_document(path)
        # The numeric ordering prefix is dropped.
        assert document.doc_id == "solar_power"

    def test_title_taken_from_first_heading(self, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("# A Real Heading\n\nBody text follows here.\n", encoding="utf-8")
        assert load_document(path).title == "A Real Heading"

    def test_corpus_is_sorted_by_filename(self, tiny_corpus_dir):
        documents = load_corpus(tiny_corpus_dir)
        assert [d.doc_id for d in documents] == ["sourdough", "bicycle"]

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="corpus directory not found"):
            load_corpus(tmp_path / "nope")

    def test_directory_without_documents(self, tmp_path):
        (tmp_path / "notes.docx").write_text("x", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="no supported documents"):
            load_corpus(tmp_path)

    def test_unsupported_suffix(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n", encoding="utf-8")
        with pytest.raises(DocumentLoadError, match="unsupported file type"):
            load_document(path)

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "blank.txt"
        path.write_text("   \n\n  \n", encoding="utf-8")
        with pytest.raises(DocumentLoadError, match="no text"):
            load_document(path)

    def test_duplicate_doc_ids_rejected(self, tmp_path):
        for name in ("a.md", "b.md"):
            (tmp_path / name).write_text(
                "---\ndoc_id: same\n---\n\nSome body text here.\n", encoding="utf-8"
            )
        with pytest.raises(DocumentLoadError, match="duplicate doc_id"):
            load_corpus(tmp_path)

    def test_real_corpus_loads(self, real_documents):
        assert len(real_documents) >= 10
        assert all(d.word_count > 100 for d in real_documents)
        stats = corpus_statistics(real_documents)
        assert stats["documents"] == len(real_documents)
        assert stats["total_words"] > 5000


class TestPdfIngestion:
    """The PDF path matters because a PDF with no readable text layer is
    invisible to the retriever, and that failure is silent."""

    def test_round_trip_through_pdf(self, tmp_path, real_documents):
        pytest.importorskip("matplotlib")
        pytest.importorskip("pypdf")
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from make_pdf_sample import render_pdf

        source = real_documents[0]
        destination = tmp_path / "sample.pdf"
        render_pdf(source.text, source.title, destination)

        extracted = load_document(destination)
        original = set(source.text.lower().split())
        recovered = original & set(extracted.text.lower().split())
        assert len(recovered) / len(original) > 0.9
        assert extracted.title == source.title

    def test_pdf_metadata_becomes_document_metadata(self, tmp_path, real_documents):
        pytest.importorskip("matplotlib")
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from make_pdf_sample import render_pdf

        destination = tmp_path / "titled.pdf"
        render_pdf("A short body of text for the page.", "Chosen Title", destination)
        assert load_document(destination).title == "Chosen Title"


class TestNormalisation:
    def test_joins_hard_wrapped_lines(self):
        text = normalise_text("a line that was\nwrapped by the editor\n\nnext block")
        assert "a line that was wrapped by the editor" in text
        assert text.count("\n\n") == 1

    def test_preserves_paragraph_breaks(self):
        text = normalise_text("first\n\n\n\nsecond")
        assert text == "first\n\nsecond"

    def test_repairs_hyphenated_line_breaks(self):
        assert "photosynthesis" in normalise_text("photo-\nsynthesis")

    def test_keeps_headings_on_their_own_line(self):
        text = normalise_text("## A Heading\nbody text here")
        assert text.splitlines()[0] == "## A Heading"

    def test_normalises_windows_line_endings(self):
        assert normalise_text("a\r\n\r\nb") == "a\n\nb"


class TestPreprocess:
    def test_tokenize_lowercases(self):
        assert tokenize("The RuBisCO Enzyme") == ["the", "rubisco", "enzyme"]

    def test_tokenize_keeps_hyphenated_terms_together(self):
        assert tokenize("crown-of-thorns starfish") == ["crown-of-thorns", "starfish"]

    def test_tokenize_drops_punctuation(self):
        assert tokenize("cells, tissues; organs.") == ["cells", "tissues", "organs"]

    def test_stopwords_removed(self):
        tokens = preprocess("the cat sat on the mat", apply_stemming=False)
        assert "the" not in tokens
        assert "cat" in tokens

    def test_stopwords_kept_when_disabled(self):
        tokens = preprocess("the cat", remove_stopwords=False, apply_stemming=False)
        assert "the" in tokens

    @pytest.mark.parametrize(
        "singular,plural",
        [
            ("wave", "waves"),
            ("fault", "faults"),
            ("capillary", "capillaries"),
            ("gas", "gases"),
            ("node", "nodes"),
            ("day", "days"),
            ("box", "boxes"),
            ("class", "classes"),
        ],
    )
    def test_plurals_conflate(self, singular, plural):
        assert stem(singular) == stem(plural)

    @pytest.mark.parametrize(
        "base,inflected",
        [
            ("erupt", "erupts"),
            ("erupt", "erupted"),
            ("erupt", "eruption"),
            ("bleach", "bleaching"),
            ("form", "formed"),
            ("stop", "stopping"),
            ("amplify", "amplifies"),
            ("amplify", "amplified"),
        ],
    )
    def test_inflections_conflate(self, base, inflected):
        assert stem(base) == stem(inflected)

    def test_short_tokens_untouched(self):
        for token in ("gas", "ion", "s", "ph"):
            assert stem(token) == token

    def test_stemming_is_deterministic(self):
        assert stem("volcanoes") == stem("volcanoes")

    def test_stopword_list_is_lowercase(self):
        assert all(word == word.lower() for word in STOPWORDS)


class TestSentenceSplitting:
    def test_splits_on_terminal_punctuation(self):
        sentences = split_sentences("First one. Second one! Third one?")
        assert len(sentences) == 3

    def test_does_not_split_on_initials(self):
        sentences = split_sentences("President J. F. Kennedy set the goal. Then it began.")
        assert len(sentences) == 2
        assert "J. F. Kennedy" in sentences[0]

    def test_empty_input(self):
        assert split_sentences("   ") == []

    def test_collapses_whitespace(self):
        assert split_sentences("a   b\n c.") == ["a b c."]
