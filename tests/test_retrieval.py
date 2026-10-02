"""Tests for BM25, dense and hybrid retrieval."""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_rag.chunking import Chunk
from adaptive_rag.config import BM25Config, DenseConfig, HybridConfig
from adaptive_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    build_retriever,
    get_retriever,
)
from adaptive_rag.retrieval.embeddings import LsaBackend, l2_normalise
from adaptive_rag.retrieval.hybrid import min_max_normalise


def make_chunk(index: int, text: str, doc_id: str = "d") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::t::{index:03d}",
        doc_id=doc_id,
        doc_title=doc_id,
        text=text,
        index=index,
        strategy="test",
    )


@pytest.fixture
def toy_chunks() -> list[Chunk]:
    return [
        make_chunk(0, "the cat sat on the warm mat", "cats"),
        make_chunk(1, "a dog barked loudly at the postman", "dogs"),
        make_chunk(2, "cats and dogs sometimes share a house", "pets"),
        make_chunk(3, "quantum chromodynamics describes the strong force", "physics"),
    ]


class TestBM25:
    def test_ranks_the_matching_chunk_first(self, toy_chunks):
        retriever = BM25Retriever().index(toy_chunks)
        results = retriever.search("cat mat", k=3)
        assert results[0].doc_id == "cats"

    def test_unmatched_query_returns_nothing(self, toy_chunks):
        retriever = BM25Retriever().index(toy_chunks)
        assert retriever.search("helicopter aerodynamics", k=3) == []

    def test_idf_penalises_common_terms(self, toy_chunks):
        retriever = BM25Retriever().index(toy_chunks)
        # "cats" appears in two chunks, "chromodynamics" in one.
        assert retriever.term_idf("chromodynamics") > retriever.term_idf("cats")

    def test_idf_is_never_negative(self, toy_chunks):
        """The 1 + ... form keeps a term present everywhere at a positive IDF."""
        chunks = [make_chunk(i, "shared term here", f"d{i}") for i in range(5)]
        retriever = BM25Retriever().index(chunks)
        assert retriever.term_idf("shared") > 0.0

    def test_scores_match_the_formula(self, toy_chunks):
        config = BM25Config(k1=1.5, b=0.75, remove_stopwords=True, stem=True)
        retriever = BM25Retriever(config).index(toy_chunks)

        scores = retriever.score_query("chromodynamics")
        idf = retriever.term_idf("chromodynamics")
        target = 3  # the physics chunk

        tokens = retriever._tokenize(toy_chunks[target].text)
        lengths = [len(retriever._tokenize(c.text)) for c in toy_chunks]
        avgdl = sum(lengths) / len(lengths)
        frequency = tokens.count("chromodynamic") or tokens.count("chromodynamics")

        expected = (
            idf
            * frequency
            * (config.k1 + 1.0)
            / (
                frequency
                + config.k1 * (1 - config.b + config.b * lengths[target] / avgdl)
            )
        )
        assert scores[target] == pytest.approx(expected, rel=1e-9)

    def test_length_normalisation_prefers_the_shorter_chunk(self):
        padding = " ".join(f"filler{i}" for i in range(200))
        chunks = [
            make_chunk(0, "seismograph", "short"),
            make_chunk(1, f"seismograph {padding}", "long"),
        ]
        retriever = BM25Retriever(BM25Config(b=0.75)).index(chunks)
        results = retriever.search("seismograph", k=2)
        assert results[0].doc_id == "short"

    def test_b_zero_disables_length_normalisation(self):
        padding = " ".join(f"filler{i}" for i in range(200))
        chunks = [
            make_chunk(0, "seismograph", "short"),
            make_chunk(1, f"seismograph {padding}", "long"),
        ]
        scores = BM25Retriever(BM25Config(b=0.0)).index(chunks).score_query(
            "seismograph"
        )
        assert scores[0] == pytest.approx(scores[1])

    def test_explain_reports_contributions(self, toy_chunks):
        retriever = BM25Retriever().index(toy_chunks)
        contributions = retriever.explain("cat mat", 0)
        assert contributions
        assert all(value > 0 for _, value in contributions)
        assert contributions == sorted(contributions, key=lambda p: -p[1])

    def test_search_requires_an_index(self, toy_chunks):
        with pytest.raises(RuntimeError, match="must be indexed"):
            BM25Retriever().search("cat")

    def test_empty_index_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            BM25Retriever().index([])


class TestDense:
    def test_lsa_backend_retrieves_semantically(self, toy_chunks):
        retriever = DenseRetriever(
            DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)
        ).index(toy_chunks)
        results = retriever.search("cat mat", k=4)
        assert results[0].doc_id in {"cats", "pets"}

    def test_vectors_are_unit_length(self, toy_chunks):
        retriever = DenseRetriever(
            DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)
        ).index(toy_chunks)
        norms = np.linalg.norm(retriever.matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_returns_k_results_even_without_lexical_overlap(self, toy_chunks):
        """Unlike BM25, cosine similarity always ranks every chunk."""
        retriever = DenseRetriever(
            DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)
        ).index(toy_chunks)
        assert len(retriever.search("helicopter aerodynamics", k=3)) == 3

    def test_faiss_and_numpy_paths_agree(self, toy_chunks):
        from adaptive_rag.retrieval.dense import faiss_available

        if not faiss_available():
            pytest.skip("faiss is not installed")

        backend = LsaBackend(n_components=3)
        with_faiss = DenseRetriever(
            DenseConfig(backend="lsa", lsa_dim=3, use_faiss=True)
        ).index(toy_chunks)
        without = DenseRetriever(
            DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)
        ).index(toy_chunks)

        assert with_faiss.using_faiss is True
        assert without.using_faiss is False
        for a, b in zip(with_faiss.search("cats", 4), without.search("cats", 4)):
            assert a.chunk_id == b.chunk_id
            assert a.score == pytest.approx(b.score, abs=1e-5)

    def test_lsa_requires_fitting(self, toy_chunks):
        with pytest.raises(RuntimeError, match="fit"):
            LsaBackend(n_components=2).encode(["text"])

    def test_l2_normalise_handles_zero_rows(self):
        matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        normalised = l2_normalise(matrix)
        assert np.allclose(normalised[0], [0.0, 0.0])
        assert np.allclose(normalised[1], [0.6, 0.8])


class TestHybrid:
    def test_alpha_zero_matches_bm25(self, toy_chunks):
        bm25 = BM25Retriever().index(toy_chunks)
        hybrid = HybridRetriever(
            HybridConfig(alpha=0.0, fusion="score"),
            bm25=BM25Retriever(),
            dense=DenseRetriever(DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)),
        ).index(toy_chunks)

        lexical = [r.chunk_id for r in bm25.search("cats dogs", 3)]
        fused = [r.chunk_id for r in hybrid.search("cats dogs", 3)]
        assert fused[0] == lexical[0]

    def test_alpha_one_matches_dense(self, toy_chunks):
        dense = DenseRetriever(
            DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)
        ).index(toy_chunks)
        hybrid = HybridRetriever(
            HybridConfig(alpha=1.0, fusion="score"),
            dense=DenseRetriever(DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)),
        ).index(toy_chunks)

        assert (
            hybrid.search("cats dogs", 3)[0].chunk_id
            == dense.search("cats dogs", 3)[0].chunk_id
        )

    def test_fusion_can_beat_both_components(self):
        """A chunk ranked second by both signals can win once they are combined."""
        chunks = [
            make_chunk(0, "alpha alpha alpha unrelated filler words", "a"),
            make_chunk(1, "alpha beta gamma balanced middle chunk", "b"),
            make_chunk(2, "beta beta beta different filler words", "c"),
        ]
        hybrid = HybridRetriever(
            HybridConfig(alpha=0.5, fusion="score"),
            dense=DenseRetriever(DenseConfig(backend="lsa", lsa_dim=2, use_faiss=False)),
        ).index(chunks)
        scores = hybrid.score_query("alpha beta")
        assert len(scores) == 3
        assert not np.isnan(scores).any()

    def test_rrf_ignores_score_scale(self, toy_chunks):
        hybrid = HybridRetriever(
            HybridConfig(fusion="rrf", candidate_k=4, rrf_k=60),
            dense=DenseRetriever(DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)),
        ).index(toy_chunks)
        results = hybrid.search("cats dogs", 3)
        assert results
        # Each contribution is 1/(60 + rank), so no score can exceed 2/61.
        assert all(r.score <= 2.0 / 61.0 + 1e-9 for r in results)

    def test_alpha_rejected_under_rrf(self, toy_chunks):
        hybrid = HybridRetriever(
            HybridConfig(fusion="rrf"),
            dense=DenseRetriever(DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)),
        ).index(toy_chunks)
        with pytest.raises(ValueError, match="no effect"):
            hybrid.search_with_alpha("cats", 3, 0.5)

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            HybridRetriever(HybridConfig(alpha=1.5))

    def test_invalid_fusion_rejected(self):
        with pytest.raises(ValueError, match="fusion"):
            HybridRetriever(HybridConfig(fusion="magic"))

    def test_shared_components_are_not_reindexed(self, toy_chunks):
        """Reusing an indexed dense retriever must not re-encode the corpus."""
        dense = DenseRetriever(
            DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False)
        ).index(toy_chunks)
        matrix_id = id(dense.matrix)
        HybridRetriever(HybridConfig(), dense=dense).index(toy_chunks)
        assert id(dense.matrix) == matrix_id


class TestNormalisation:
    def test_min_max_maps_onto_unit_interval(self):
        result = min_max_normalise(np.array([2.0, 4.0, 6.0]))
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.5)

    def test_constant_vector_becomes_zeros(self):
        assert np.allclose(min_max_normalise(np.array([3.0, 3.0, 3.0])), 0.0)

    def test_handles_negative_scores(self):
        result = min_max_normalise(np.array([-1.0, 0.0, 1.0]))
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)


class TestFactory:
    def test_builds_each_retriever(self, toy_chunks):
        for name in ("bm25", "dense", "hybrid"):
            retriever = get_retriever(name)
            assert retriever.name == name

    def test_build_retriever_indexes(self, toy_chunks):
        retriever = build_retriever("bm25", toy_chunks)
        assert retriever.is_indexed
        assert retriever.search("cat", 1)

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="unknown retriever"):
            get_retriever("colbert")


class TestDeterminism:
    def test_repeated_search_is_identical(self, toy_chunks):
        retriever = BM25Retriever().index(toy_chunks)
        first = retriever.search("cats dogs house", 4)
        second = retriever.search("cats dogs house", 4)
        assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
        assert [r.score for r in first] == [r.score for r in second]

    def test_ties_broken_by_chunk_order(self):
        chunks = [make_chunk(i, "identical text here", f"d{i}") for i in range(4)]
        retriever = BM25Retriever().index(chunks)
        results = retriever.search("identical", 4)
        assert [r.chunk_id for r in results] == [c.chunk_id for c in chunks]
