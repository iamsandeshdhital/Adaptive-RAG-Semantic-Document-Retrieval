"""Tests for the end-to-end pipeline, generation and the experiment runner."""

from __future__ import annotations

import pytest

from adaptive_rag.config import DenseConfig, GenerationConfig, RAGConfig
from adaptive_rag.generation import (
    NOT_ANSWERABLE_MESSAGE,
    NO_CONTEXT_MESSAGE,
    ExtractiveGenerator,
    build_context_block,
    get_generator,
)
from adaptive_rag.pipeline import RAGPipeline, SharedCorpus, build_pipeline


def lsa_config(directory, **kwargs) -> RAGConfig:
    """A config that avoids downloading a transformer model in the tests."""
    return RAGConfig(
        documents_dir=directory,
        dense=DenseConfig(backend="lsa", lsa_dim=3, use_faiss=False),
        **kwargs,
    )


class TestPipeline:
    def test_builds_and_reports(self, tiny_corpus_dir):
        pipeline = build_pipeline(lsa_config(tiny_corpus_dir, retriever="bm25"))
        assert pipeline.report is not None
        assert pipeline.report.documents == 2
        assert pipeline.report.chunks > 0
        assert pipeline.report.total_seconds >= 0.0

    def test_retrieve_returns_ranked_results(self, tiny_corpus_dir):
        pipeline = build_pipeline(lsa_config(tiny_corpus_dir, retriever="bm25"))
        results = pipeline.retrieve("how often should I feed a starter?", k=2)
        assert results
        assert results[0].doc_id == "sourdough"
        assert [r.rank for r in results] == sorted(r.rank for r in results)

    def test_answer_cites_its_context(self, tiny_corpus_dir):
        pipeline = build_pipeline(lsa_config(tiny_corpus_dir, retriever="bm25"))
        answer, contexts = pipeline.answer("what wears the sprockets?", k=2)
        assert contexts
        assert answer.backend == "extractive"
        assert "[1]" in answer.text or "[2]" in answer.text
        assert answer.format_citations()

    @pytest.mark.parametrize("retriever", ["bm25", "dense", "hybrid"])
    def test_every_retriever_runs_end_to_end(self, tiny_corpus_dir, retriever):
        pipeline = build_pipeline(lsa_config(tiny_corpus_dir, retriever=retriever))
        answer, contexts = pipeline.answer("what keeps a starter active?", k=2)
        assert contexts
        assert answer.text

    @pytest.mark.parametrize("chunking", ["fixed", "paragraph"])
    def test_every_chunking_runs_end_to_end(self, tiny_corpus_dir, chunking):
        pipeline = build_pipeline(lsa_config(tiny_corpus_dir, chunking=chunking))
        assert pipeline.retrieve("gear ratios", k=1)

    def test_accessors_require_build(self, tiny_corpus_dir):
        pipeline = RAGPipeline(lsa_config(tiny_corpus_dir))
        with pytest.raises(RuntimeError, match="has not been built"):
            _ = pipeline.chunks

    def test_describe_records_the_configuration(self, tiny_corpus_dir):
        pipeline = build_pipeline(
            lsa_config(tiny_corpus_dir, retriever="hybrid", chunking="paragraph")
        )
        described = pipeline.describe()
        assert described["chunking"] == "paragraph"
        assert described["retriever_config"]["retriever"] == "hybrid"


class TestSharedCorpus:
    def test_chunks_are_cached_per_strategy(self, tiny_corpus_dir):
        corpus = SharedCorpus(lsa_config(tiny_corpus_dir))
        first = corpus.chunks("fixed")
        assert corpus.chunks("fixed") is first
        assert corpus.chunks("paragraph") is not first

    def test_dense_index_is_reused_by_hybrid(self, tiny_corpus_dir):
        """The expensive step is encoding, so it must happen only once."""
        corpus = SharedCorpus(lsa_config(tiny_corpus_dir))
        dense = corpus.dense("fixed")
        matrix_id = id(dense.matrix)
        hybrid_pipeline = corpus.pipeline("fixed", "hybrid")
        assert hybrid_pipeline.retriever.dense is dense
        assert id(dense.matrix) == matrix_id

    def test_pipeline_uses_the_requested_combination(self, tiny_corpus_dir):
        corpus = SharedCorpus(lsa_config(tiny_corpus_dir))
        pipeline = corpus.pipeline("paragraph", "bm25")
        assert pipeline.config.chunking == "paragraph"
        assert pipeline.retriever.name == "bm25"


class TestExtractiveGenerator:
    def test_selects_the_matching_sentence(self, tiny_chunks):
        from adaptive_rag.retrieval.base import Retrieved

        contexts = [
            Retrieved(chunk=chunk, score=1.0, rank=i + 1)
            for i, chunk in enumerate(tiny_chunks[:2])
        ]
        answer = ExtractiveGenerator().generate("wild yeast", contexts)
        assert "yeast" in answer.text.lower()
        assert answer.citations

    def test_section_heading_is_not_quoted_as_an_answer(self):
        """The propagated heading is a label, not a statement.

        The paragraph chunker prefixes each chunk with its section heading so
        the heading words stay searchable. That makes the heading a short,
        topically dense pseudo-sentence, which the length normalisation would
        rank above the sentence that actually answers the question.
        """
        from adaptive_rag.chunking import Chunk
        from adaptive_rag.retrieval.base import Retrieved

        chunk = Chunk(
            chunk_id="c::paragraph::000",
            doc_id="c",
            doc_title="Cardiac",
            text=(
                "The Heart and the Cardiac Cycle. The rhythm is set by the "
                "sinoatrial node in the wall of the right atrium."
            ),
            index=0,
            strategy="paragraph",
            section="The Heart and the Cardiac Cycle",
        )
        answer = ExtractiveGenerator().generate(
            "what sets the rhythm of the heart?",
            [Retrieved(chunk=chunk, score=1.0, rank=1)],
        )
        assert "sinoatrial" in answer.text
        assert "The Heart and the Cardiac Cycle." not in answer.text

    def test_short_fragments_are_not_quoted(self):
        from adaptive_rag.chunking import Chunk
        from adaptive_rag.retrieval.base import Retrieved

        chunk = Chunk(
            chunk_id="c::t::000",
            doc_id="c",
            doc_title="Fragments",
            text="Coral reefs. Reefs are built from the skeletons of coral polyps.",
            index=0,
            strategy="test",
        )
        answer = ExtractiveGenerator().generate(
            "what are reefs built from?",
            [Retrieved(chunk=chunk, score=1.0, rank=1)],
        )
        assert "skeletons" in answer.text
        assert not answer.text.startswith("Coral reefs.")

    def test_no_context_message(self):
        answer = ExtractiveGenerator().generate("anything", [])
        assert answer.text == NO_CONTEXT_MESSAGE

    def test_says_so_when_context_does_not_answer(self, tiny_chunks):
        from adaptive_rag.retrieval.base import Retrieved

        contexts = [Retrieved(chunk=tiny_chunks[0], score=1.0, rank=1)]
        answer = ExtractiveGenerator().generate("helicopter turbine blades", contexts)
        assert answer.text == NOT_ANSWERABLE_MESSAGE

    def test_respects_the_sentence_budget(self, tiny_chunks):
        from adaptive_rag.retrieval.base import Retrieved

        contexts = [
            Retrieved(chunk=chunk, score=1.0, rank=i + 1)
            for i, chunk in enumerate(tiny_chunks)
        ]
        answer = ExtractiveGenerator(GenerationConfig(max_sentences=1)).generate(
            "starter flour water", contexts
        )
        # One sentence plus its citation marker.
        assert answer.text.count("[") == 1

    def test_context_block_is_numbered(self, tiny_chunks):
        from adaptive_rag.retrieval.base import Retrieved

        contexts = [
            Retrieved(chunk=chunk, score=1.0, rank=i + 1)
            for i, chunk in enumerate(tiny_chunks[:2])
        ]
        block = build_context_block(contexts)
        assert "[1] Source:" in block
        assert "[2] Source:" in block


class TestGeneratorFactory:
    def test_extractive_by_name(self):
        assert get_generator("extractive").name == "extractive"

    def test_auto_without_credentials_is_extractive(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        assert get_generator("auto").name == "extractive"

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="unknown generator"):
            get_generator("gpt")


class TestExperimentRunner:
    def test_runs_a_small_grid(self, tiny_corpus_dir):
        from adaptive_rag.evaluation import run_experiments
        from adaptive_rag.evaluation.metrics import EvalQuestion

        import json

        questions_path = tiny_corpus_dir.parent / "questions.json"
        questions_path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "qid": "q1",
                            "question": "how often should a starter be fed?",
                            "gold_doc_ids": ["sourdough"],
                            "answer_keywords": ["once a day"],
                            "kind": "lexical",
                        },
                        {
                            "qid": "q2",
                            "question": "what happens to a worn chain?",
                            "gold_doc_ids": ["bicycle"],
                            "answer_keywords": ["sprockets"],
                            "kind": "lexical",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        report = run_experiments(
            lsa_config(tiny_corpus_dir),
            questions_path,
            chunkings=["paragraph"],
            retrievers=["bm25", "dense", "hybrid"],
            k_values=(1, 2),
            repeats=1,
            include_alpha_sweep=False,
            include_k_sweep=False,
            verbose=False,
        )

        assert len(report.runs) == 3
        assert report.questions == 2
        for run in report.runs:
            assert run.latency.samples == 2
            assert 0.0 <= run.metrics["hit@1"] <= 1.0
            assert run.index_seconds >= 0.0

    def test_saves_every_output_file(self, tiny_corpus_dir, tmp_path):
        from adaptive_rag.evaluation import run_experiments, save_report

        import json

        questions_path = tmp_path / "q.json"
        questions_path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "qid": "q1",
                            "question": "how often should a starter be fed?",
                            "gold_doc_ids": ["sourdough"],
                            "answer_keywords": ["once a day"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        report = run_experiments(
            lsa_config(tiny_corpus_dir),
            questions_path,
            chunkings=["paragraph"],
            retrievers=["bm25"],
            k_values=(1,),
            repeats=1,
            include_alpha_sweep=False,
            include_k_sweep=False,
            verbose=False,
        )
        written = save_report(report, tmp_path / "out")
        for key in ("summary_csv", "json", "markdown", "per_query_csv"):
            assert written[key].is_file()
            assert written[key].stat().st_size > 0
