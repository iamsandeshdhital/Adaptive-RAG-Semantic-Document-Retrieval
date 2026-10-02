"""Tests for the evaluation metrics."""

from __future__ import annotations

import math

import pytest

from adaptive_rag.chunking import Chunk
from adaptive_rag.evaluation.metrics import (
    EvalQuestion,
    LatencySummary,
    aggregate,
    aggregate_by_kind,
    answer_in_context,
    dcg,
    document_recall_at_k,
    evaluate_query,
    hit_at_k,
    load_questions,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    relevance_flags,
    validate_questions,
)
from adaptive_rag.retrieval.base import Retrieved


def retrieved(doc_ids: list[str], texts: list[str] | None = None) -> list[Retrieved]:
    items = []
    for rank, doc_id in enumerate(doc_ids, start=1):
        text = texts[rank - 1] if texts else f"text of {doc_id}"
        chunk = Chunk(
            chunk_id=f"{doc_id}::t::{rank:03d}",
            doc_id=doc_id,
            doc_title=doc_id,
            text=text,
            index=rank,
            strategy="test",
        )
        items.append(Retrieved(chunk=chunk, score=1.0 / rank, rank=rank))
    return items


class TestRankMetrics:
    def test_relevance_flags(self):
        results = retrieved(["a", "b", "a", "c"])
        assert relevance_flags(results, ["a"]) == [1, 0, 1, 0]

    def test_hit_at_k(self):
        assert hit_at_k([0, 0, 1], 3) == 1.0
        assert hit_at_k([0, 0, 1], 2) == 0.0
        assert hit_at_k([0, 0, 0], 3) == 0.0

    def test_precision_at_k(self):
        assert precision_at_k([1, 0, 1, 0], 4) == 0.5
        assert precision_at_k([1, 1], 2) == 1.0
        assert precision_at_k([1, 1], 0) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank([0, 1, 0]) == pytest.approx(0.5)
        assert reciprocal_rank([1, 0]) == 1.0
        assert reciprocal_rank([0, 0]) == 0.0

    def test_document_recall_with_two_gold_documents(self):
        results = retrieved(["a", "x", "y"])
        assert document_recall_at_k(results, ["a", "b"], 3) == 0.5
        assert document_recall_at_k(retrieved(["a", "b"]), ["a", "b"], 2) == 1.0

    def test_document_recall_respects_k(self):
        results = retrieved(["x", "x", "a"])
        assert document_recall_at_k(results, ["a"], 2) == 0.0
        assert document_recall_at_k(results, ["a"], 3) == 1.0

    def test_ndcg_rewards_earlier_relevance(self):
        early = ndcg_at_k([1, 0, 0], 3)
        late = ndcg_at_k([0, 0, 1], 3)
        assert early == 1.0
        assert late < early

    def test_ndcg_is_zero_when_nothing_relevant(self):
        assert ndcg_at_k([0, 0, 0], 3) == 0.0

    def test_dcg_discount(self):
        # Relevance at rank 2 is discounted by log2(3).
        assert dcg([0, 1], 2) == pytest.approx(1.0 / math.log2(3))


class TestAnswerInContext:
    def test_all_keywords_present(self):
        results = retrieved(["a"], ["the sinoatrial node sets the rhythm"])
        assert answer_in_context(results, ["sinoatrial node"], 1) == 1.0

    def test_missing_keyword_fails(self):
        results = retrieved(["a"], ["the sinoatrial node sets the rhythm"])
        assert answer_in_context(results, ["sinoatrial node", "haemoglobin"], 1) == 0.0

    def test_keywords_may_span_several_chunks(self):
        results = retrieved(["a", "b"], ["mentions haemoglobin", "mentions the node"])
        assert answer_in_context(results, ["haemoglobin", "node"], 2) == 1.0

    def test_respects_k(self):
        results = retrieved(["a", "b"], ["irrelevant", "contains aragonite"])
        assert answer_in_context(results, ["aragonite"], 1) == 0.0
        assert answer_in_context(results, ["aragonite"], 2) == 1.0

    def test_multi_word_keyword_must_be_contiguous(self):
        """Two unrelated mentions must not satisfy a two-word keyword."""
        results = retrieved(["a"], ["the rain fell and a shadow moved"])
        assert answer_in_context(results, ["rain shadow"], 1) == 0.0

    def test_matching_is_stem_insensitive(self):
        results = retrieved(["a"], ["surface waves cause the damage"])
        assert answer_in_context(results, ["surface wave"], 1) == 1.0

    def test_no_keywords_is_undefined(self):
        assert math.isnan(answer_in_context(retrieved(["a"]), [], 1))

    def test_no_results_scores_zero(self):
        assert answer_in_context([], ["anything"], 5) == 0.0


class TestAggregation:
    def test_averages_over_questions(self):
        questions = [
            EvalQuestion("q1", "first", ("a",), ("text",)),
            EvalQuestion("q2", "second", ("b",), ("text",)),
        ]
        results = [
            evaluate_query(questions[0], retrieved(["a"], ["text here"]), k_values=(1,)),
            evaluate_query(questions[1], retrieved(["x"], ["nothing"]), k_values=(1,)),
        ]
        summary = aggregate(results)
        assert summary.get("hit@1") == 0.5

    def test_undefined_values_excluded_from_that_metric_only(self):
        with_keywords = EvalQuestion("q1", "first", ("a",), ("text",))
        without = EvalQuestion("q2", "second", ("a",), ())
        results = [
            evaluate_query(with_keywords, retrieved(["a"], ["text here"]), k_values=(1,)),
            evaluate_query(without, retrieved(["a"], ["text here"]), k_values=(1,)),
        ]
        summary = aggregate(results)
        # Both questions count towards hit@1; only the first has keywords.
        assert summary.get("hit@1") == 1.0
        assert summary.get("answer_in_context@1") == 1.0

    def test_empty_input(self):
        assert aggregate([]).as_dict() == {}

    def test_group_by_kind(self):
        questions = [
            EvalQuestion("q1", "a", ("a",), ("text",), kind="lexical"),
            EvalQuestion("q2", "b", ("b",), ("text",), kind="paraphrased"),
        ]
        results = [
            evaluate_query(questions[0], retrieved(["a"], ["text"]), k_values=(1,)),
            evaluate_query(questions[1], retrieved(["z"], ["other"]), k_values=(1,)),
        ]
        grouped = aggregate_by_kind(results)
        assert grouped["lexical"].get("hit@1") == 1.0
        assert grouped["paraphrased"].get("hit@1") == 0.0

    def test_context_words_recorded(self):
        question = EvalQuestion("q1", "a", ("a",), ("text",))
        result = evaluate_query(
            question, retrieved(["a", "a"], ["one two three", "four five"]), k_values=(2,)
        )
        assert result.metrics["context_words@2"] == 5.0


class TestLatencySummary:
    def test_summary_statistics(self):
        summary = LatencySummary.from_samples([1.0, 2.0, 3.0, 4.0, 100.0])
        assert summary.samples == 5
        assert summary.min_ms == 1.0
        assert summary.max_ms == 100.0
        assert summary.median_ms == 3.0
        assert summary.mean_ms == pytest.approx(22.0)

    def test_p95_is_nearest_rank(self):
        summary = LatencySummary.from_samples([float(i) for i in range(1, 101)])
        assert summary.p95_ms == 95.0

    def test_empty_samples(self):
        summary = LatencySummary.from_samples([])
        assert summary.samples == 0
        assert summary.mean_ms == 0.0


class TestQuestionLoading:
    def test_loads_the_project_eval_set(self, eval_path, real_documents):
        questions = load_questions(eval_path)
        assert len(questions) >= 50
        validate_questions(questions, [d.doc_id for d in real_documents])

    def test_every_keyword_is_present_in_its_gold_document(
        self, eval_path, real_documents
    ):
        """The evaluation set must be achievable: a perfect retriever scores 1.0."""
        from adaptive_rag.preprocess import stem, tokenize

        documents = {d.doc_id: d for d in real_documents}
        for question in load_questions(eval_path):
            for doc_id in question.gold_doc_ids:
                haystack = (
                    " "
                    + " ".join(stem(t) for t in tokenize(documents[doc_id].text))
                    + " "
                )
                for keyword in question.answer_keywords:
                    needle = " ".join(stem(t) for t in tokenize(keyword))
                    assert f" {needle} " in haystack, (
                        f"{question.qid}: {keyword!r} is absent from {doc_id}"
                    )

    def test_unknown_gold_document_rejected(self):
        questions = [EvalQuestion("q1", "text", ("missing",), ())]
        with pytest.raises(ValueError, match="unknown doc_ids"):
            validate_questions(questions, ["present"])

    def test_duplicate_ids_rejected(self, tmp_path):
        import json

        path = tmp_path / "dupes.json"
        path.write_text(
            json.dumps(
                {
                    "questions": [
                        {"qid": "q1", "question": "a", "gold_doc_ids": ["x"]},
                        {"qid": "q1", "question": "b", "gold_doc_ids": ["x"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate question id"):
            load_questions(path)

    def test_empty_set_rejected(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text('{"questions": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="no questions"):
            load_questions(path)
