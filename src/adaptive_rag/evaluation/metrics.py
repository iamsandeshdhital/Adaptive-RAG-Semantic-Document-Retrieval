"""Retrieval metrics and latency summaries.

Relevance is judged at document level: a retrieved chunk counts as relevant
when it comes from one of the documents annotated as containing the answer.
Judging every chunk by hand would be infeasible for a corpus that is re-chunked
by each strategy, and document-level labels stay valid no matter how the text is
split, which is exactly what a comparison of chunking strategies needs.

The one chunk-level metric is ``answer_in_context``, which checks whether the
retrieved text actually contains the facts needed to answer. It is the metric
that matters most for a RAG system, because a generator can only use what it is
given, and it is the metric reported as "accuracy" in the results table.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..preprocess import stem, tokenize
from ..retrieval.base import Retrieved


@dataclass(frozen=True)
class EvalQuestion:
    """One annotated question from the evaluation set."""

    qid: str
    question: str
    #: Documents that contain the answer.
    gold_doc_ids: tuple[str, ...]
    #: Terms that must all appear in the retrieved text for the answer to be
    #: recoverable from it. Matched after stemming, so "chambers" matches
    #: "chamber".
    answer_keywords: tuple[str, ...] = ()
    #: "lexical" questions reuse the wording of the source; "paraphrased"
    #: questions deliberately avoid it; "multi_hop" questions need two sections.
    kind: str = "lexical"
    note: str = ""


def load_questions(path: Path | str) -> list[EvalQuestion]:
    """Read the evaluation set from JSON."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) else payload

    parsed: list[EvalQuestion] = []
    seen: set[str] = set()
    for entry in questions:
        qid = str(entry["qid"])
        if qid in seen:
            raise ValueError(f"duplicate question id {qid!r} in {path}")
        seen.add(qid)
        parsed.append(
            EvalQuestion(
                qid=qid,
                question=entry["question"],
                gold_doc_ids=tuple(entry["gold_doc_ids"]),
                answer_keywords=tuple(entry.get("answer_keywords", ())),
                kind=entry.get("kind", "lexical"),
                note=entry.get("note", ""),
            )
        )
    if not parsed:
        raise ValueError(f"no questions found in {path}")
    return parsed


def validate_questions(
    questions: Sequence[EvalQuestion], known_doc_ids: Iterable[str]
) -> None:
    """Fail loudly if the evaluation set references a document that is absent."""
    known = set(known_doc_ids)
    for question in questions:
        unknown = set(question.gold_doc_ids) - known
        if unknown:
            raise ValueError(
                f"question {question.qid} references unknown doc_ids {sorted(unknown)}"
            )


# -- rank-based metrics -----------------------------------------------------
def relevance_flags(
    retrieved: Sequence[Retrieved], gold_doc_ids: Iterable[str]
) -> list[int]:
    """1 for each retrieved chunk that comes from a gold document."""
    gold = set(gold_doc_ids)
    return [1 if item.doc_id in gold else 0 for item in retrieved]


def hit_at_k(flags: Sequence[int], k: int) -> float:
    """1.0 when at least one of the top ``k`` results is relevant."""
    return 1.0 if any(flags[:k]) else 0.0


def precision_at_k(flags: Sequence[int], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    window = flags[:k]
    return sum(window) / k


def document_recall_at_k(
    retrieved: Sequence[Retrieved], gold_doc_ids: Sequence[str], k: int
) -> float:
    """Fraction of the gold documents represented in the top ``k`` chunks."""
    if not gold_doc_ids:
        return 0.0
    found = {item.doc_id for item in retrieved[:k]} & set(gold_doc_ids)
    return len(found) / len(set(gold_doc_ids))


def reciprocal_rank(flags: Sequence[int]) -> float:
    """1 / rank of the first relevant result, or 0 if there is none."""
    for position, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / position
    return 0.0


def dcg(flags: Sequence[int], k: int) -> float:
    """Discounted cumulative gain with binary relevance."""
    return sum(flag / math.log2(position + 1) for position, flag in enumerate(flags[:k], start=1))


def ndcg_at_k(flags: Sequence[int], k: int) -> float:
    """DCG normalised by the best achievable ordering of the same results.

    The ideal ranking is taken over the relevant items actually present in the
    result list, so a query whose gold document was never retrieved scores 0
    rather than being excluded.
    """
    ideal = dcg(sorted(flags, reverse=True), k)
    if ideal <= 0.0:
        return 0.0
    return dcg(flags, k) / ideal


def answer_in_context(
    retrieved: Sequence[Retrieved], keywords: Sequence[str], k: int
) -> float:
    """1.0 when every answer keyword appears in the top ``k`` chunks.

    Keywords are matched on stemmed tokens. A multi-word keyword such as
    "atrioventricular node" must appear as a contiguous token sequence, which
    prevents a match on two unrelated mentions in different sentences.
    """
    if not keywords:
        return float("nan")

    tokens: list[str] = []
    for item in retrieved[:k]:
        tokens.extend(stem(token) for token in tokenize(item.chunk.text))
    if not tokens:
        return 0.0

    joined = " " + " ".join(tokens) + " "
    for keyword in keywords:
        needle = " ".join(stem(token) for token in tokenize(keyword))
        if not needle:
            continue
        if f" {needle} " not in joined:
            return 0.0
    return 1.0


# -- aggregation ------------------------------------------------------------
#: A sample this many times the median is treated as an operating-system stall
#: (page fault, antivirus scan, scheduler pre-emption) rather than as the cost
#: of the retrieval being measured.
OUTLIER_FACTOR = 10.0


@dataclass(frozen=True)
class LatencySummary:
    """Latency of a repeated measurement, in milliseconds.

    Several statistics are kept because a plain mean is not trustworthy on a
    general-purpose machine. A single scheduling stall of a few seconds is
    enough to move the mean by an order of magnitude while leaving the median
    untouched, so the median is the figure to compare between retrievers and
    the mean is reported next to it as a check. ``outliers`` counts how many
    samples exceeded the median by more than :data:`OUTLIER_FACTOR`; if it is
    non-zero, the mean should not be quoted on its own.
    """

    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    samples: int
    #: Mean after discarding the slowest five per cent of samples.
    trimmed_mean_ms: float = 0.0
    #: Samples more than OUTLIER_FACTOR times the median.
    outliers: int = 0

    @property
    def is_noisy(self) -> bool:
        """True when the mean has been distorted by stalls."""
        return self.outliers > 0 or (
            self.median_ms > 0.0 and self.mean_ms > 3.0 * self.median_ms
        )

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencySummary":
        if not samples:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        ordered = sorted(samples)
        median = statistics.median(ordered)
        # Nearest-rank percentile: simple, and exact for the sample sizes here.
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        keep = max(1, len(ordered) - max(1, int(round(0.05 * len(ordered)))))
        return cls(
            mean_ms=round(statistics.fmean(ordered), 3),
            median_ms=round(median, 3),
            p95_ms=round(ordered[index], 3),
            min_ms=round(ordered[0], 3),
            max_ms=round(ordered[-1], 3),
            samples=len(ordered),
            trimmed_mean_ms=round(statistics.fmean(ordered[:keep]), 3),
            outliers=sum(1 for value in ordered if value > OUTLIER_FACTOR * median),
        )


@dataclass
class QueryResult:
    """Per-question metrics, kept for the error analysis."""

    qid: str
    question: str
    kind: str
    gold_doc_ids: tuple[str, ...]
    retrieved_doc_ids: list[str]
    retrieved_chunk_ids: list[str]
    scores: list[float]
    flags: list[int]
    metrics: dict[str, float]
    latency_ms: float


@dataclass
class MetricSummary:
    """Metrics averaged over the evaluation set."""

    values: dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


def evaluate_query(
    question: EvalQuestion,
    retrieved: Sequence[Retrieved],
    *,
    k_values: Sequence[int] = (1, 3, 5),
    latency_ms: float = 0.0,
) -> QueryResult:
    """Compute every metric for a single question."""
    flags = relevance_flags(retrieved, question.gold_doc_ids)
    top_k = max(k_values)

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hit@{k}"] = hit_at_k(flags, k)
        metrics[f"precision@{k}"] = precision_at_k(flags, k)
        metrics[f"recall@{k}"] = document_recall_at_k(retrieved, question.gold_doc_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(flags, k)
        metrics[f"answer_in_context@{k}"] = answer_in_context(
            retrieved, question.answer_keywords, k
        )
    metrics["mrr"] = reciprocal_rank(flags)
    metrics[f"mrr@{top_k}"] = reciprocal_rank(flags[:top_k])
    # The number of words handed to the generator. Recorded because the two
    # chunking strategies produce chunks of different sizes, so the same k does
    # not give the two runs the same amount of context; any comparison of
    # answer_in_context has to be read alongside this figure.
    metrics[f"context_words@{top_k}"] = float(
        sum(item.chunk.word_count for item in retrieved[:top_k])
    )

    return QueryResult(
        qid=question.qid,
        question=question.question,
        kind=question.kind,
        gold_doc_ids=question.gold_doc_ids,
        retrieved_doc_ids=[item.doc_id for item in retrieved],
        retrieved_chunk_ids=[item.chunk_id for item in retrieved],
        scores=[round(item.score, 5) for item in retrieved],
        flags=flags,
        metrics=metrics,
        latency_ms=latency_ms,
    )


def aggregate(results: Sequence[QueryResult]) -> MetricSummary:
    """Average each metric over the questions, ignoring undefined values.

    ``answer_in_context`` is NaN for a question with no annotated keywords, so
    those questions are excluded from that metric alone rather than from the
    whole row.
    """
    if not results:
        return MetricSummary()

    totals: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.metrics.items():
            if isinstance(value, float) and math.isnan(value):
                continue
            totals.setdefault(key, []).append(value)

    return MetricSummary(
        {key: round(statistics.fmean(values), 4) for key, values in totals.items()}
    )


def aggregate_by_kind(results: Sequence[QueryResult]) -> dict[str, MetricSummary]:
    """Aggregate separately for each question type."""
    grouped: dict[str, list[QueryResult]] = {}
    for result in results:
        grouped.setdefault(result.kind, []).append(result)
    return {kind: aggregate(items) for kind, items in sorted(grouped.items())}
