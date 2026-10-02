"""The experiment grid: two chunking strategies by three retrievers.

Every combination is evaluated on the same annotated question set, and both
retrieval quality and query latency are recorded. Latency is measured by
repeating each query several times after a warm-up call, because the first call
to a retriever pays one-off costs (lazily imported modules, memory allocation,
the first forward pass through the embedding model) that are not representative
of steady-state behaviour.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from ..config import CHUNKING_STRATEGIES, RETRIEVER_NAMES, RAGConfig
from ..pipeline import SharedCorpus
from ..retrieval import build_retriever, faiss_available, sentence_transformers_available
from .metrics import (
    EvalQuestion,
    LatencySummary,
    MetricSummary,
    QueryResult,
    aggregate,
    aggregate_by_kind,
    evaluate_query,
    load_questions,
    validate_questions,
)

DEFAULT_K_VALUES = (1, 3, 5)
DEFAULT_REPEATS = 5
DEFAULT_WARMUP = 5
#: Threads the embedding model may use while latency is being measured.
DEFAULT_TORCH_THREADS = 1


@dataclass
class RunResult:
    """Everything measured for one cell of the grid."""

    chunking: str
    retriever: str
    top_k: int
    chunk_count: int
    chunk_stats: dict[str, float | int]
    metrics: dict[str, float]
    latency: LatencySummary
    index_seconds: float
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    retriever_config: dict[str, object] = field(default_factory=dict)
    query_results: list[QueryResult] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.chunking}+{self.retriever}"

    def accuracy(self) -> float:
        """The headline accuracy figure: answer-in-context at the top k."""
        return self.metrics.get(f"answer_in_context@{self.top_k}", 0.0)

    def summary_row(self) -> dict[str, object]:
        """One flat row for the results table."""
        row: dict[str, object] = {
            "chunking": self.chunking,
            "retriever": self.retriever,
            "chunks": self.chunk_count,
            "mean_chunk_words": self.chunk_stats.get("mean_words", 0),
        }
        for key in (
            "hit@1",
            "hit@3",
            "hit@5",
            "recall@5",
            "precision@5",
            "mrr",
            "ndcg@5",
            "answer_in_context@5",
            "context_words@5",
        ):
            if key in self.metrics:
                row[key] = self.metrics[key]
        row["median_latency_ms"] = self.latency.median_ms
        row["trimmed_mean_latency_ms"] = self.latency.trimmed_mean_ms
        row["p95_latency_ms"] = self.latency.p95_ms
        row["raw_mean_latency_ms"] = self.latency.mean_ms
        row["latency_stalls"] = self.latency.outliers
        row["index_seconds"] = round(self.index_seconds, 3)
        return row


@dataclass
class ExperimentReport:
    """The full set of runs plus the environment they were produced in."""

    runs: list[RunResult]
    questions: int
    k_values: tuple[int, ...]
    repeats: int
    environment: dict[str, object]
    corpus_stats: dict[str, float | int]
    alpha_sweep: list[dict[str, float]] = field(default_factory=list)
    k_sweep: list[dict[str, float | str]] = field(default_factory=list)

    def run(self, chunking: str, retriever: str) -> RunResult:
        for item in self.runs:
            if item.chunking == chunking and item.retriever == retriever:
                return item
        raise KeyError(f"no run for {chunking}+{retriever}")

    def best(self, metric: str = "answer_in_context@5") -> RunResult:
        return max(self.runs, key=lambda r: r.metrics.get(metric, 0.0))


def configure_threads(threads: int | None = DEFAULT_TORCH_THREADS) -> int | None:
    """Pin the number of threads PyTorch may use, and report what was set.

    Left unpinned, the intra-op thread pool makes per-query latency vary by an
    order of magnitude between otherwise identical calls, which swamps the
    difference between the retrievers being compared. One thread also matches
    how a service under concurrent load would actually execute a single query.
    Pass ``None`` to leave the default in place.
    """
    if threads is None:
        return None
    try:  # pragma: no cover - optional dependency
        import torch

        torch.set_num_threads(threads)
        return threads
    except ImportError:
        return None


def describe_environment() -> dict[str, object]:
    """Record what the numbers were measured on, for reproducibility."""
    environment: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "sentence_transformers": sentence_transformers_available(),
        "faiss": faiss_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:  # pragma: no cover - optional dependency
        import numpy

        environment["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:  # pragma: no cover - optional dependency
        import torch

        environment["torch"] = torch.__version__
        environment["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return environment


def measure_index_build(
    chunks: Sequence,
    retriever_name: str,
    config: RAGConfig,
) -> float:
    """Time a cold index build.

    A fresh retriever is constructed so the measurement is not contaminated by
    the shared, already-populated indexes used for the query timings.
    """
    started = time.perf_counter()
    build_retriever(retriever_name, chunks, config)
    return time.perf_counter() - started


def _time_query(pipeline, question: str, k: int, repeats: int) -> list[float]:
    """Return one latency sample per repeat, in milliseconds."""
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        pipeline.retrieve(question, k)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def run_single(
    corpus: SharedCorpus,
    chunking: str,
    retriever_name: str,
    questions: Sequence[EvalQuestion],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    measure_build: bool = True,
) -> RunResult:
    """Evaluate one chunking and retriever combination."""
    pipeline = corpus.pipeline(chunking, retriever_name)
    top_k = max(k_values)

    # Warm up on a real question so that lazy imports, the first embedding
    # forward pass and any cache allocation are excluded from the timings.
    for _ in range(warmup):
        pipeline.retrieve(questions[0].question, top_k)

    query_results: list[QueryResult] = []
    all_samples: list[float] = []
    for question in questions:
        samples = _time_query(pipeline, question.question, top_k, repeats)
        all_samples.extend(samples)
        retrieved = pipeline.retrieve(question.question, top_k)
        query_results.append(
            evaluate_query(
                question,
                retrieved,
                k_values=k_values,
                latency_ms=round(statistics.median(samples), 3),
            )
        )

    index_seconds = 0.0
    if measure_build:
        index_seconds = measure_index_build(
            corpus.chunks(chunking), retriever_name, pipeline.config
        )

    summary: MetricSummary = aggregate(query_results)
    return RunResult(
        chunking=chunking,
        retriever=retriever_name,
        top_k=top_k,
        chunk_count=len(corpus.chunks(chunking)),
        chunk_stats=dict(pipeline.report.chunk_stats) if pipeline.report else {},
        metrics=summary.as_dict(),
        latency=LatencySummary.from_samples(all_samples),
        index_seconds=index_seconds,
        by_kind={
            kind: value.as_dict()
            for kind, value in aggregate_by_kind(query_results).items()
        },
        retriever_config=pipeline.retriever.describe(),
        query_results=query_results,
    )


def sweep_k(
    corpus: SharedCorpus,
    questions: Sequence[EvalQuestion],
    *,
    chunkings: Sequence[str] = CHUNKING_STRATEGIES,
    retrievers: Sequence[str] = RETRIEVER_NAMES,
    k_values: Sequence[int] = (1, 2, 3, 5, 8, 10),
) -> list[dict[str, float | str]]:
    """Accuracy as a function of ``k``, with the context volume it costs.

    The two chunking strategies produce chunks of different average length, so
    comparing them at a single ``k`` also compares two different context
    budgets. This sweep records the words retrieved at every ``k``, which lets
    the two strategies be compared at a matched budget rather than a matched
    number of chunks.
    """
    rows: list[dict[str, float | str]] = []
    for chunking in chunkings:
        for retriever_name in retrievers:
            pipeline = corpus.pipeline(chunking, retriever_name)
            for k in k_values:
                results = [
                    evaluate_query(
                        question, pipeline.retrieve(question.question, k), k_values=(k,)
                    )
                    for question in questions
                ]
                summary = aggregate(results)
                rows.append(
                    {
                        "chunking": chunking,
                        "retriever": retriever_name,
                        "k": k,
                        "answer_in_context": summary.get(f"answer_in_context@{k}"),
                        "hit": summary.get(f"hit@{k}"),
                        "recall": summary.get(f"recall@{k}"),
                        "context_words": round(summary.get(f"context_words@{k}"), 1),
                    }
                )
    return rows


def sweep_alpha(
    corpus: SharedCorpus,
    chunking: str,
    questions: Sequence[EvalQuestion],
    *,
    alphas: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
    k: int = 5,
) -> list[dict[str, float]]:
    """Measure hybrid accuracy as the dense weight varies.

    ``alpha=0`` is pure BM25 and ``alpha=1`` is pure dense retrieval, so the
    sweep shows whether the fusion is genuinely better than either component or
    merely inherits the better of the two.
    """
    pipeline = corpus.pipeline(chunking, "hybrid")
    retriever = pipeline.retriever
    rows: list[dict[str, float]] = []

    for alpha in alphas:
        results: list[QueryResult] = []
        for question in questions:
            retrieved = retriever.search_with_alpha(question.question, k, alpha)
            results.append(evaluate_query(question, retrieved, k_values=(1, 3, k)))
        summary = aggregate(results)
        rows.append(
            {
                "alpha": round(float(alpha), 2),
                f"answer_in_context@{k}": summary.get(f"answer_in_context@{k}"),
                f"hit@{k}": summary.get(f"hit@{k}"),
                "mrr": summary.get("mrr"),
                f"ndcg@{k}": summary.get(f"ndcg@{k}"),
            }
        )
    return rows


def run_experiments(
    config: RAGConfig | None = None,
    questions_path: Path | str | None = None,
    *,
    chunkings: Sequence[str] = CHUNKING_STRATEGIES,
    retrievers: Sequence[str] = RETRIEVER_NAMES,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    repeats: int = DEFAULT_REPEATS,
    include_alpha_sweep: bool = True,
    include_k_sweep: bool = True,
    torch_threads: int | None = DEFAULT_TORCH_THREADS,
    verbose: bool = True,
) -> ExperimentReport:
    """Run the whole grid and return a report."""
    from ..config import EVAL_PATH
    from ..loaders import corpus_statistics

    config = config or RAGConfig()
    questions = load_questions(questions_path or EVAL_PATH)
    pinned_threads = configure_threads(torch_threads)

    corpus = SharedCorpus(config)
    validate_questions(questions, (doc.doc_id for doc in corpus.documents))

    runs: list[RunResult] = []
    for chunking in chunkings:
        for retriever_name in retrievers:
            if verbose:
                print(f"  running {chunking:<9} + {retriever_name:<7} ...", end=" ", flush=True)
            result = run_single(
                corpus,
                chunking,
                retriever_name,
                questions,
                k_values=k_values,
                repeats=repeats,
            )
            runs.append(result)
            if verbose:
                print(
                    f"accuracy={result.accuracy():.3f} "
                    f"latency={result.latency.median_ms:.2f} ms (median)"
                )

    alpha_sweep: list[dict[str, float]] = []
    if include_alpha_sweep and "hybrid" in retrievers:
        sweep_chunking = "paragraph" if "paragraph" in chunkings else chunkings[0]
        if verbose:
            print(f"  sweeping hybrid alpha on {sweep_chunking} chunks ...")
        alpha_sweep = sweep_alpha(corpus, sweep_chunking, questions, k=max(k_values))

    k_sweep: list[dict[str, float | str]] = []
    if include_k_sweep:
        if verbose:
            print("  sweeping k to expose the context-budget difference ...")
        k_sweep = sweep_k(
            corpus, questions, chunkings=chunkings, retrievers=retrievers
        )

    environment = describe_environment()
    environment["pinned_torch_threads"] = pinned_threads
    environment["warmup_queries"] = DEFAULT_WARMUP

    return ExperimentReport(
        runs=runs,
        questions=len(questions),
        k_values=tuple(k_values),
        repeats=repeats,
        environment=environment,
        corpus_stats=corpus_statistics(corpus.documents),
        alpha_sweep=alpha_sweep,
        k_sweep=k_sweep,
    )


# -- output -----------------------------------------------------------------
def _format_table(rows: Sequence[dict[str, object]]) -> str:
    """Render rows as a GitHub-flavoured Markdown table."""
    if not rows:
        return "(no results)"
    headers = list(rows[0].keys())
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    lines = [
        "| " + " | ".join(header.ljust(widths[header]) for header in headers) + " |",
        "| " + " | ".join("-" * widths[header] for header in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers)
            + " |"
        )
    return "\n".join(lines)


def latency_note(report: ExperimentReport) -> str:
    """State how latency was measured, and flag any run that was disturbed."""
    threads = report.environment.get("pinned_torch_threads")
    lines = [
        f"Latency: {report.repeats} timed repeats per question after "
        f"{report.environment.get('warmup_queries', DEFAULT_WARMUP)} warm-up "
        "queries, measured on the retrieval step only (no generation).",
    ]
    if threads:
        lines.append(
            f"PyTorch was pinned to {threads} thread so that repeated calls are "
            "comparable."
        )
    lines.append(
        "The median is the figure to compare; the raw mean is shown next to it "
        "because it is sensitive to operating-system stalls on a shared machine."
    )
    disturbed = [run for run in report.runs if run.latency.is_noisy]
    if disturbed:
        details = ", ".join(
            f"{run.label} (max {run.latency.max_ms:.0f} ms, "
            f"{run.latency.outliers} stall(s))"
            for run in disturbed
        )
        lines.append(
            "Measurement warning: the following runs contain samples far above "
            f"their own median and their raw means are not meaningful: {details}."
        )
    return "\n".join(lines)


def headline_table(report: ExperimentReport) -> str:
    """The retriever comparison requested by the brief: accuracy and latency.

    Accuracy is reported at three cut-offs. At k=5 a corpus of ten clearly
    separated documents is easy enough that every retriever finds the right
    document, so the interesting comparison is at k=1 and k=3, where the
    ranking still has to be correct.
    """
    rows = []
    for retriever in RETRIEVER_NAMES:
        for run in report.runs:
            if run.retriever != retriever:
                continue
            row: dict[str, object] = {
                "retrieval": retriever,
                "chunking": run.chunking,
            }
            for k in report.k_values:
                row[f"accuracy@{k}"] = f"{run.metrics.get(f'answer_in_context@{k}', 0.0):.3f}"
            row["hit@1"] = f"{run.metrics.get('hit@1', 0.0):.3f}"
            row["mrr"] = f"{run.metrics.get('mrr', 0.0):.3f}"
            row["median latency (ms)"] = f"{run.latency.median_ms:.3f}"
            row["trimmed mean (ms)"] = f"{run.latency.trimmed_mean_ms:.3f}"
            row["p95 latency (ms)"] = f"{run.latency.p95_ms:.3f}"
            row["raw mean (ms)"] = f"{run.latency.mean_ms:.3f}"
            row["stalls"] = run.latency.outliers
            rows.append(row)
    return _format_table(rows)


def kind_table(report: ExperimentReport, metric: str = "answer_in_context@3") -> str:
    """One row per run, one column per question type.

    This is where the retrievers separate. The "confusable" questions are
    written so that their strongest keywords point at the wrong document, and
    the "paraphrased" ones avoid the vocabulary of the source altogether.
    """
    kinds = sorted({kind for run in report.runs for kind in run.by_kind})
    rows = []
    for run in report.runs:
        row: dict[str, object] = {"chunking": run.chunking, "retriever": run.retriever}
        for kind in kinds:
            value = run.by_kind.get(kind, {}).get(metric)
            row[kind] = f"{value:.3f}" if value is not None else "-"
        row["all"] = f"{run.metrics.get(metric, 0.0):.3f}"
        rows.append(row)
    header = f"Metric: {metric}"
    return "\n\n".join([header, _format_table(rows)])


def full_table(report: ExperimentReport) -> str:
    return _format_table([run.summary_row() for run in report.runs])


def alpha_table(report: ExperimentReport) -> str:
    if not report.alpha_sweep:
        return "(alpha sweep not run)"
    return _format_table([{k: v for k, v in row.items()} for row in report.alpha_sweep])


def k_sweep_table(report: ExperimentReport) -> str:
    if not report.k_sweep:
        return "(k sweep not run)"
    return _format_table([dict(row) for row in report.k_sweep])


def save_report(
    report: ExperimentReport,
    directory: Path | str,
    *,
    write_per_query: bool = True,
) -> dict[str, Path]:
    """Write the report to CSV, JSON and Markdown. Returns the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # Summary CSV.
    summary_rows = [run.summary_row() for run in report.runs]
    summary_path = directory / "results.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    written["summary_csv"] = summary_path

    # Full JSON, including every per-question outcome.
    payload = {
        "questions": report.questions,
        "k_values": list(report.k_values),
        "repeats": report.repeats,
        "environment": report.environment,
        "corpus": report.corpus_stats,
        "alpha_sweep": report.alpha_sweep,
        "k_sweep": report.k_sweep,
        "runs": [
            {
                "chunking": run.chunking,
                "retriever": run.retriever,
                "top_k": run.top_k,
                "chunk_count": run.chunk_count,
                "chunk_stats": run.chunk_stats,
                "metrics": run.metrics,
                "latency": asdict(run.latency),
                "index_seconds": round(run.index_seconds, 4),
                "by_kind": run.by_kind,
                "retriever_config": run.retriever_config,
            }
            for run in report.runs
        ],
    }
    json_path = directory / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    written["json"] = json_path

    # Markdown tables for pasting into the report.
    k = max(report.k_values)
    lines = [
        "# Experiment results",
        "",
        f"Questions: {report.questions}. Latency repeats per query: {report.repeats}.",
        f"Corpus: {report.corpus_stats.get('documents')} documents, "
        f"{report.corpus_stats.get('total_words')} words.",
        f"Measured on {report.environment.get('platform')} with Python "
        f"{report.environment.get('python')} at {report.environment.get('timestamp')}.",
        "",
        "## Retrieval comparison",
        "",
        latency_note(report),
        "",
        headline_table(report),
        "",
        "## All runs",
        "",
        full_table(report),
        "",
        "## Accuracy by question type",
        "",
        kind_table(report, "answer_in_context@3"),
        "",
        kind_table(report, "hit@1"),
        "",
        "## Hybrid alpha sweep",
        "",
        f"alpha=0 is pure BM25, alpha=1 is pure dense retrieval (top {k}).",
        "",
        alpha_table(report),
        "",
        "## Accuracy against k and context budget",
        "",
        "The two chunking strategies produce chunks of different average length,",
        "so the same k is not the same amount of context. Compare rows with a",
        "similar context_words figure rather than a similar k.",
        "",
        k_sweep_table(report),
        "",
    ]
    markdown_path = directory / "results.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    written["markdown"] = markdown_path

    if write_per_query:
        per_query_path = directory / "per_query.csv"
        with per_query_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "chunking",
                    "retriever",
                    "qid",
                    "kind",
                    "question",
                    "gold_doc_ids",
                    "retrieved_doc_ids",
                    "hit@1",
                    f"hit@{k}",
                    "mrr",
                    f"answer_in_context@{k}",
                    "latency_ms",
                ]
            )
            for run in report.runs:
                for result in run.query_results:
                    writer.writerow(
                        [
                            run.chunking,
                            run.retriever,
                            result.qid,
                            result.kind,
                            result.question,
                            "|".join(result.gold_doc_ids),
                            "|".join(result.retrieved_doc_ids),
                            result.metrics.get("hit@1", 0.0),
                            result.metrics.get(f"hit@{k}", 0.0),
                            round(result.metrics.get("mrr", 0.0), 4),
                            result.metrics.get(f"answer_in_context@{k}", ""),
                            result.latency_ms,
                        ]
                    )
        written["per_query_csv"] = per_query_path

    return written
