"""Command line interface.

    python -m adaptive_rag stats
    python -m adaptive_rag query "why do corals bleach?" --retriever hybrid
    python -m adaptive_rag ask "what sets the heart rhythm?" --chunking paragraph
    python -m adaptive_rag compare "how is magma generated at a rift?"
    python -m adaptive_rag evaluate --repeats 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .chunking import chunk_statistics, get_chunker
from .config import (
    CHUNKING_STRATEGIES,
    DOCUMENTS_DIR,
    EVAL_PATH,
    RESULTS_DIR,
    RETRIEVER_NAMES,
    RAGConfig,
)
from .loaders import corpus_statistics, load_corpus
from .pipeline import RAGPipeline, SharedCorpus, build_pipeline


def _config_from_args(args: argparse.Namespace) -> RAGConfig:
    """Translate parsed arguments into a :class:`RAGConfig`."""
    base = RAGConfig()
    chunking_params = base.chunking_params
    if getattr(args, "chunk_size", None) or getattr(args, "overlap", None) is not None:
        from dataclasses import replace

        chunking_params = replace(
            chunking_params,
            chunk_size=getattr(args, "chunk_size", None) or chunking_params.chunk_size,
            overlap=(
                chunking_params.overlap
                if getattr(args, "overlap", None) is None
                else args.overlap
            ),
        )

    from dataclasses import replace

    dense = replace(
        base.dense,
        backend=getattr(args, "embeddings", base.dense.backend),
        use_faiss=not getattr(args, "no_faiss", False),
    )
    hybrid = replace(
        base.hybrid,
        alpha=getattr(args, "alpha", base.hybrid.alpha),
        fusion=getattr(args, "fusion", base.hybrid.fusion),
    )
    generation = replace(
        base.generation,
        backend=getattr(args, "generator", base.generation.backend),
    )

    return base.with_(
        documents_dir=Path(getattr(args, "documents", DOCUMENTS_DIR)),
        chunking=getattr(args, "chunking", base.chunking),
        retriever=getattr(args, "retriever", base.retriever),
        top_k=getattr(args, "top_k", base.top_k),
        chunking_params=chunking_params,
        dense=dense,
        hybrid=hybrid,
        generation=generation,
    )


def _print_heading(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


# -- commands ---------------------------------------------------------------
def command_stats(args: argparse.Namespace) -> int:
    """Corpus and chunking statistics, with no retrieval."""
    documents = load_corpus(args.documents)
    stats = corpus_statistics(documents)

    _print_heading("Corpus")
    print(f"documents   : {stats['documents']}")
    print(f"total words : {stats['total_words']}")
    print(
        f"words/doc   : mean {stats['mean_words']}, "
        f"min {stats['min_words']}, max {stats['max_words']}"
    )

    print()
    for document in documents:
        print(f"  {document.doc_id:<20} {document.word_count:>5} words  {document.title}")

    config = _config_from_args(args)
    for strategy in CHUNKING_STRATEGIES:
        chunker = get_chunker(strategy, config.chunking_params)
        chunks = chunker.chunk_corpus(documents)
        chunk_stats = chunk_statistics(chunks)
        _print_heading(f"Chunking: {strategy} ({chunker!r})")
        print(f"chunks      : {chunk_stats['chunks']}")
        print(
            f"words/chunk : mean {chunk_stats['mean_words']}, "
            f"median {chunk_stats['median_words']}, "
            f"min {chunk_stats['min_words']}, max {chunk_stats['max_words']}"
        )
        print(
            f"redundancy  : {chunk_stats['total_words'] / stats['total_words']:.2f}x "
            f"the corpus word count"
        )
    return 0


def command_query(args: argparse.Namespace) -> int:
    """Show the chunks a retriever returns, without generating an answer."""
    config = _config_from_args(args)
    pipeline = build_pipeline(config)
    results = pipeline.retrieve(args.question, config.top_k)

    _print_heading(f"{config.retriever} + {config.chunking} chunking")
    print(f"query: {args.question}")
    print(f"index: {pipeline.retriever.describe()}")
    if not results:
        print("\nno chunk scored above zero for this query")
        return 0

    for item in results:
        print(f"\n[{item.rank}] score={item.score:.4f}  {item.chunk.display_source()}")
        print(f"    id={item.chunk.chunk_id} ({item.chunk.word_count} words)")
        print(f"    {_snippet(item.chunk.text)}")
    return 0


def command_ask(args: argparse.Namespace) -> int:
    """Run the full pipeline and print the generated answer."""
    config = _config_from_args(args)
    pipeline = build_pipeline(config)
    answer, contexts = pipeline.answer(args.question, config.top_k)

    _print_heading("Answer")
    print(f"query    : {args.question}")
    print(f"pipeline : {config.chunking} chunking, {config.retriever} retrieval, "
          f"{answer.backend} generation")
    if answer.note:
        print(f"note     : {answer.note}")
    print()
    print(answer.text)

    if contexts:
        _print_heading("Sources")
        print(answer.format_citations())
    return 0


def command_compare(args: argparse.Namespace) -> int:
    """Run one query through all three retrievers side by side."""
    config = _config_from_args(args)
    corpus = SharedCorpus(config)

    for retriever_name in RETRIEVER_NAMES:
        pipeline = corpus.pipeline(config.chunking, retriever_name)
        results = pipeline.retrieve(args.question, config.top_k)
        _print_heading(f"{retriever_name} ({config.chunking} chunking)")
        if not results:
            print("  no results")
            continue
        for item in results:
            print(
                f"  [{item.rank}] {item.score:.4f}  {item.doc_id:<20} "
                f"{_snippet(item.chunk.text, 90)}"
            )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """Run the full experiment grid and write the results."""
    from .evaluation import (
        alpha_table,
        full_table,
        headline_table,
        k_sweep_table,
        kind_table,
        latency_note,
        run_experiments,
        save_report,
    )

    config = _config_from_args(args)
    chunkings = args.chunkings or list(CHUNKING_STRATEGIES)
    retrievers = args.retrievers or list(RETRIEVER_NAMES)

    print("Running experiments")
    report = run_experiments(
        config,
        args.questions,
        chunkings=chunkings,
        retrievers=retrievers,
        repeats=args.repeats,
        include_alpha_sweep=not args.no_alpha_sweep,
        include_k_sweep=not args.no_k_sweep,
    )

    _print_heading("Retrieval comparison")
    print(latency_note(report))
    print()
    print(headline_table(report))
    _print_heading("All runs")
    print(full_table(report))
    _print_heading("Accuracy by question type")
    print(kind_table(report, "answer_in_context@3"))
    if report.alpha_sweep:
        _print_heading("Hybrid alpha sweep")
        print(alpha_table(report))
    if report.k_sweep:
        _print_heading("Accuracy against k and context budget")
        print(k_sweep_table(report))

    written = save_report(report, args.output)
    _print_heading("Written")
    for label, path in written.items():
        print(f"  {label:<15} {path}")
    return 0


def _snippet(text: str, width: int = 160) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= width:
        return flattened
    return flattened[: width - 3].rstrip() + "..."


# -- argument parsing -------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-rag",
        description="Adaptive-RAG: compare BM25, dense and hybrid retrieval over a "
        "small document collection.",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=DOCUMENTS_DIR,
        help=f"corpus directory (default: {DOCUMENTS_DIR})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_retrieval_options(sub: argparse.ArgumentParser, *, with_retriever: bool = True) -> None:
        sub.add_argument(
            "--chunking",
            choices=CHUNKING_STRATEGIES,
            default="paragraph",
            help="chunking strategy (default: paragraph)",
        )
        if with_retriever:
            sub.add_argument(
                "--retriever",
                choices=RETRIEVER_NAMES,
                default="hybrid",
                help="retrieval strategy (default: hybrid)",
            )
        sub.add_argument("--top-k", type=int, default=5, dest="top_k",
                         help="chunks to retrieve (default: 5)")
        sub.add_argument("--chunk-size", type=int, default=None,
                         help="words per fixed-size chunk (default: 300)")
        sub.add_argument("--overlap", type=int, default=None,
                         help="overlap between fixed-size chunks (default: 50)")
        sub.add_argument(
            "--embeddings",
            choices=("auto", "sentence-transformers", "lsa"),
            default="auto",
            help="dense embedding backend (default: auto)",
        )
        sub.add_argument("--no-faiss", action="store_true",
                         help="use the NumPy search path instead of FAISS")
        sub.add_argument("--alpha", type=float, default=0.5,
                         help="hybrid dense weight, 0=BM25 1=dense (default: 0.5)")
        sub.add_argument("--fusion", choices=("score", "rrf"), default="score",
                         help="hybrid fusion rule (default: score)")

    stats = subparsers.add_parser("stats", help="corpus and chunking statistics")
    stats.add_argument("--chunk-size", type=int, default=None)
    stats.add_argument("--overlap", type=int, default=None)
    stats.set_defaults(func=command_stats)

    query = subparsers.add_parser("query", help="show retrieved chunks for a question")
    query.add_argument("question")
    add_retrieval_options(query)
    query.set_defaults(func=command_query)

    ask = subparsers.add_parser("ask", help="retrieve and generate an answer")
    ask.add_argument("question")
    add_retrieval_options(ask)
    ask.add_argument(
        "--generator",
        choices=("extractive", "claude", "auto"),
        default="extractive",
        help="answer generator (default: extractive, which needs no API key)",
    )
    ask.set_defaults(func=command_ask)

    compare = subparsers.add_parser(
        "compare", help="run one question through all three retrievers"
    )
    compare.add_argument("question")
    add_retrieval_options(compare, with_retriever=False)
    compare.set_defaults(func=command_compare)

    evaluate = subparsers.add_parser("evaluate", help="run the full experiment grid")
    evaluate.add_argument("--questions", type=Path, default=EVAL_PATH,
                          help=f"evaluation set (default: {EVAL_PATH})")
    evaluate.add_argument("--output", type=Path, default=RESULTS_DIR,
                          help=f"output directory (default: {RESULTS_DIR})")
    evaluate.add_argument("--repeats", type=int, default=5,
                          help="latency measurements per query (default: 5)")
    evaluate.add_argument("--chunkings", nargs="*", choices=CHUNKING_STRATEGIES,
                          default=None, help="restrict the chunking strategies tested")
    evaluate.add_argument("--retrievers", nargs="*", choices=RETRIEVER_NAMES,
                          default=None, help="restrict the retrievers tested")
    evaluate.add_argument("--no-alpha-sweep", action="store_true",
                          help="skip the hybrid alpha sensitivity sweep")
    evaluate.add_argument("--no-k-sweep", action="store_true",
                          help="skip the sweep of k and context budget")
    evaluate.add_argument("--top-k", type=int, default=5, dest="top_k")
    evaluate.add_argument("--chunk-size", type=int, default=None)
    evaluate.add_argument("--overlap", type=int, default=None)
    evaluate.add_argument("--embeddings", choices=("auto", "sentence-transformers", "lsa"),
                          default="auto")
    evaluate.add_argument("--no-faiss", action="store_true")
    evaluate.add_argument("--alpha", type=float, default=0.5)
    evaluate.add_argument("--fusion", choices=("score", "rrf"), default="score")
    evaluate.set_defaults(func=command_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
