#!/usr/bin/env python
"""Run the full experiment grid and write everything the report needs.

    python scripts/run_experiments.py
    python scripts/run_experiments.py --repeats 10 --output results

Equivalent to ``python -m adaptive_rag evaluate``, kept as a script so the
experiments can be reproduced with one command and no arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a checkout without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config import EVAL_PATH, RESULTS_DIR, RAGConfig  # noqa: E402
from adaptive_rag.evaluation import (  # noqa: E402
    alpha_table,
    full_table,
    headline_table,
    k_sweep_table,
    kind_table,
    latency_note,
    run_experiments,
    save_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=EVAL_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--repeats", type=int, default=7,
                        help="timed repeats per query (default: 7)")
    parser.add_argument("--embeddings", choices=("auto", "sentence-transformers", "lsa"),
                        default="auto")
    parser.add_argument("--threads", type=int, default=1,
                        help="PyTorch threads during timing; 0 leaves the default")
    parser.add_argument("--quick", action="store_true",
                        help="skip both sweeps for a faster run")
    args = parser.parse_args()

    from dataclasses import replace

    base = RAGConfig()
    config = base.with_(dense=replace(base.dense, backend=args.embeddings))

    print("Adaptive-RAG experiments")
    print("=" * 24)
    report = run_experiments(
        config,
        args.questions,
        repeats=args.repeats,
        include_alpha_sweep=not args.quick,
        include_k_sweep=not args.quick,
        torch_threads=args.threads or None,
    )

    print()
    print(latency_note(report))
    print()
    print(headline_table(report))
    print()
    print(full_table(report))
    print()
    print(kind_table(report, "answer_in_context@3"))
    if report.alpha_sweep:
        print()
        print(alpha_table(report))
    if report.k_sweep:
        print()
        print(k_sweep_table(report))

    written = save_report(report, args.output)
    print()
    for label, path in written.items():
        print(f"{label:<15} {path}")

    best = report.best("answer_in_context@3")
    print()
    print(f"Best configuration by answer-in-context@3: {best.label} "
          f"({best.metrics.get('answer_in_context@3', 0.0):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
