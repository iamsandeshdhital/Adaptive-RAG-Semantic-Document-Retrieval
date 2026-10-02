#!/usr/bin/env python
"""Render the figures used in the report from ``results/results.json``.

    python scripts/run_experiments.py     # produces results/results.json
    python scripts/make_plots.py          # produces results/figures/*.png

Three figures, one per question the results section has to answer:

``accuracy_by_retriever.png``
    How accuracy varies with the retriever and with the cut-off k.
``latency.png``
    What each retriever costs per query, on a log scale because the sparse and
    dense figures differ by three orders of magnitude.
``accuracy_vs_budget.png``
    Accuracy against the amount of context retrieved, which is the only fair
    way to compare two chunking strategies whose chunks are different sizes.

Colour identifies the retriever and nothing else, and it is consistent across
every figure. Every mark carries a printed value as well, so the figures stay
readable in greyscale and for colour-vision deficiency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Categorical slots 1-3 of the validated default palette. Verified for the
# light surface with all three pairs in play: worst CVD Delta E 9.2, worst
# normal-vision Delta E 24.0. Aqua sits below 3:1 against the surface, so every
# figure also prints its values directly (the relief rule).
SERIES_COLOURS = {
    "bm25": "#2a78d6",     # blue
    "dense": "#eb6834",    # orange
    "hybrid": "#1baf7a",   # aqua
}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#1a1a19"
INK_SECONDARY = "#4a4943"
INK_MUTED = "#84837b"
GRID = "#e4e3dd"

RETRIEVER_ORDER = ("bm25", "dense", "hybrid")
CHUNKING_ORDER = ("fixed", "paragraph")
CHUNKING_LABEL = {
    "fixed": "Fixed-size chunks (300 words, 50 overlap)",
    "paragraph": "Paragraph chunks (section-aware)",
}


def apply_style(pyplot) -> None:
    """A recessive, print-friendly base style."""
    pyplot.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9,
            "font.family": "sans-serif",
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "figure.dpi": 160,
        }
    )


def add_title_block(figure, title: str, subtitle: str) -> None:
    """Title and subtitle as figure text with explicit positions.

    ``suptitle`` plus a second text call collide once the subtitle wraps, so
    both lines are placed by hand and the layout is given matching headroom.
    """
    figure.text(0.01, 0.985, title, ha="left", va="top", fontsize=11,
                fontweight="bold", color=INK_PRIMARY)
    figure.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=8,
                color=INK_MUTED)


def load_results(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"{path} not found; run 'python scripts/run_experiments.py' first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_lookup(payload: dict) -> dict[tuple[str, str], dict]:
    return {(r["chunking"], r["retriever"]): r for r in payload["runs"]}


def plot_accuracy(payload: dict, output: Path) -> Path:
    """Grouped bars: accuracy at each cut-off, one panel per chunking strategy."""
    import matplotlib.pyplot as pyplot

    apply_style(pyplot)
    runs = run_lookup(payload)
    k_values = [int(k) for k in payload["k_values"]]

    figure, axes = pyplot.subplots(1, 2, figsize=(9.0, 4.1), sharey=True)
    group_width = 0.78
    bar_width = group_width / len(RETRIEVER_ORDER)

    for axis, chunking in zip(axes, CHUNKING_ORDER):
        positions = range(len(k_values))
        for slot, retriever in enumerate(RETRIEVER_ORDER):
            run = runs.get((chunking, retriever))
            if run is None:
                continue
            values = [
                run["metrics"].get(f"answer_in_context@{k}", 0.0) for k in k_values
            ]
            # A 2px gap between adjacent bars: shrink each bar slightly.
            offsets = [
                p - group_width / 2 + slot * bar_width + bar_width / 2
                for p in positions
            ]
            axis.bar(
                offsets,
                values,
                width=bar_width * 0.88,
                color=SERIES_COLOURS[retriever],
                label=retriever if chunking == CHUNKING_ORDER[0] else None,
                zorder=3,
            )
            for x, value in zip(offsets, values):
                axis.text(
                    x,
                    value + 0.02,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=INK_SECONDARY,
                    zorder=4,
                )

        axis.set_xticks(list(positions))
        axis.set_xticklabels([f"k = {k}" for k in k_values])
        axis.set_ylim(0, 1.14)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(axis="y", zorder=0)
        axis.set_axisbelow(True)
        axis.set_title(CHUNKING_LABEL[chunking], color=INK_PRIMARY, loc="left")

    axes[0].set_ylabel("Answer-in-context accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04)
    )
    add_title_block(
        figure,
        "Retrieval accuracy by cut-off",
        f"Fraction of {payload['questions']} questions whose answer keywords all "
        "appear in the retrieved chunks",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.89))
    path = output / "accuracy_by_retriever.png"
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)
    return path


def plot_latency(payload: dict, output: Path) -> Path:
    """Dot plot on a log axis: median query latency for each run.

    A dot plot rather than bars. The values span three orders of magnitude, and
    bar length on a logarithmic axis is not proportional to the value it
    encodes: BM25 is roughly five hundred times faster than dense retrieval, but
    drawn as bars it would look about seven times faster. A dot encodes position
    only, which a log axis represents honestly.
    """
    import matplotlib.pyplot as pyplot

    apply_style(pyplot)
    runs = run_lookup(payload)

    rows = []
    for chunking in CHUNKING_ORDER:
        for retriever in RETRIEVER_ORDER:
            run = runs.get((chunking, retriever))
            if run is not None:
                rows.append((chunking, retriever, run["latency"]))
    rows.reverse()  # first row of the table ends up at the top of the axis

    figure, axis = pyplot.subplots(figsize=(8.4, 3.8))
    positions = list(range(len(rows)))

    for position, (chunking, retriever, latency) in zip(positions, rows):
        median = max(latency["median_ms"], 1e-4)
        # A faint leader line for scanning only; it carries no magnitude.
        axis.plot(
            [0.02, median],
            [position, position],
            color=GRID,
            linewidth=1.0,
            zorder=2,
        )
        axis.plot(
            median,
            position,
            marker="o",
            markersize=9,
            color=SERIES_COLOURS[retriever],
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        axis.text(
            median * 1.35,
            position,
            f"{median:.3f} ms" if median < 1 else f"{median:.1f} ms",
            va="center",
            ha="left",
            fontsize=8,
            color=INK_SECONDARY,
            zorder=4,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([f"{r}  ({c})" for c, r, _ in rows])
    axis.set_xscale("log")
    axis.set_xlabel("Median latency per query (ms, log scale)")
    axis.set_xlim(0.02, 400)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    add_title_block(
        figure,
        "Query latency: retrieval step only",
        f"Median of {payload['repeats']} timed repeats per question, PyTorch pinned "
        "to one thread. Sparse and dense retrieval differ by about five hundred "
        "times, hence the log axis.",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    path = output / "latency.png"
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)
    return path


def plot_accuracy_vs_budget(payload: dict, output: Path) -> Path | None:
    """Accuracy against retrieved context volume, the budget-matched comparison."""
    import matplotlib.pyplot as pyplot

    sweep = payload.get("k_sweep") or []
    if not sweep:
        return None

    apply_style(pyplot)
    figure, axes = pyplot.subplots(1, 2, figsize=(9.0, 4.1), sharey=True, sharex=True)

    # Vertical offsets so that two series starting at the same value do not
    # print their labels on top of each other.
    label_offsets = {"bm25": -9, "dense": 1, "hybrid": 9}

    for axis, chunking in zip(axes, CHUNKING_ORDER):
        for retriever in RETRIEVER_ORDER:
            points = [
                row
                for row in sweep
                if row["chunking"] == chunking and row["retriever"] == retriever
            ]
            points.sort(key=lambda row: row["context_words"])
            if not points:
                continue
            x = [row["context_words"] for row in points]
            y = [row["answer_in_context"] for row in points]
            axis.plot(
                x,
                y,
                marker="o",
                markersize=4.5,
                color=SERIES_COLOURS[retriever],
                label=retriever if chunking == CHUNKING_ORDER[0] else None,
                zorder=3,
            )
            # Label at the left end, where the three lines are still apart; at
            # the right end they all reach 1.0 and the labels would collide.
            axis.annotate(
                retriever,
                (x[0], y[0]),
                textcoords="offset points",
                xytext=(-7, label_offsets[retriever]),
                fontsize=7,
                color=INK_SECONDARY,
                ha="right",
                va="center",
            )
        axis.grid(zorder=0)
        axis.set_axisbelow(True)
        axis.set_xscale("log")
        axis.set_xlim(70, 4200)
        axis.set_xticks([100, 200, 500, 1000, 2000])
        axis.set_xticklabels(["100", "200", "500", "1000", "2000"])
        axis.set_xlabel("Words of context retrieved (log scale)")
        axis.set_title(CHUNKING_LABEL[chunking], loc="left", color=INK_PRIMARY)

    axes[0].set_ylabel("Answer-in-context accuracy")
    axes[0].set_ylim(0.55, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04)
    )
    add_title_block(
        figure,
        "Accuracy against context budget",
        "Each point is one value of k (1, 2, 3, 5, 8, 10). Read the panels at the "
        "same x position to compare at equal context, not at equal chunk count.",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.89))
    path = output / "accuracy_vs_budget.png"
    figure.savefig(path, bbox_inches="tight")
    pyplot.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    from adaptive_rag.config import RESULTS_DIR

    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "results.json")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "figures")
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise SystemExit(
            "matplotlib is required for the figures; install it with "
            "'pip install matplotlib' (or 'pip install -r requirements-dev.txt')"
        )
    import matplotlib

    matplotlib.use("Agg")

    payload = load_results(args.results)
    args.output.mkdir(parents=True, exist_ok=True)

    written = [
        plot_accuracy(payload, args.output),
        plot_latency(payload, args.output),
        plot_accuracy_vs_budget(payload, args.output),
    ]
    for path in written:
        if path is not None:
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
