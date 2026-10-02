#!/usr/bin/env python
"""Render one corpus document to PDF so the PDF ingestion path can be tried.

    python scripts/make_pdf_sample.py --doc 04_circulatory_system.md

The pipeline reads PDFs through ``pypdf``, but the corpus ships as text so that
it stays diffable and reviewable. This script produces a real PDF from one of
those documents and then checks that the text can be extracted again, which is
the part that actually matters: a PDF whose text layer cannot be read is
useless to a retrieval system, and it is a common failure with scanned files.

The output is written next to the corpus by default, so the next
``python -m adaptive_rag stats`` will pick it up as an eleventh document.
Delete it to go back to the ten-document corpus.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LINES_PER_PAGE = 46
WRAP_WIDTH = 88


def render_pdf(text: str, title: str, destination: Path) -> Path:
    """Lay the text out as pages of a PDF with an embedded TrueType font."""
    import matplotlib

    matplotlib.use("Agg")
    # Type 42 embeds a TrueType subset with a ToUnicode map, which is what
    # makes the text extractable afterwards. The default Type 3 output is not
    # reliably extractable.
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as pyplot
    from matplotlib.backends.backend_pdf import PdfPages

    lines: list[str] = []
    for paragraph in text.split("\n\n"):
        flattened = " ".join(paragraph.split())
        if not flattened:
            continue
        lines.extend(textwrap.wrap(flattened, WRAP_WIDTH) or [""])
        lines.append("")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(destination) as pdf:
        for start in range(0, len(lines), LINES_PER_PAGE):
            page = lines[start : start + LINES_PER_PAGE]
            figure = pyplot.figure(figsize=(8.27, 11.69))  # A4
            figure.text(
                0.08,
                0.94,
                "\n".join(page),
                va="top",
                ha="left",
                fontsize=9,
                family="DejaVu Sans",
                linespacing=1.6,
            )
            pdf.savefig(figure)
            pyplot.close(figure)
        info = pdf.infodict()
        info["Title"] = title
        info["Subject"] = "Adaptive-RAG teaching corpus"
    return destination


def main() -> int:
    from adaptive_rag.config import DOCUMENTS_DIR
    from adaptive_rag.loaders import load_document

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default="04_circulatory_system.md",
                        help="corpus file to convert (default: %(default)s)")
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="output path (default: alongside the corpus)")
    args = parser.parse_args()

    source = args.documents / args.doc
    if not source.is_file():
        raise SystemExit(f"{source} not found")

    document = load_document(source)
    destination = args.output or args.documents / f"{source.stem}.pdf"

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise SystemExit(
            "matplotlib is required to build the sample PDF; install it with "
            "'pip install matplotlib'"
        )

    render_pdf(document.text, document.title, destination)
    print(f"wrote {destination}")

    # Read it back through the loader that the pipeline itself uses.
    extracted = load_document(destination)
    original_words = set(document.text.lower().split())
    extracted_words = set(extracted.text.lower().split())
    overlap = len(original_words & extracted_words) / max(1, len(original_words))

    print(f"extracted {extracted.word_count} words (source has {document.word_count})")
    print(f"vocabulary recovered: {overlap:.1%}")
    if overlap < 0.9:
        print(
            "warning: less than 90% of the source vocabulary was recovered, so "
            "the text layer of this PDF is not reliable",
            file=sys.stderr,
        )
        return 1
    print("PDF text extraction works; the file is now part of the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
