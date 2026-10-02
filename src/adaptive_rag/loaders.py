"""Document loading and text extraction.

Supported inputs are plain text, Markdown and PDF. Text and Markdown files may
carry a small YAML-style front matter block delimited by ``---`` lines, which is
used for document metadata; PDFs take their metadata from the file name and
embedded document info.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SIMPLE_KEY_VALUE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")


@dataclass(frozen=True)
class Document:
    """One source document after text extraction."""

    doc_id: str
    title: str
    text: str
    path: Path
    topic: str = "unknown"
    source: str = "unknown"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class DocumentLoadError(RuntimeError):
    """Raised when a file exists but no text could be extracted from it."""


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split ``key: value`` front matter from the body of a text file."""
    match = _FRONT_MATTER.match(raw)
    if match is None:
        return {}, raw

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kv = _SIMPLE_KEY_VALUE.match(line)
        if kv is not None:
            metadata[kv.group(1).strip().lower()] = kv.group(2).strip().strip("\"'")
    return metadata, raw[match.end():]


def normalise_text(text: str) -> str:
    """Tidy extracted text without destroying paragraph structure.

    Paragraph breaks matter because the paragraph chunker relies on them, so
    blank lines are preserved while single line breaks inside a paragraph
    (introduced by hard wrapping or by PDF extraction) are joined into spaces.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # De-hyphenate words broken across a line, a common PDF artefact.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse runs of blank lines to exactly one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = []
    for block in text.split("\n\n"):
        lines = [line.strip() for line in block.split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            continue
        # Keep a Markdown heading on its own line so section detection works.
        rebuilt: list[str] = []
        buffer: list[str] = []
        for line in lines:
            if line.startswith("#"):
                if buffer:
                    rebuilt.append(" ".join(buffer))
                    buffer = []
                rebuilt.append(line)
            else:
                buffer.append(line)
        if buffer:
            rebuilt.append(" ".join(buffer))
        paragraphs.append("\n".join(rebuilt))

    return "\n\n".join(paragraphs).strip()


def _title_from_text(text: str, fallback: str) -> str:
    """Use the first Markdown heading or the first short line as a title."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if len(line.split()) <= 15:
            return line
        break
    return fallback


def _doc_id_from_path(path: Path) -> str:
    """Derive a stable identifier such as ``coral_reefs`` from a file name."""
    stem = path.stem
    stem = re.sub(r"^\d+[-_\s]*", "", stem)  # drop a leading ordering prefix
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return stem or path.stem.lower()


def _load_text_file(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _parse_front_matter(raw)
    return metadata, normalise_text(body)


def _load_pdf_file(path: Path) -> tuple[dict[str, str], str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DocumentLoadError(
            f"reading {path.name} requires pypdf; install it with 'pip install pypdf'"
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    metadata: dict[str, str] = {}
    info = getattr(reader, "metadata", None)
    if info:
        if info.title:
            metadata["title"] = str(info.title)
        if info.subject:
            metadata["topic"] = str(info.subject)
    return metadata, normalise_text("\n\n".join(pages))


def load_document(path: Path | str) -> Document:
    """Load and normalise a single document."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        metadata, text = _load_text_file(path)
    elif suffix in PDF_SUFFIXES:
        metadata, text = _load_pdf_file(path)
    else:
        raise DocumentLoadError(
            f"unsupported file type {suffix!r}; expected one of "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )

    if not text.strip():
        raise DocumentLoadError(f"no text could be extracted from {path.name}")

    doc_id = metadata.get("doc_id") or _doc_id_from_path(path)
    title = metadata.get("title") or _title_from_text(text, doc_id.replace("_", " "))
    return Document(
        doc_id=doc_id,
        title=title,
        text=text,
        path=path,
        topic=metadata.get("topic", "unknown"),
        source=metadata.get("source", "unknown"),
        metadata=metadata,
    )


def load_corpus(
    directory: Path | str,
    suffixes: Iterable[str] | None = None,
) -> list[Document]:
    """Load every supported document in ``directory``, sorted by file name.

    Sorting keeps chunk identifiers stable between runs, which matters because
    the experiment reports per-chunk results.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")

    allowed = {s.lower() for s in (suffixes or SUPPORTED_SUFFIXES)}
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in allowed)
    if not paths:
        raise FileNotFoundError(f"no supported documents found in {directory}")

    documents: list[Document] = []
    seen: dict[str, Path] = {}
    for path in paths:
        document = load_document(path)
        if document.doc_id in seen:
            raise DocumentLoadError(
                f"duplicate doc_id {document.doc_id!r} in {path.name} and "
                f"{seen[document.doc_id].name}"
            )
        seen[document.doc_id] = path
        documents.append(document)
    return documents


def corpus_statistics(documents: Sequence[Document]) -> dict[str, float | int]:
    """Summary counts used by the CLI and by the dataset section of the report."""
    words = [doc.word_count for doc in documents]
    total = sum(words)
    return {
        "documents": len(documents),
        "total_words": total,
        "mean_words": round(total / len(documents), 1) if documents else 0.0,
        "min_words": min(words) if words else 0,
        "max_words": max(words) if words else 0,
    }
