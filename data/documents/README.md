# Corpus

Ten short documents written for this project, covering four broad areas:

| Document | Topic area |
| --- | --- |
| `01_photosynthesis.md` | biology |
| `02_water_cycle.md` | earth science |
| `03_volcanoes.md` | earth science |
| `04_circulatory_system.md` | biology |
| `05_antibiotics.md` | medicine |
| `06_apollo_programme.md` | history of technology |
| `07_renewable_energy.md` | engineering |
| `08_vaccination.md` | medicine |
| `09_coral_reefs.md` | marine science |
| `10_earthquakes.txt` | earth science |

The topics deliberately overlap in places -- volcanoes and earthquakes both
discuss plate tectonics, coral reefs and the water cycle both discuss the
ocean, photosynthesis and renewable energy both discuss capturing light -- so
that a retriever has to do more than match a single distinctive keyword.

Nine files are Markdown with `## ` section headings. `10_earthquakes.txt` is
plain text with no headings at all, so that the paragraph chunker is exercised
on a document whose only structure is its blank lines.

Each file begins with a small front matter block:

```
---
title: The Human Circulatory System
doc_id: circulatory_system
topic: biology
source: Adaptive-RAG teaching corpus (original text)
---
```

`doc_id` is the label the evaluation set refers to, so it must stay stable.

PDF files placed in this directory are also indexed; text is extracted with
`pypdf`. To try that path, run `python scripts/make_pdf_sample.py`, which
renders one of the Markdown documents to PDF.
