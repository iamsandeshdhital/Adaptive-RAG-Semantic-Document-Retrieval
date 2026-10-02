# Adaptive-RAG — Semantic Document Retrieval

A retrieval-augmented generation pipeline built to answer one question with
measurements rather than assertions: **on a small document collection, which
retrieval strategy and which chunking strategy actually work best, and what does
each cost per query?**

The pipeline implements three retrievers (BM25, dense, hybrid) and two chunking
strategies (fixed-size, paragraph-based), and evaluates all six combinations on
the same annotated question set.

**Authors:** Anurag Jha and Sandesh Dhital

---

## The pipeline

```
                     PDF / Markdown / Text documents
                                  |
                          Text extraction
                       (pypdf, front matter, de-hyphenation)
                                  |
                              Chunking
                                  |
                  +---------------+---------------+
                  |                               |
            Fixed-size chunks              Paragraph chunks
          (300 words, 50 overlap)        (section-aware, merged
                  |                       and split to a range)
                  +---------------+---------------+
                                  |
                              Retrieval
                                  |
              +-------------------+-------------------+
              |                   |                   |
            BM25                Dense               Hybrid
        (Okapi, from        (MiniLM embeddings,   (normalised score
          scratch)          cosine, FAISS)        fusion, or RRF)
              |                   |                   |
              +-------------------+-------------------+
                                  |
                          Relevant chunks
                                  |
                          Answer generation
                  (extractive by default; Claude optional)
                                  |
                            Answer + citations
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere

pip install -r requirements-dense.txt   # pretrained embeddings + FAISS
pip install -e .
```

`requirements.txt` alone is enough to run everything: dense retrieval then falls
back to a NumPy/scikit-learn embedding backend (TF-IDF plus truncated SVD)
instead of downloading PyTorch. The reported results use the pretrained model,
so install `requirements-dense.txt` to reproduce them.

```bash
# What is in the corpus, and what each chunker does to it
python -m adaptive_rag stats

# Retrieve without generating
python -m adaptive_rag query "why do corals bleach?" --retriever hybrid

# The full pipeline, with citations
python -m adaptive_rag ask "what sets the rhythm of the heart?"

# The same question through all three retrievers, side by side
python -m adaptive_rag compare "how is magma generated where plates pull apart?"

# Reproduce every number in the report
python scripts/run_experiments.py
python scripts/make_plots.py
```

## Results

Sixty annotated questions over ten documents. **Accuracy** is
answer-in-context: the fraction of questions for which every annotated answer
keyword appears somewhere in the retrieved chunks, which is the thing a
generator actually depends on. Latency is the median over seven timed repeats
per question, retrieval step only.

| Retrieval | Chunking | Accuracy@1 | Accuracy@3 | Hit@1 | MRR | Median latency | Index build |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | fixed | 0.800 | 0.983 | 0.950 | 0.971 | **0.066 ms** | **0.014 s** |
| BM25 | paragraph | 0.750 | 0.900 | 0.967 | 0.978 | 0.085 ms | 0.009 s |
| Dense | fixed | 0.750 | 0.967 | 0.967 | 0.974 | 38.6 ms | 14.9 s |
| Dense | paragraph | 0.750 | 0.917 | 0.983 | 0.989 | 41.6 ms | 21.6 s |
| Hybrid | fixed | **0.833** | **1.000** | **1.000** | **1.000** | 43.1 ms | 16.9 s |
| Hybrid | paragraph | **0.833** | 0.933 | **1.000** | **1.000** | 36.0 ms | 16.1 s |

Four findings:

- **Hybrid retrieval was never worse than either component**, and was the only
  method to rank a correct chunk first for all sixty questions (Hit@1 = MRR =
  1.000 under both chunkers). The reason is visible per question rather than in
  the averages: the questions that defeat BM25 and those that defeat dense
  retrieval form *disjoint* sets, so fusion recovers all of them. The alpha
  sweep confirms it — MRR reaches 1.000 across `alpha` 0.4–0.8 while both
  endpoints (pure BM25, pure dense) fall short.
- **BM25 is ~500× faster per query and ~1,000× faster to index**, and came
  within one question in sixty of the best configuration. Hybrid retrieval costs
  what dense retrieval costs, so the real question is whether to pay for the
  embedding model at all.
- **The chunking comparison reverses depending on how it is measured.** At equal
  *k*, fixed-size chunking wins — but it also returns 2.3× more text. At equal
  *context budget*, paragraph chunking reaches accuracy 1.000 on 573 words where
  fixed chunking needs 1,340, and nearly halves the irrelevant text in the
  context (precision@5 0.827 vs 0.570).
- **Paragraph chunking pays for that on multi-hop questions**, losing 3 of 10 at
  k = 3 regardless of retriever: two facts in different sections do not fit in
  one 112-word chunk.

With sixty questions none of the pairwise differences is statistically
significant on its own (paired sign test, p ≥ 0.125); the evidence is the
consistent direction plus the identified mechanism, not the magnitude. The
report is explicit about this.

Full tables, per-question outcomes and the environment the numbers were measured
on are in [`results/`](results/); the written analysis is in
[`docs/REPORT.md`](docs/REPORT.md).

## What the code does

| Module | Responsibility |
| --- | --- |
| `loaders.py` | Read PDF, Markdown and text; front matter; normalise whitespace and repair hyphenated line breaks |
| `preprocess.py` | Tokenise, drop stopwords, stem (a short suffix-rule stemmer, no dependency) |
| `chunking.py` | `FixedSizeChunker` (sliding window) and `ParagraphChunker` (section-aware) |
| `retrieval/bm25.py` | Okapi BM25 over an inverted index, implemented directly |
| `retrieval/embeddings.py` | Pretrained sentence embeddings, with an LSA fallback behind the same interface |
| `retrieval/dense.py` | Cosine similarity search, FAISS `IndexFlatIP` or a NumPy matrix product |
| `retrieval/hybrid.py` | Weighted score fusion and reciprocal rank fusion |
| `generation.py` | Extractive reader (default, offline) and an optional Claude backend |
| `pipeline.py` | Wires the stages together; `SharedCorpus` caches indexes across the experiment grid |
| `evaluation/metrics.py` | Hit@k, precision@k, recall@k, MRR, nDCG@k, answer-in-context, latency summaries |
| `evaluation/experiment.py` | The grid, the alpha and k sweeps, and result serialisation |

## Repository layout

```
adaptive-rag/
├── data/
│   ├── documents/            ten source documents (9 Markdown, 1 plain text)
│   └── eval/questions.json   sixty annotated questions
├── src/adaptive_rag/         the package
├── scripts/
│   ├── run_experiments.py    reproduce every reported number
│   ├── make_plots.py         render the figures
│   └── make_pdf_sample.py    build a PDF and verify its text layer
├── results/                  generated tables, per-question CSV, figures
├── docs/REPORT.md            the written report
└── tests/                    the test suite
```

## Optional: answering with Claude

The default generator is extractive, so the pipeline runs with no API key and no
network. To generate answers with the Claude API instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # or: setx ANTHROPIC_API_KEY ... on Windows
python -m adaptive_rag ask "why do corals bleach?" --generator claude
```

The retrieved chunks are supplied as numbered passages and the model is
instructed to cite them and to say so when they do not contain the answer.
Retrieval is unchanged, so the reported retrieval metrics do not depend on which
generator is used.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the chunk boundaries and overlap arithmetic, the BM25 scoring
formula against a hand-computed value, agreement between the FAISS and NumPy
search paths, the fusion rules at their limits (`alpha=0` reproduces BM25,
`alpha=1` reproduces dense retrieval), every metric, the PDF text layer, and one
end-to-end run of each retriever and chunker. It also asserts that the
evaluation set is achievable: every answer keyword really does occur in the
document it is attributed to, so a perfect retriever would score 1.0.

## Licence

MIT — see [LICENSE](LICENSE).
