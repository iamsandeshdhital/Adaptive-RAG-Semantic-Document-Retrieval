# Experiment results

Questions: 60. Latency repeats per query: 7.
Corpus: 10 documents, 7061 words.
Measured on Windows-11-10.0.26200-SP0 with Python 3.13.3 at 2026-09-11 01:25:35.

## Retrieval comparison

Latency: 7 timed repeats per question after 5 warm-up queries, measured on the retrieval step only (no generation).
PyTorch was pinned to 1 thread so that repeated calls are comparable.
The median is the figure to compare; the raw mean is shown next to it because it is sensitive to operating-system stalls on a shared machine.
Measurement warning: the following runs contain samples far above their own median and their raw means are not meaningful: paragraph+bm25 (max 17 ms, 3 stall(s)).

| retrieval | chunking  | accuracy@1 | accuracy@3 | accuracy@5 | hit@1 | mrr   | median latency (ms) | trimmed mean (ms) | p95 latency (ms) | raw mean (ms) | stalls |
| --------- | --------- | ---------- | ---------- | ---------- | ----- | ----- | ------------------- | ----------------- | ---------------- | ------------- | ------ |
| bm25      | fixed     | 0.800      | 0.983      | 1.000      | 0.950 | 0.971 | 0.066               | 0.065             | 0.119            | 0.069         | 0      |
| bm25      | paragraph | 0.750      | 0.900      | 0.933      | 0.967 | 0.978 | 0.085               | 0.087             | 0.154            | 0.143         | 3      |
| dense     | fixed     | 0.750      | 0.967      | 1.000      | 0.967 | 0.974 | 38.578              | 46.231            | 120.678          | 51.164        | 0      |
| dense     | paragraph | 0.750      | 0.917      | 1.000      | 0.983 | 0.989 | 41.627              | 50.792            | 129.205          | 56.948        | 0      |
| hybrid    | fixed     | 0.833      | 1.000      | 1.000      | 1.000 | 1.000 | 43.104              | 53.557            | 130.770          | 59.542        | 0      |
| hybrid    | paragraph | 0.833      | 0.933      | 1.000      | 1.000 | 1.000 | 35.999              | 36.309            | 51.645           | 37.896        | 0      |

## All runs

| chunking  | retriever | chunks | mean_chunk_words | hit@1  | hit@3  | hit@5 | recall@5 | precision@5 | mrr    | ndcg@5 | answer_in_context@5 | context_words@5 | median_latency_ms | trimmed_mean_latency_ms | p95_latency_ms | raw_mean_latency_ms | latency_stalls | index_seconds |
| --------- | --------- | ------ | ---------------- | ------ | ------ | ----- | -------- | ----------- | ------ | ------ | ------------------- | --------------- | ----------------- | ----------------------- | -------------- | ------------------- | -------------- | ------------- |
| fixed     | bm25      | 30     | 268.4            | 0.95   | 0.9833 | 1.0   | 1.0      | 0.5033      | 0.9708 | 0.9611 | 1.0                 | 1327.5167       | 0.066             | 0.065                   | 0.119          | 0.069               | 0              | 0.014         |
| fixed     | dense     | 30     | 268.4            | 0.9667 | 0.9667 | 1.0   | 1.0      | 0.57        | 0.9742 | 0.9665 | 1.0                 | 1340.0833       | 38.578            | 46.231                  | 120.678        | 51.164              | 0              | 14.949        |
| fixed     | hybrid    | 30     | 268.4            | 1.0    | 1.0    | 1.0   | 1.0      | 0.5533      | 1.0    | 0.9885 | 1.0                 | 1350.5833       | 43.104            | 53.557                  | 130.77         | 59.542              | 0              | 16.892        |
| paragraph | bm25      | 64     | 111.5            | 0.9667 | 0.9833 | 1.0   | 1.0      | 0.6167      | 0.9783 | 0.9597 | 0.9333              | 570.4167        | 0.085             | 0.087                   | 0.154          | 0.143               | 3              | 0.009         |
| paragraph | dense     | 64     | 111.5            | 0.9833 | 1.0    | 1.0   | 1.0      | 0.8267      | 0.9889 | 0.9788 | 1.0                 | 572.5833        | 41.627            | 50.792                  | 129.205        | 56.948              | 0              | 21.569        |
| paragraph | hybrid    | 64     | 111.5            | 1.0    | 1.0    | 1.0   | 1.0      | 0.8         | 1.0    | 0.9876 | 1.0                 | 568.6           | 35.999            | 36.309                  | 51.645         | 37.896              | 0              | 16.065        |

## Accuracy by question type

Metric: answer_in_context@3

| chunking  | retriever | confusable | lexical | multi_hop | paraphrased | all   |
| --------- | --------- | ---------- | ------- | --------- | ----------- | ----- |
| fixed     | bm25      | 1.000      | 1.000   | 1.000     | 0.950       | 0.983 |
| fixed     | dense     | 0.900      | 1.000   | 1.000     | 0.950       | 0.967 |
| fixed     | hybrid    | 1.000      | 1.000   | 1.000     | 1.000       | 1.000 |
| paragraph | bm25      | 1.000      | 1.000   | 0.700     | 0.850       | 0.900 |
| paragraph | dense     | 1.000      | 1.000   | 0.700     | 0.900       | 0.917 |
| paragraph | hybrid    | 1.000      | 1.000   | 0.700     | 0.950       | 0.933 |

Metric: hit@1

| chunking  | retriever | confusable | lexical | multi_hop | paraphrased | all   |
| --------- | --------- | ---------- | ------- | --------- | ----------- | ----- |
| fixed     | bm25      | 0.900      | 1.000   | 1.000     | 0.900       | 0.950 |
| fixed     | dense     | 0.900      | 1.000   | 1.000     | 0.950       | 0.967 |
| fixed     | hybrid    | 1.000      | 1.000   | 1.000     | 1.000       | 1.000 |
| paragraph | bm25      | 0.900      | 1.000   | 1.000     | 0.950       | 0.967 |
| paragraph | dense     | 1.000      | 1.000   | 1.000     | 0.950       | 0.983 |
| paragraph | hybrid    | 1.000      | 1.000   | 1.000     | 1.000       | 1.000 |

## Hybrid alpha sweep

alpha=0 is pure BM25, alpha=1 is pure dense retrieval (top 5).

| alpha | answer_in_context@5 | hit@5 | mrr    | ndcg@5 |
| ----- | ------------------- | ----- | ------ | ------ |
| 0.0   | 0.9333              | 1.0   | 0.9783 | 0.9597 |
| 0.2   | 0.9833              | 1.0   | 0.9875 | 0.9748 |
| 0.4   | 1.0                 | 1.0   | 1.0    | 0.9819 |
| 0.5   | 1.0                 | 1.0   | 1.0    | 0.9876 |
| 0.6   | 1.0                 | 1.0   | 1.0    | 0.9857 |
| 0.8   | 1.0                 | 1.0   | 1.0    | 0.9887 |
| 1.0   | 1.0                 | 1.0   | 0.9889 | 0.9788 |

## Accuracy against k and context budget

The two chunking strategies produce chunks of different average length,
so the same k is not the same amount of context. Compare rows with a
similar context_words figure rather than a similar k.

| chunking  | retriever | k  | answer_in_context | hit    | recall | context_words |
| --------- | --------- | -- | ----------------- | ------ | ------ | ------------- |
| fixed     | bm25      | 1  | 0.8               | 0.95   | 0.9417 | 279.4         |
| fixed     | bm25      | 2  | 0.95              | 0.9833 | 0.9833 | 545.4         |
| fixed     | bm25      | 3  | 0.9833            | 0.9833 | 0.9833 | 817.5         |
| fixed     | bm25      | 5  | 1.0               | 1.0    | 1.0    | 1327.5        |
| fixed     | bm25      | 8  | 1.0               | 1.0    | 1.0    | 2024.0        |
| fixed     | bm25      | 10 | 1.0               | 1.0    | 1.0    | 2466.6        |
| fixed     | dense     | 1  | 0.75              | 0.9667 | 0.9583 | 280.8         |
| fixed     | dense     | 2  | 0.8667            | 0.9667 | 0.9583 | 553.6         |
| fixed     | dense     | 3  | 0.9667            | 0.9667 | 0.9583 | 804.0         |
| fixed     | dense     | 5  | 1.0               | 1.0    | 1.0    | 1340.1        |
| fixed     | dense     | 8  | 1.0               | 1.0    | 1.0    | 2124.3        |
| fixed     | dense     | 10 | 1.0               | 1.0    | 1.0    | 2666.6        |
| fixed     | hybrid    | 1  | 0.8333            | 1.0    | 0.9917 | 284.5         |
| fixed     | hybrid    | 2  | 0.9667            | 1.0    | 0.9917 | 538.9         |
| fixed     | hybrid    | 3  | 1.0               | 1.0    | 1.0    | 806.3         |
| fixed     | hybrid    | 5  | 1.0               | 1.0    | 1.0    | 1350.6        |
| fixed     | hybrid    | 8  | 1.0               | 1.0    | 1.0    | 2152.1        |
| fixed     | hybrid    | 10 | 1.0               | 1.0    | 1.0    | 2683.2        |
| paragraph | bm25      | 1  | 0.75              | 0.9667 | 0.9583 | 120.8         |
| paragraph | bm25      | 2  | 0.85              | 0.9833 | 0.9833 | 232.4         |
| paragraph | bm25      | 3  | 0.9               | 0.9833 | 0.9833 | 349.5         |
| paragraph | bm25      | 5  | 0.9333            | 1.0    | 1.0    | 570.4         |
| paragraph | bm25      | 8  | 0.9833            | 1.0    | 1.0    | 872.9         |
| paragraph | bm25      | 10 | 1.0               | 1.0    | 1.0    | 1080.9        |
| paragraph | dense     | 1  | 0.75              | 0.9833 | 0.975  | 121.4         |
| paragraph | dense     | 2  | 0.8833            | 0.9833 | 0.9833 | 231.6         |
| paragraph | dense     | 3  | 0.9167            | 1.0    | 1.0    | 345.1         |
| paragraph | dense     | 5  | 1.0               | 1.0    | 1.0    | 572.6         |
| paragraph | dense     | 8  | 1.0               | 1.0    | 1.0    | 908.5         |
| paragraph | dense     | 10 | 1.0               | 1.0    | 1.0    | 1133.5        |
| paragraph | hybrid    | 1  | 0.8333            | 1.0    | 0.9917 | 117.7         |
| paragraph | hybrid    | 2  | 0.9167            | 1.0    | 1.0    | 239.0         |
| paragraph | hybrid    | 3  | 0.9333            | 1.0    | 1.0    | 347.9         |
| paragraph | hybrid    | 5  | 1.0               | 1.0    | 1.0    | 568.6         |
| paragraph | hybrid    | 8  | 1.0               | 1.0    | 1.0    | 905.9         |
| paragraph | hybrid    | 10 | 1.0               | 1.0    | 1.0    | 1131.4        |
