# Supplement Evidence Lens

Supplement Evidence Lens is an evidence-grounded RAG assistant for questions about dietary supplements.

It retrieves evidence from official EU, Canadian, and US sources, reranks the retrieved passages, and uses an LLM to produce a cited answer. The current version is available through the command line; a web UI is the next development step.

Example questions:

- Does melatonin actually help with sleep?
- What are the risks of taking too much zinc?
- Is the claim that magnesium glycinate eliminates anxiety supported?
- Can a supplement make a particular health claim in the EU?

## Problem

Information about supplement benefits, permitted claims, dosage conditions, and safety is distributed across different government sources. Search results can also mix regulatory claims, scientific summaries, and marketing language.

This project brings several official sources into one searchable evidence base and makes the distinction visible through citations and source metadata.

It is an informational tool, not medical advice.

## Current Status

Implemented:

- ingestion from three official data sources
- normalized document construction and chunking
- Elasticsearch indexing
- BM25, vector, hybrid, and reranked retrieval
- an Agentic RAG command-line workflow
- retrieval and answer-quality evaluation

Still to be built:

- web UI
- user feedback and monitoring
- full application containerization
- final deployment documentation and screenshots

## Data Sources

| Source | Jurisdiction | Content | Processed records |
|---|---|---|---:|
| EU Register on Nutrition and Health Claims | EU | authorised and non-authorised health claims, conditions, and regulatory status | 2,337 claims |
| Health Canada NHPID monographs | Canada | single-ingredient and product monograph sections | 3,009 sections from 243 monographs |
| NIH Office of Dietary Supplements | United States | health-professional fact-sheet sections | 379 sections from 42 fact sheets |

After document construction and chunking, these sources produce 8,021 searchable chunks:

| Source | Chunks |
|---|---:|
| EU Register | 2,340 |
| Health Canada NHPID | 4,354 |
| NIH ODS | 1,327 |

The index keeps source, jurisdiction, title, section, URL, and document identifiers as metadata. All sources are stored in one Elasticsearch index so they can be searched together and filtered by metadata when needed.

## Architecture

```text
User question
    |
    v
Agent decides what to search
    |
    v
BM25 search + vector search
    |
    v
Reciprocal Rank Fusion
    |
    v
Cross-encoder reranking
    |
    v
Top evidence passages
    |
    v
Grounded answer with citations
```

The Agentic workflow can rewrite or split a question into additional searches when the first evidence is insufficient. Search is capped at four queries to keep the workflow predictable.

## Models and Search

- Search engine: Elasticsearch 9.4.4
- Embedding model: `BAAI/bge-small-en-v1.5`
- Embedding size: 384 dimensions
- Keyword retrieval: Elasticsearch BM25
- Hybrid fusion: Reciprocal Rank Fusion
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Answer and search-planning model: `gpt-5.4-mini`

The application retrieves a larger hybrid candidate set, reranks it, and supplies the top five passages to the answer model.

## Project Structure

```text
app/
  rag.py                         Agentic RAG workflow and CLI
  retrieval.py                   BM25, vector, hybrid, and reranked search

ingestion/
  eu_health_claims.py            EU Register ingestion
  ca_nhpid_monographs.py         Health Canada monograph ingestion
  us_nih_ods.py                  NIH ODS fact-sheet ingestion
  chunk_documents.py             Normalization and chunking
  index_documents.py             Elasticsearch index creation

evaluation/
  generate_questions.py          Evaluation-question generation
  build_relevance_pool.py        Candidate-pool construction
  judge_relevance.py             LLM relevance judgments
  evaluate_retrieval.py          Retrieval metrics
  evaluate_answers.py            RAG answer generation and judging

data/
  raw/                           Downloaded source files
  processed/                     Normalized records and chunks
  evaluation/                    Evaluation datasets and results

docs/
  future_work.md                 Candidate data and evaluation improvements
```

## Setup

The project uses Python 3.13 and `uv`.

Install dependencies:

```bash
uv sync
```

Start Elasticsearch:

```bash
docker compose up -d
```

Check that it is running:

```bash
curl http://localhost:9200
```

Create a `.env` file for RAG and LLM-based evaluation:

```text
OPENAI_API_KEY=your-api-key
```

The `.env` file is ignored by Git.

## Build the Evidence Index

Run the ingestion scripts:

```bash
uv run python ingestion/eu_health_claims.py
uv run python ingestion/ca_nhpid_monographs.py
uv run python ingestion/us_nih_ods.py
```

Build the common document collection and index it:

```bash
uv run python ingestion/chunk_documents.py
uv run python ingestion/index_documents.py
```

Confirm the indexed document count:

```bash
curl http://localhost:9200/supplement_evidence/_count
```

The expected count for the current dataset is 8,021.

## Run the Assistant

Ask a question from the command line:

```bash
uv run python app/rag.py "What are the risks of taking too much zinc?"
```

The response includes numbered citations and the URLs of the retrieved official sources.

## Retrieval Evaluation

The retrieval evaluation uses 75 source-grounded questions: 25 seeded from each source. Candidate pools are formed from the union of BM25, vector, and hybrid results. An LLM assigns binary relevance judgments with explanations, and the seed document is retained as a known relevant document.

Four retrieval approaches are compared at rank 5:

| Approach | Hit Rate@5 | MRR@5 | Pooled Recall@5 |
|---|---:|---:|---:|
| BM25 | 0.907 | 0.808 | 0.593 |
| Vector | 0.867 | 0.789 | 0.609 |
| Hybrid | 0.907 | 0.853 | 0.643 |
| Hybrid + reranker | **0.947** | **0.928** | **0.711** |

Hybrid retrieval with reranking is therefore used by the application. It produced the strongest overall result, with especially large gains on the Health Canada subset. On the EU subset, plain hybrid retrieval remained slightly stronger on some recall measures, so the overall result should not be interpreted as a universal improvement for every source.

The detailed results are stored in [`data/evaluation/retrieval_metrics.json`](data/evaluation/retrieval_metrics.json).

Run the retrieval evaluation pipeline:

```bash
uv run python -m evaluation.generate_questions
uv run python -m evaluation.build_relevance_pool
uv run python -m evaluation.judge_relevance
uv run python -m evaluation.evaluate_retrieval
```

## Answer Evaluation

The same 75 questions are used to compare two RAG workflows:

- Fixed RAG: searches the original question once.
- Agentic RAG: allows the model to rewrite, decompose, or repeat searches when useful.

Both approaches use the same index, retrieval pipeline, reranker, answer model, and evaluation rubric. Because there are no manually written reference answers, an LLM judge scores:

- answer relevance
- faithfulness to retrieved evidence
- citation correctness

Each metric is scored from 1 to 5 with a written reason.

| Workflow | Relevance | Faithfulness | Citation correctness | Perfect answers |
|---|---:|---:|---:|---:|
| Fixed RAG | 4.920 | **4.720** | 4.667 | **49** |
| Agentic RAG | **4.933** | 4.667 | **4.680** | 48 |

The paired combined result was close:

- Agentic wins: 16
- Fixed wins: 18
- Ties: 41

The Agentic workflow remains the application default because it can handle less search-ready questions through query rewriting and decomposition. However, the current source-generated evaluation questions already resemble effective search queries, so this dataset does not demonstrate a clear overall advantage for the Agentic approach.

The Agentic run averaged 1.04 searches per question, used at most two searches in this evaluation, and returned evidence for all 75 questions.

Detailed results are stored in [`data/evaluation/answer_metrics.json`](data/evaluation/answer_metrics.json).

Run answer generation and judging:

```bash
uv run python -m evaluation.evaluate_answers
```

This evaluation calls the OpenAI API and can incur usage costs.

## Evaluation Limitations

- The questions were generated from source documents rather than collected from real users.
- Source-grounded questions can favor direct keyword search and under-test query rewriting.
- Relevance and answer judgments use an LLM rather than human annotators.
- The generator, answer model, and judge are not fully independent.
- Pooled recall measures recall against the judged candidate pool, not against every possibly relevant passage in the full corpus.

Testing with real consumer questions and human review is planned for a later version. Candidate sources and a recorded real-question failure case are described in [`docs/future_work.md`](docs/future_work.md).

## Disclaimer

Supplement Evidence Lens summarizes retrieved official-source excerpts. It does not diagnose conditions, recommend treatment, or replace advice from a qualified health professional. Users should consult a health professional before changing medication or supplement use, particularly for children, pregnancy, existing health conditions, or possible interactions.
