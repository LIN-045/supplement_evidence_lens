# Supplement Evidence Lens

Supplement Evidence Lens is an evidence-grounded RAG assistant for questions about dietary supplements.

It retrieves evidence from official EU, Canadian, and US sources, reranks the retrieved passages, and uses an LLM to produce a cited answer through a Streamlit interface.

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

- ingestion from seven official datasets
- normalized document construction and chunking
- safe, versioned Elasticsearch indexing
- BM25, vector, hybrid, and reranked retrieval
- baseline and Agentic RAG workflows
- a Streamlit application and SQLite monitoring
- retrieval and answer-quality evaluation

Still to be built:

- full application containerization
- final deployment documentation and screenshots

## Data Sources

| Source | Jurisdiction | Content | Processed records |
|---|---|---|---:|
| EU Register on Nutrition and Health Claims | EU | authorised and non-authorised health claims | 2,337 |
| Health Canada NHPID | Canada | natural-health-product monographs | 3,009 |
| NIH ODS Professional Fact Sheets | United States | professional supplement fact sheets | 379 |
| NIH ODS Consumer Guidance | United States | consumer supplement guidance | 19 |
| US Dietary Reference Intakes | United States | nutrient reference values | 2,022 |
| NCCIH Herbs at a Glance | United States | herb benefit and safety summaries | 392 |
| NIH ODS FAQ | United States | original consumer questions and answers | 74 |

Document preparation and filtering produce 7,675 complete documents and
9,046 searchable chunks. The index keeps source, jurisdiction, title, section,
URL, parent-document ID, and chunk ID as metadata. A completed physical index
is atomically assigned to the public `supplement_evidence` alias, so a failed
rebuild does not destroy the working index.

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
  retrieval.py                   BM25, vector, hybrid, and reranked search
  base_rag.py                    Single-search baseline RAG
  agentic_rag.py                 Query planning and multi-search RAG
  ui.py                          Streamlit application

ingestion/
  sources/                       Seven source-specific adapters
  chunk_documents.py             Normalization and chunking
  index_documents.py             Safe index build and alias switch
  prefect_flow.py                End-to-end ingestion workflow

evaluation/
  generate_questions.py          134-question dataset construction
  structured_llm.py              Shared structured judge output
  retrieval/                     Pool, relevance judge, and metrics
  llm/                           Answers, answer judge, and metrics

data/
  raw/                           Downloaded source files
  processed/                     Normalized records and chunks
  evaluation/                    Regenerated evaluation artifacts
  monitoring/                    Persistent interaction database
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

Run the complete ingestion pipeline:

```bash
uv run python -m ingestion.prefect_flow
```

Or, after the processed source files exist, rebuild only the common document collection and index:

```bash
uv run python -m ingestion.chunk_documents
uv run python -m ingestion.index_documents
```

Confirm the indexed document count:

```bash
curl http://localhost:9200/supplement_evidence/_count
```

## Run the Assistant

Start the Streamlit application:

```bash
uv run streamlit run app/ui.py
```

The response includes numbered citations and the URLs of the retrieved official sources.

## Retrieval Evaluation

The final evaluation set contains 134 questions: all 74 original NIH ODS FAQ questions plus 10 questions generated from complete documents for each of the other six sources. Retrieval is performed on chunks but scored by parent document ID.

Candidate pools contain BM25, vector, and RRF-hybrid results. An LLM assigns binary relevance judgments. BM25, vector, hybrid, and hybrid-plus-reranker are compared at rank 5.

| Method | Hit Rate@5 | MRR@5 | Pooled Recall@5 |
|---|---:|---:|---:|
| BM25 | 0.881 | 0.819 | 0.447 |
| Vector | 0.963 | 0.936 | 0.658 |
| Hybrid | 0.970 | 0.914 | 0.643 |
| Hybrid + reranker | **0.993** | **0.966** | **0.677** |

Hybrid retrieval with cross-encoder reranking performs best overall. The
source-level results remain important: Health Canada is the hardest subset,
and reranking raises its Hit Rate@5 from 0.600 to 0.900.

Run the retrieval evaluation pipeline:

```bash
uv run python -m evaluation.generate_questions
uv run python -m evaluation.retrieval.build_relevance_pool
uv run python -m evaluation.retrieval.judge_relevance
uv run python -m evaluation.retrieval.calculate_retrieval_eval_metrics
```

## Answer Evaluation

The same 134 questions compare two RAG workflows:

- Fixed RAG: searches the original question once.
- Agentic RAG: allows the model to rewrite, decompose, or repeat searches when useful.

Both approaches use the same index, retrieval pipeline, reranker, and answer model. The judge uses official FAQ answers or complete seed-document context as reference evidence and scores:

- correctness
- completeness
- faithfulness
- citation correctness

Each metric is scored from 1 to 5 with a written reason.

| Workflow | Correctness | Completeness | Faithfulness | Citation correctness | Perfect answers |
|---|---:|---:|---:|---:|---:|
| Baseline RAG | **4.836** | **4.515** | **4.709** | **4.619** | **67** |
| Agentic RAG | 4.761 | 4.440 | 4.649 | 4.612 | 54 |

Baseline wins 47 paired questions, Agentic wins 37, and 50 are tied. The
overall baseline advantage is concentrated in the 74 original FAQ questions,
whose exact wording is also indexed. On the 60 non-FAQ questions, the systems
are approximately tied: Agentic wins 17 paired totals, baseline wins 16, and
27 are tied. Agentic is slightly higher on correctness, faithfulness, and
citation correctness in that subset, while baseline is slightly higher on
completeness.

Agentic averages 1.022 searches and performs multiple searches for only 3 of
134 questions, so this dataset does not establish a strong benefit from
multi-search orchestration.

Run answer generation and judging:

```bash
uv run python -m evaluation.llm.generate_answers
uv run python -m evaluation.llm.judge_answers
uv run python -m evaluation.llm.calculate_llm_eval_metrics
```

Question generation, answer generation, and judging call the OpenAI API and can incur usage costs. Evaluation artifacts store question, answer, prompt/version, model, pool, and index-data hashes; a stale partial run is rejected instead of being silently mixed with new results.

## Evaluation Limitations

- The questions were generated from source documents rather than collected from real users.
- Source-grounded questions can favor direct keyword search and under-test query rewriting.
- Exact-match FAQ questions favor retaining the original query and therefore
  favor the baseline workflow.
- Only three questions triggered multiple Agentic searches, so complex
  decomposition behavior remains under-tested.
- Relevance and answer judgments use an LLM rather than human annotators.
- The generator, answer model, and judge are not fully independent.
- Pooled recall measures recall against the judged candidate pool, not against every possibly relevant passage in the full corpus.

Testing with real consumer questions and human review is planned for a later version. Candidate sources and a recorded real-question failure case are described in [`docs/future_work.md`](docs/future_work.md).

## Disclaimer

Supplement Evidence Lens summarizes retrieved official-source excerpts. It does not diagnose conditions, recommend treatment, or replace advice from a qualified health professional. Users should consult a health professional before changing medication or supplement use, particularly for children, pregnancy, existing health conditions, or possible interactions.
