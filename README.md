# Supplement Evidence Lens

![Official supplement evidence being filtered into a cited answer](assets/supplement-evidence-lens-hero.png)

**Does melatonin really help with sleep? Can taking too much zinc be harmful?
How much magnesium should an adult get each day?...**

It is easy to find answers to supplement questions online, but much harder to
tell what those answers are based on. Reliable information is spread across
government fact sheets, regulatory records, product monographs, nutrient
reference tables, and consumer guidance. A claim that regulators allow, a
summary of clinical research, and a recommended daily intake may all sound
authoritative, but they answer different questions.

**Supplement Evidence Lens** makes this information easier to use. It searches
official sources from the EU, Canada, and the United States, identifies the
evidence most relevant to the user's question, and returns a concise answer
with links to the supporting material.

This is an informational tool to help people find reliable information. The aim is not to recommend a particular supplement or replace professional medical advice.

---

## Evidence sources

| Source                                     | What it contains                                            | Indexed documents |
| ------------------------------------------ | ----------------------------------------------------------- | ----------------: |
| EU Register on Nutrition and Health Claims | Official health-claim decisions and conditions of use       |             2,337 |
| Health Canada NHPID                        | Uses, doses, and safety information from product monographs |             2,620 |
| NIH ODS professional fact sheets           | Research-based nutrient and supplement summaries            |               379 |
| NIH ODS consumer guidance                  | Guidance on choosing and using dietary supplements          |                19 |
| US Dietary Reference Intake tables         | Reference values by nutrient and population group           |             2,022 |
| NCCIH Herbs at a Glance                    | Research and safety summaries for herbs and botanicals      |               224 |
| NIH ODS consumer FAQ                       | Official answers to common supplement questions             |                74 |

After processing, the searchable index contains **7,675 documents and 9,046
chunks**. Raw snapshots, processed documents, chunks, and frozen evaluation artifacts are
included in the repository.

---

## How it works

### Data pipeline

```text
7 official sources
        |
        v
source adapters -> normalized documents -> chunks -> embeddings
                                                        |
                                                        v
                                                 Elasticsearch
```

Prefect orchestrates ingestion as one ordered workflow. It runs the seven source
adapters, then document preparation, chunking, and indexing, while making each
stage and any failure visible as a separate task. A new physical index is
validated before the public `supplement_evidence` alias is switched, so a
failed rebuild does not destroy the working index.

### Answer flow

```text
User question
     |
     v
Agentic RAG chooses a focused search query
     |
     v
BM25 top 20 + vector top 20
     |
     v
Reciprocal Rank Fusion
     |
     v
Cross-encoder reranks the top candidates
     |
     v
Top 5 excerpts return to the agent
     |
     +---- evidence incomplete? ----> another focused search
     |
     v
Cited answer -> citation compaction -> Streamlit UI -> Monitoring
```

Every answer begins with a search. The agent reviews the returned excerpts and
either answers from that evidence or runs a more focused follow-up search when
important information is still missing. It can search up to four times, and a
repeated query reuses its previous results.

![Example of a cited answer about melatonin and sleep](assets/cited-answer-example.jpg)

---

## Models and infrastructure

- **Data orchestration:** Prefect coordinates and tracks the ingestion stages
- **Storage and search:** Elasticsearch 9.4.4
- **Embeddings:** `BAAI/bge-small-en-v1.5`, 384 dimensions
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L6-v2`
- **Answering and search planning:** `gpt-5.4-mini`
- **Application and monitoring:** Streamlit and SQLite
- **Deployment:** Docker Compose with CPU-only PyTorch

---

## Evaluation

Both evaluations use the same balanced set of **105 consumer-style questions:
15 from each source**. Every question is verified as answerable from its source
document. The same set is used to test retrieval and final answers.

### Retrieval evaluation

Retrieval is scored at chunk level. Each question receives a pooled BM25,
vector, and hybrid candidate set; an LLM judge labels each pooled chunk as
relevant or not relevant.

- **Hit Rate@5:** how often at least one relevant chunk appears in the top five.
- **MRR@5:** how early the first relevant chunk appears; rank one scores highest.
- **Pooled Recall@5:** how much of the judged relevant pool appears in the top
  five.

| Method                | Hit Rate@5 |     MRR@5 | Pooled Recall@5 |
| --------------------- | ---------: | --------: | --------------: |
| BM25                  |      0.800 |     0.683 |           0.434 |
| Vector                |      0.895 |     0.878 |           0.581 |
| Hybrid (RRF)          |      0.952 |     0.881 |           0.614 |
| **Hybrid + reranker** |  **0.962** | **0.918** |       **0.690** |

The final application therefore uses **hybrid retrieval (BM25 + vector search,
combined with RRF) followed by cross-encoder reranking**, with the top five
excerpts passed to the RAG workflow. Per-source results are available in
[`retrieval_eval_metrics.json`](data/evaluation/retrieval/retrieval_eval_metrics.json).

### LLM evaluation

The same questions compare two query strategies over the same retriever,
reranker, answer model, and answer rules:

- **Baseline RAG** searches the original question once.
- **Agentic RAG** chooses and, when useful, refines searches.

An LLM judge scores each dimension from 1 to 5:

- **Correctness:** whether the conclusions match the reference evidence.
- **Completeness:** whether the answer covers the important supported points.
- **Faithfulness:** whether factual statements are supported by cited excerpts.
- **Citation correctness:** whether citations are attached to claims they
  actually support.

| Workflow        | Correctness | Completeness | Faithfulness | Citation correctness | Perfect answers |
| --------------- | ----------: | -----------: | -----------: | -------------------: | --------------: |
| Baseline RAG    |       4.533 |        4.390 |        4.562 |                4.438 |              53 |
| **Agentic RAG** |   **4.695** |    **4.600** |    **4.695** |            **4.619** |          **67** |

On combined paired scores, Agentic wins 33 questions, Baseline wins 16, and 56
are tied. Agentic averages 1.038 searches and uses a second search on four
questions. Most questions were generated from a single answerable source
document, so one focused search was usually sufficient. A future challenge set
built from ambiguous, comparative, or multi-part real-world questions would
better test when query refinement and multiple searches add value.

The final application therefore uses **Agentic RAG** over the shared hybrid and
reranked retrieval pipeline: it plans a focused search and can refine or split
the query when another search is useful.

Full results are in
[`llm_eval_metrics.json`](data/evaluation/llm/llm_eval_metrics.json).

The frozen questions, retrieval pools, judgments, answers, and metrics are
stored in [`data/evaluation`](data/evaluation). Docker startup does not rerun
the evaluation.

---

## Monitoring

The application records the question, answer, RAG/model versions, search
queries, cited contexts, latency, and optional user feedback in SQLite.

The Streamlit dashboard includes:

- questions over time
- daily average answer latency
- positive, negative, and unrated feedback
- searches per question
- cited contexts by source
- recent interactions with search queries, feedback, and notes

The dashboard can be filtered by RAG version. These versions identify Prompt
iterations used during development, making behavior changes easier to compare.

![Monitoring dashboard showing usage, latency, feedback, searches, and cited sources](assets/monitoring-dashboard.jpg)

---

## Run the application

The complete application runs with Docker Compose; no local Python installation
is required.

Requirements:

- Docker Desktop with Docker Compose
- an OpenAI API key

Clone the repository and create `.env`:

```bash
git clone https://github.com/LIN-045/supplement_evidence_lens.git
cd supplement_evidence_lens
```

```text
OPENAI_API_KEY=your-api-key
```

Start the complete project:

```bash
docker compose up -d --build
```

Open:

- App: <http://localhost:8501>
- Monitoring: <http://localhost:8502>
- Elasticsearch: <http://localhost:9200>

![Supplement Evidence Lens product interface](assets/product-ui.png)

The first run downloads the embedding model and creates embeddings for the
committed 9,046 chunks. The reranker is downloaded when the application first
loads its RAG resources. These steps do not call OpenAI, but can take several
minutes. Later starts reuse the model cache and Elasticsearch volume. If the
chunk hash, embedding model, and document count are unchanged, the indexer exits
without recomputing embeddings.

Check the services and index:

```bash
docker compose ps -a
docker compose logs indexer
curl http://localhost:9200/supplement_evidence/_count
```

The expected count is `9046`, and the one-shot indexer should exit with code
`0`.

Stop the project without deleting the index or model cache:

```bash
docker compose down
```

Monitoring data is stored locally in `data/monitoring/interactions.db`. It is
ignored by Git and survives container removal.

---

## Developer workflows (optional)

Docker is sufficient for using the application. These workflows are for
reviewers or developers who want to inspect the code, run the tests, refresh
the evidence, or rerun the evaluation.

### Project structure

```text
app/             retrieval, baseline RAG, Agentic RAG, product Streamlit UI
ingestion/       source adapters, chunking, indexing, Prefect flow
evaluation/      question generation, retrieval evaluation, LLM evaluation
monitoring/      SQLite store and monitoring Streamlit dashboard
tests/           ingestion, retrieval, RAG, evaluation, monitoring tests
data/            raw, processed, evaluation, and local monitoring data
Dockerfile       shared CPU-only Python image
docker-compose.yaml
```

### Local setup

The project uses Python 3.13, `uv` 0.11.8, and locked dependencies. From the
cloned repository root, install the development environment:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

### Refresh source data and rebuild the index

The repository already contains the processed data used by the application, so
this workflow is optional. Use it to fetch fresh copies of all seven official
sources and rebuild everything that follows:

```text
source downloads -> processed documents -> chunks -> embeddings -> search index
```

This workflow does not call OpenAI, but it downloads source data and embedding
models and may take several minutes.

Start the optional ingestion services:

```bash
docker compose --profile ingestion up -d --build
```

This starts the Prefect server, registers the `local-manual` deployment, and
starts its worker. It does **not** run ingestion automatically. Open
<http://localhost:4200>, select the deployment, and choose **Run** when the
source data should be refreshed. The UI shows the status and logs for each
stage.

### Rerun the full evaluation

The complete evaluation uses the local `uv` environment and the Elasticsearch
index. Question generation, relevance judging, answer generation, and answer
judging call OpenAI and incur API usage.

```bash
# Retrieval
uv run python -m evaluation.generate_questions
uv run python -m evaluation.retrieval.build_relevance_pool
uv run python -m evaluation.retrieval.judge_relevance
uv run python -m evaluation.retrieval.calculate_retrieval_eval_metrics

# Answers
uv run python -m evaluation.llm.generate_answers
uv run python -m evaluation.llm.judge_answers
uv run python -m evaluation.llm.calculate_llm_eval_metrics
```

Compatible partial outputs resume automatically; hashes prevent results from
different prompts, models, questions, or indexes from being mixed.

---

## Limitations and next steps

- Most evaluation questions are clear and answerable from one source document.
  Real users may ask vaguer questions or combine several concerns in one query.
- Only four questions needed a second Agentic search, so the current evaluation
  does not fully test the value of multi-step searching.
- An LLM judged relevance and answer quality. Human expert review would provide
  a stronger check on those scores.
- Retrieval was judged against a pooled set of likely candidates rather than
  every chunk in the index, so additional relevant evidence may exist outside
  that pool.

The next step is to add a small challenge set of de-identified real questions
and manually review a representative sample of the judgments. Broader US
regulatory data would also make the evidence base more complete.

---

## Disclaimer

Supplement Evidence Lens summarizes retrieved official-source excerpts. Consult
a qualified health professional before changing medication or supplement use,
especially for children, pregnancy, existing conditions, or possible drug
interactions.
