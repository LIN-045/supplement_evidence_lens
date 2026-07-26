# Supplement Evidence Lens

Ask a dietary supplement question in plain language. Get a cited answer assembled from official regulatory and health sources.

```text
“Magnesium glycinate improves sleep and eliminates anxiety. Is this supported?”
“What supplement ingredients are associated with sleep?”
“What are the recognised uses and dose range for melatonin?”
“What safety concerns apply to high-dose zinc?”
```

---

## Problem

Supplement marketing may say:

> “Boosts immunity”  
> “Your late-night rescue”  
> “Eliminates stress”

Regulatory sources use more precise language, such as:

> “Contributes to the normal function of the immune system.”

These statements may refer to similar concepts while sharing few words, making official information difficult to find through ordinary keyword search.

Relevant information is also distributed across multiple technical sources that cannot easily be searched together:

- The **EU Register of Nutrition and Health Claims** contains authorised and non-authorised health claims, official wording, and conditions of use.
- **Health Canada’s NHPID and Compendium of Monographs** provide ingredient terminology, pre-cleared uses, dose conditions, target populations, duration of use, and risk information.
- **NIH Office of Dietary Supplements fact sheets** provide research context, evidence limitations, upper intake levels, adverse effects, and drug interactions.

Supplement Evidence Lens makes these sources searchable through a single free-text interface and combines relevant results into a cited explanation.

**The application does not independently decide whether a supplement claim is scientifically true.** Scientific conclusions requiring systematic evidence review remain with qualified scientific and regulatory bodies. The application reports what official sources state and clearly indicates when those sources do not cover a question.

---

## Data Sources

| Source                                           | Provides                                                                                               | Access                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| EU Register of Nutrition and Health Claims       | Claim status, official wording, health relationships, and conditions of use                            | Structured download                     |
| Health Canada NHPID and Compendium of Monographs | Ingredient terminology, pre-cleared uses, dose conditions, populations, duration, and risk information | Searchable database and HTML monographs |
| NIH ODS Fact Sheets                              | Research findings, evidence limitations, upper limits, adverse effects, and interactions               | HTML                                    |

Versioned source snapshots or reproducible download manifests are stored under `data/raw/`, depending on source size and redistribution constraints.

---

## How It Works

```text
Free-text question
        │
[LLM] Interpret the query:
      ingredient, health concern, claim wording, dose
        │
[LLM] Decompose complex questions and rewrite retrieval queries
        │
Source-specific hybrid retrieval
├── EU Register claims
├── Health Canada monographs
└── NIH ODS sections
        │
Lexical search + vector search + reranking
        │
[Tools] Structured lookups:
        claim status, dose comparison, upper-limit checks
        │
[LLM] Filter context, compare wording, synthesise, and cite
```

The parsed query determines the retrieval emphasis for each source. A claim-wording question may prioritise the EU Register, while a dose or safety question may retrieve more context from Health Canada and NIH ODS.

The system explicitly states when a source does not contain relevant information.

---

## Why the Sources Are Indexed Separately

The sources have different schemas and retrieval units:

- EU claims are short, structured regulatory records.
- Health Canada monographs contain conditional relationships between ingredient, purpose, dose, population, and risk.
- NIH ODS fact sheets are longer documents requiring section-level chunking.

Separate indexes allow each source to use appropriate metadata, chunking rules, retrieval thresholds, and result limits.

The EU Register is stored both as structured records and as a searchable index:

- structured records provide exact claim status and official wording;
- retrieval connects informal user language with the most relevant regulatory records.

For example, a database lookup may not connect:

> “Your late-night rescue”

with:

> “Contributes to the reduction of tiredness and fatigue.”

Finding the correct record despite this wording gap is a central retrieval problem in the project.

---

## Role of the LLM

| LLM-assisted tasks                                         | Deterministic tasks                                               |
| ---------------------------------------------------------- | ----------------------------------------------------------------- |
| Interpreting open-ended questions                          | Reading claim status from source records                          |
| Extracting ingredients, concerns, claims, and doses        | Returning official claim wording                                  |
| Splitting compound questions into sub-queries              | Reading dose ranges and conditions of use                         |
| Normalising informal and promotional language              | Dose arithmetic and upper-limit comparison                        |
| Rewriting retrieval queries                                | Verifying structured source fields                                |
| Identifying potentially therapeutic or overstated language | Flagging high-risk terms such as _treat_, _cure_, and _eliminate_ |
| Comparing user wording with official wording               | Attaching the correct source and jurisdiction                     |
| Synthesising results into a readable explanation           |                                                                   |

Facts that map to explicit source fields are read directly from the records rather than generated by the LLM.

The LLM is responsible for language understanding and explanation, not for inventing regulatory status, dose limits, or safety conditions.

---

## Example Output

For the query:

> “Magnesium glycinate improves sleep and eliminates anxiety.”

The response may include:

1. **Query interpretation**
   - Ingredient: magnesium glycinate
   - Health concerns: sleep and anxiety
   - Claim strength: improvement and elimination

2. **Closest official claims**
   - Relevant EU-authorised wording for magnesium
   - Relevant Health Canada uses, where available

3. **Wording comparison**
   - Which parts are covered by official wording
   - Which parts are broader or stronger
   - Which parts have no matching official claim

4. **Dose and conditions**
   - Recognised dose conditions
   - Population or duration restrictions
   - Upper-limit comparison, when applicable

5. **Evidence and safety context**
   - NIH research summary
   - Evidence limitations
   - Adverse effects and interactions

6. **Citations**
   - Direct links to the retrieved official records

The application does not make a definitive legal judgment that a product or claim is unlawful.

---

## Retrieval Evaluation

Retrieval evaluation is implemented in:

```text
notebooks/01-retrieval-eval.ipynb
```

Each source index is evaluated separately because retrieval difficulty and document structure differ across sources.

### Retrieval configurations

Four configurations are compared:

1. lexical search;
2. vector search;
3. hybrid search;
4. hybrid search with reranking.

Query rewriting is evaluated as a separate ablation:

- original user query;
- LLM-rewritten queries.

### Metrics

| Index                    | Hit Rate@5 | Recall@5 | MRR |
| ------------------------ | ---------: | -------: | --: |
| EU Register claims       |            |          |     |
| Health Canada monographs |            |          |     |
| NIH ODS sections         |            |          |     |

Additional task-specific metrics may include:

- top-result accuracy;
- ingredient extraction accuracy;
- sub-query coverage;
- source coverage.

### Evaluation dataset

Candidate queries are generated from indexed records and manually reviewed.

The test set also includes independently written free-text questions representing:

- informal language;
- marketing language;
- paraphrased claims;
- compound claims;
- misspellings;
- ingredient-first questions;
- health-concern-first questions;
- questions with no supported answer.

Development queries may be used for retrieval tuning. Final test queries are kept separate from configuration selection.

**Best retrieval configuration: `[to be completed]`**

The selected configuration is used in `app/retrieval.py`.

---

## LLM Evaluation

LLM evaluation is implemented in:

```text
notebooks/02-llm-eval.ipynb
```

Three prompt strategies are compared:

| Prompt strategy                 | Relevance | Groundedness | Correct abstention |
| ------------------------------- | --------: | -----------: | -----------------: |
| Zero-shot                       |           |              |                    |
| Few-shot with explicit criteria |           |              |                    |
| Decompose then synthesise       |           |              |                    |

Evaluation focuses on:

- correct query interpretation;
- claim decomposition;
- ingredient and health-concern extraction;
- faithfulness to retrieved context;
- correct explanation of wording differences;
- citation support;
- correct abstention when sources do not cover the question;
- avoidance of unsupported medical or legal conclusions.

**Best prompt strategy: `[to be completed]`**

The selected prompt is stored in `app/prompts.py`.

---

## Structured Factual Checks

Facts that map to explicit structured fields are checked automatically:

- cited records belong to the retrieved result set;
- EU authorisation status matches the source record;
- official wording matches the source record;
- dose values and units match parsed monograph fields;
- conditions of use are attached to the correct ingredient and purpose;
- source links and jurisdictions are correct.

Statements synthesised from narrative NIH ODS text are evaluated separately for faithfulness and citation support.

---

## Judge Reliability

An LLM judge may produce incorrect evaluation scores.

A manually annotated sample is therefore compared with the automated judge using a written rubric.

```text
evaluation/rubric.md
evaluation/spot_check.csv
```

**Manual sample size:** `[N]`  
**Judge agreement:** `[to be completed]%`

---

## Ingestion

The ingestion pipeline is implemented in:

```text
ingestion/pipeline.py
```

The pipeline performs:

```text
fetch
  → parse
  → normalise
  → validate
  → chunk
  → embed
  → index
```

Rebuild the datasets and indexes with:

```bash
docker compose run app python ingestion/pipeline.py
```

Ingredient names differ across jurisdictions and sources, for example:

```text
Vitamin C
Ascorbic acid
L-ascorbic acid
```

Name alignment uses:

```text
data/ingredient_map.csv
```

The initial mapping is seeded from proper-name, common-name, and synonym fields available in Health Canada data, then reviewed for ambiguous cases.

The pipeline processes all successfully accessible source records. Formal evaluation focuses on a representative subset of common ingredients and health concerns.

---

## Interface

The application provides one free-text input box.

There is no required ingredient dropdown or fixed question type. Interpreting ordinary and inconsistent user language is part of the problem being solved.

Users can:

- ask a supplement-related question;
- inspect the parsed query;
- read the generated answer;
- inspect retrieved source records;
- follow citations;
- submit thumbs-up or thumbs-down feedback.

---

## Monitoring

Each request logs:

- original query;
- parsed query;
- rewritten retrieval queries;
- retrieved records and scores;
- retrieval configuration;
- generated response;
- source coverage;
- response latency;
- user feedback.

The monitoring dashboard is implemented in:

```text
monitoring/dashboard.py
```

Planned dashboard views include:

- query volume over time;
- feedback rate;
- response latency distribution;
- retrieval-score distribution;
- source coverage per query;
- share of answers that abstained because of insufficient coverage.

---

## Running the Project

```bash
git clone [repo-url]
cd supplement-evidence-lens
cp .env.example .env
```

Add the required API credentials to `.env`, then run:

```bash
docker compose up
```

Open:

```text
http://localhost:8501
```

`docker-compose.yml` runs the application, search infrastructure, and logging database.

The system is designed to run without a GPU. Dependencies are version-pinned in `requirements.txt`.

**Live demo:** `[url]`

---

## Repository Layout

```text
ingestion/
    pipeline.py
    sources/
    parsers/
    index.py

data/
    raw/
    processed/
    ingredient_map.csv
    ground_truth/

app/
    main.py
    agent.py
    retrieval.py
    tools.py
    prompts.py
    db.py

notebooks/
    01-retrieval-eval.ipynb
    02-llm-eval.ipynb

evaluation/
    rubric.md
    spot_check.csv

monitoring/
    dashboard.py
```

---

## Scope and Limitations

The initial version focuses on:

- ingredient-level dietary supplement information;
- English-language free-text questions;
- EU and Canadian regulatory references;
- NIH evidence and safety context;
- recognised uses, claims, doses, and safety conditions.

Users may paste wording from a real product, but the application does not verify the product’s quality, authenticity, composition, or overall legal compliance.

Dose comparison is limited to cases where dose units and ingredient forms can be meaningfully aligned.

It is initially enabled for straightforward single compounds such as vitamins, minerals, and melatonin.

Dose comparison is disabled when equivalence cannot be established reliably, including many botanical extracts. For example, 500 mg of a concentrated extract cannot automatically be compared with 500 mg of raw plant material.

The application does not provide:

- personalised supplement recommendations;
- diagnosis or treatment advice;
- individual medication-safety decisions;
- product-quality or contamination testing;
- definitive legal compliance determinations.

Regulatory conclusions are jurisdiction-specific. An EU non-authorised claim does not automatically determine a product’s legal status in another country.

---

## Future Work

Potential extensions include:

- human-scored explanation clarity;
- US supplement product-label data;
- FDA warning letters and recalls;
- broader drug–supplement interaction coverage;
- product and multi-ingredient formula analysis;
- multilingual queries;
- improved ingredient-form and botanical-extract normalisation.

---

## Project Evaluation Criteria

| Criterion            | Implementation                               |
| -------------------- | -------------------------------------------- |
| Problem description  | `README.md`                                  |
| Retrieval flow       | `app/retrieval.py`, `app/agent.py`           |
| Retrieval evaluation | `notebooks/01-retrieval-eval.ipynb`          |
| LLM evaluation       | `notebooks/02-llm-eval.ipynb`                |
| Interface            | `app/main.py`                                |
| Ingestion pipeline   | `ingestion/pipeline.py`                      |
| Monitoring           | `monitoring/dashboard.py`                    |
| Containerisation     | `docker-compose.yml`                         |
| Reproducibility      | Running instructions and pinned dependencies |
| Hybrid search        | `app/retrieval.py`                           |
| Document reranking   | `app/retrieval.py`                           |
| Query rewriting      | `app/prompts.py`                             |
| Cloud deployment     | Live demo link                               |

---

## Disclaimer

Supplement Evidence Lens is for educational and informational purposes only.

It is not a substitute for advice from a physician, pharmacist, dietitian, lawyer, or regulatory professional.
