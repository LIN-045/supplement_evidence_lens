"""Generate and evaluate answers from the RAG application.

Run from the project root:

    uv run python -m evaluation.evaluate_answers
"""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

from app.rag import (
    LLM_MODEL_NAME,
    PROJECT_ROOT,
    answer_question,
    build_sources,
    compact_citations,
)
from app.retrieval import (
    ELASTICSEARCH_URL,
    MODEL_NAME as EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    search,
)


# Configuration

QUESTIONS_PATH = Path("data/evaluation/questions.jsonl")
ANSWERS_PATH = Path("data/evaluation/rag_answers.jsonl")
JUDGMENTS_PATH = Path("data/evaluation/answer_judgments.jsonl")
JUDGE_MODEL_NAME = "gpt-5.4-mini"
ANSWER_JUDGE_INSTRUCTIONS = """
Evaluate a RAG answer to a dietary supplement question using only the supplied
question, answer, and cited contexts.

Score three dimensions from 1 to 5:

Answer relevance:
- 5: Directly and fully answers the question without unnecessary material.
- 3: Partially answers it or includes noticeable irrelevant material.
- 1: Does not answer the question.

Faithfulness:
- 5: All factual claims are supported by the supplied contexts.
- 3: Mostly supported, but contains a meaningful unsupported inference.
- 1: Major claims contradict or are unsupported by the contexts.

Citation correctness:
- 5: Citations are attached to the appropriate claims and each cited context
  supports those claims.
- 3: Citations are generally useful but some are incomplete or weakly matched.
- 1: Citations are missing, misleading, or do not support the claims.

For each dimension, provide a brief evidence-based reason before assigning the
score. Do not evaluate the truth of the official sources themselves.
"""
FIXED_RAG_INSTRUCTIONS = """
Answer the dietary supplement question using only the supplied official-source
excerpts.

Answer rules:
- Directly answer the user's question.
- Cite supporting excerpts with their reference numbers, such as [1] or [1, 2].
- Do not add factual claims unsupported by the excerpts.
- Do not treat "studied for" as proof of effectiveness.
- Preserve distinctions such as "studied for", "may help", and "authorised
  claim".
- Distinguish regulatory claims from general health information.
- For dose questions, report only source-stated dose ranges and their use,
  population, jurisdiction, and conditions.
- Do not turn source-stated doses into personalised recommendations.
- Do not assume that the upper end of a dose range is more effective, optimal,
  safer, or an appropriate starting dose.
- If the excerpts do not answer the question, say so explicitly.
- Do not make definitive legal judgments or provide personalised medical advice.
- Keep the answer clear and concise.
"""

BASELINE_ANSWERS_PATH = Path(
    "data/evaluation/baseline_answers.jsonl"
)
BASELINE_JUDGMENTS_PATH = Path(
    "data/evaluation/baseline_answer_judgments.jsonl"
)
METRICS_PATH = Path(
    "data/evaluation/answer_metrics.json"
)

# Data loading

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records

def generate_fixed_rag_answer(
    question: str,
    elasticsearch_client: Elasticsearch,
    embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    openai_client: OpenAI,
) -> dict[str, Any]:
    results = search(
        elasticsearch_client,
        embedding_model,
        question,
        reranker_model,
    )

    contexts_for_prompt = []

    for reference_number, result in enumerate(
        results,
        start=1,
    ):
        document = result["_source"]

        contexts_for_prompt.append(
            {
                "reference": reference_number,
                "title": document["title"],
                "source": document["source"],
                "jurisdiction": document["jurisdiction"],
                "source_url": document["source_url"],
                "excerpt": document["content"],
            }
        )

    prompt = json.dumps(
        {
            "question": question,
            "contexts": contexts_for_prompt,
        },
        ensure_ascii=False,
    )

    response = openai_client.responses.create(
        model=LLM_MODEL_NAME,
        instructions=FIXED_RAG_INSTRUCTIONS,
        input=prompt,
    )

    answer, cited_results = compact_citations(
        response.output_text,
        results,
    )
    full_response = (
        f"{answer}\n\n"
        f"{build_sources(cited_results)}"
    )

    cited_contexts = []

    for reference_number, result in enumerate(
        cited_results,
        start=1,
    ):
        document = result["_source"]

        cited_contexts.append(
            {
                "reference": reference_number,
                "document_id": result["_id"],
                "title": document["title"],
                "source": document["source"],
                "jurisdiction": document["jurisdiction"],
                "source_url": document["source_url"],
                "content": document["content"],
            }
        )

    return {
        "question": question,
        "answer": answer,
        "response": full_response,
        "search_queries": [question],
        "contexts": cited_contexts,
    }

def judge_answer(
    client: OpenAI,
    record: dict[str, Any],
) -> dict[str, Any]:
    prompt = json.dumps(
        {
            "question": record["question"],
            "answer": record["answer"],
            "cited_contexts": record["contexts"],
        },
        ensure_ascii=False,
    )

    response = client.responses.create(
        model=JUDGE_MODEL_NAME,
        instructions=ANSWER_JUDGE_INSTRUCTIONS,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_answer_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer_relevance": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string"},
                                "score": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                },
                            },
                            "required": ["reason", "score"],
                            "additionalProperties": False,
                        },
                        "faithfulness": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string"},
                                "score": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                },
                            },
                            "required": ["reason", "score"],
                            "additionalProperties": False,
                        },
                        "citation_correctness": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string"},
                                "score": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5,
                                },
                            },
                            "required": ["reason", "score"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "answer_relevance",
                        "faithfulness",
                        "citation_correctness",
                    ],
                    "additionalProperties": False,
                },
            }
        },
    )

    return json.loads(response.output_text)


def summarize_judgments(
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "answer_relevance",
        "faithfulness",
        "citation_correctness",
    )

    averages = {
        metric: sum(
            record[metric]["score"]
            for record in judgments
        )
        / len(judgments)
        for metric in metric_names
    }
    perfect_answers = sum(
        all(record[metric]["score"] == 5 for metric in metric_names)
        for record in judgments
    )

    return {
        "question_count": len(judgments),
        "average_scores": averages,
        "perfect_answers": perfect_answers,
    }


def summarize_trajectory(
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    search_counts = [
        len(record["search_queries"])
        for record in answers
    ]
    context_counts = [
        len(record["contexts"])
        for record in answers
    ]

    return {
        "average_searches": sum(search_counts) / len(search_counts),
        "maximum_searches": max(search_counts),
        "multi_search_answers": sum(
            count > 1
            for count in search_counts
        ),
        "average_cited_contexts": (
            sum(context_counts) / len(context_counts)
        ),
        "answers_without_cited_contexts": sum(
            count == 0
            for count in context_counts
        ),
    }


def compare_judgments(
    agentic_judgments: list[dict[str, Any]],
    fixed_judgments: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    agentic_by_id = {
        record["question_id"]: record
        for record in agentic_judgments
    }
    fixed_by_id = {
        record["question_id"]: record
        for record in fixed_judgments
    }

    if set(agentic_by_id) != set(fixed_by_id):
        raise ValueError(
            "Agentic and fixed judgments contain different question IDs"
        )

    metric_names = (
        "answer_relevance",
        "faithfulness",
        "citation_correctness",
    )

    def compare_scores(
        agentic_score: int,
        fixed_score: int,
        counts: dict[str, int],
    ) -> None:
        if agentic_score > fixed_score:
            counts["agentic_wins"] += 1
        elif fixed_score > agentic_score:
            counts["fixed_wins"] += 1
        else:
            counts["ties"] += 1

    comparisons: dict[str, dict[str, int]] = {}

    for metric in metric_names:
        counts = {
            "agentic_wins": 0,
            "fixed_wins": 0,
            "ties": 0,
        }

        for question_id in agentic_by_id:
            compare_scores(
                agentic_by_id[question_id][metric]["score"],
                fixed_by_id[question_id][metric]["score"],
                counts,
            )

        comparisons[metric] = counts

    combined_counts = {
        "agentic_wins": 0,
        "fixed_wins": 0,
        "ties": 0,
    }

    for question_id in agentic_by_id:
        agentic_total = sum(
            agentic_by_id[question_id][metric]["score"]
            for metric in metric_names
        )
        fixed_total = sum(
            fixed_by_id[question_id][metric]["score"]
            for metric in metric_names
        )
        compare_scores(
            agentic_total,
            fixed_total,
            combined_counts,
        )

    comparisons["combined_score"] = combined_counts
    return comparisons


def write_answer_metrics() -> None:
    agentic_answers = load_jsonl(ANSWERS_PATH)
    fixed_answers = load_jsonl(BASELINE_ANSWERS_PATH)
    agentic_judgments = load_jsonl(JUDGMENTS_PATH)
    fixed_judgments = load_jsonl(BASELINE_JUDGMENTS_PATH)

    results = {
        "agentic_rag": {
            **summarize_judgments(agentic_judgments),
            "trajectory": summarize_trajectory(agentic_answers),
        },
        "fixed_rag": {
            **summarize_judgments(fixed_judgments),
            "trajectory": summarize_trajectory(fixed_answers),
        },
        "paired_comparison": compare_judgments(
            agentic_judgments,
            fixed_judgments,
        ),
        "selected_approach": "agentic_rag",
        "selection_rationale": (
            "The two approaches were effectively tied on source-generated "
            "questions. Agentic RAG was retained for its ability to rewrite, "
            "decompose, and refine less search-ready real-world questions."
        ),
        "limitations": [
            "Questions were generated from source documents and are often "
            "already search-ready.",
            "The answer generator and judge both used gpt-5.4-mini.",
            "Real-world query rewriting behavior requires a separate challenge "
            "set or production query logs.",
        ],
    }

    METRICS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote answer metrics to {METRICS_PATH}")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    questions = load_jsonl(QUESTIONS_PATH)

    elasticsearch_client = Elasticsearch(
        ELASTICSEARCH_URL
    )
    if not elasticsearch_client.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at "
            f"{ELASTICSEARCH_URL}"
        )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )
    reranker_model = CrossEncoder(
        RERANKER_MODEL_NAME
    )
    openai_client = OpenAI()

    completed_ids = set()

    if ANSWERS_PATH.exists():
        completed_records = load_jsonl(ANSWERS_PATH)
        completed_ids = {
            record["question_id"]
            for record in completed_records
        }

    ANSWERS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ANSWERS_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        for number, question_record in enumerate(
            questions,
            start=1,
        ):
            question_id = question_record["question_id"]

            if question_id in completed_ids:
                print(
                    f"[{number}/{len(questions)}] "
                    f"Skipped {question_id}"
                )
                continue

            result = answer_question(
                question_record["question"],
                elasticsearch_client,
                embedding_model,
                reranker_model,
                openai_client,
                return_trace=True,
            )

            if not isinstance(result, dict):
                raise TypeError(
                    f"Expected trace result for {question_id}"
                )

            output_record = {
                "question_id": question_id,
                "source": question_record["source"],
                "seed_document_id": (
                    question_record["seed_document_id"]
                ),
                **result,
            }

            file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()

            print(
                f"[{number}/{len(questions)}] "
                f"Wrote {question_id}: "
                f"{len(result['contexts'])} cited contexts"
            )

    print(f"Wrote answers to {ANSWERS_PATH}")

    answers = load_jsonl(ANSWERS_PATH)
    judged_ids = set()

    if JUDGMENTS_PATH.exists():
        existing_judgments = load_jsonl(
            JUDGMENTS_PATH
        )
        judged_ids = {
            record["question_id"]
            for record in existing_judgments
        }

    with JUDGMENTS_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        for number, answer_record in enumerate(
            answers,
            start=1,
        ):
            question_id = answer_record["question_id"]

            if question_id in judged_ids:
                print(
                    f"[{number}/{len(answers)}] "
                    f"Skipped judgment {question_id}"
                )
                continue

            judgment = judge_answer(
                openai_client,
                answer_record,
            )

            output_record = {
                "question_id": question_id,
                "source": answer_record["source"],
                **judgment,
            }

            file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()

            print(
                f"[{number}/{len(answers)}] "
                f"{question_id}: "
                f"relevance="
                f"{judgment['answer_relevance']['score']}, "
                f"faithfulness="
                f"{judgment['faithfulness']['score']}, "
                f"citations="
                f"{judgment['citation_correctness']['score']}"
            )

    print(f"Wrote judgments to {JUDGMENTS_PATH}")

    baseline_completed_ids = set()

    if BASELINE_ANSWERS_PATH.exists():
        baseline_completed = load_jsonl(
            BASELINE_ANSWERS_PATH
        )
        baseline_completed_ids = {
            record["question_id"]
            for record in baseline_completed
        }

    with BASELINE_ANSWERS_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        for number, question_record in enumerate(
            questions,
            start=1,
        ):
            question_id = question_record["question_id"]

            if question_id in baseline_completed_ids:
                print(
                    f"[{number}/{len(questions)}] "
                    f"Skipped baseline {question_id}"
                )
                continue

            result = generate_fixed_rag_answer(
                question_record["question"],
                elasticsearch_client,
                embedding_model,
                reranker_model,
                openai_client,
            )

            output_record = {
                "question_id": question_id,
                "source": question_record["source"],
                "seed_document_id": (
                    question_record["seed_document_id"]
                ),
                **result,
            }

            file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()

            print(
                f"[{number}/{len(questions)}] "
                f"Wrote baseline {question_id}"
            )

    print(
        f"Wrote baseline answers to "
        f"{BASELINE_ANSWERS_PATH}"
    )

    baseline_answers = load_jsonl(
        BASELINE_ANSWERS_PATH
    )
    baseline_judged_ids = set()

    if BASELINE_JUDGMENTS_PATH.exists():
        existing_baseline_judgments = load_jsonl(
            BASELINE_JUDGMENTS_PATH
        )
        baseline_judged_ids = {
            record["question_id"]
            for record in existing_baseline_judgments
        }

    with BASELINE_JUDGMENTS_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        for number, answer_record in enumerate(
            baseline_answers,
            start=1,
        ):
            question_id = answer_record["question_id"]

            if question_id in baseline_judged_ids:
                print(
                    f"[{number}/{len(baseline_answers)}] "
                    f"Skipped baseline judgment "
                    f"{question_id}"
                )
                continue

            judgment = judge_answer(
                openai_client,
                answer_record,
            )

            output_record = {
                "question_id": question_id,
                "source": answer_record["source"],
                **judgment,
            }

            file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()

            print(
                f"[{number}/{len(baseline_answers)}] "
                f"{question_id}: "
                f"relevance="
                f"{judgment['answer_relevance']['score']}, "
                f"faithfulness="
                f"{judgment['faithfulness']['score']}, "
                f"citations="
                f"{judgment['citation_correctness']['score']}"
            )

    print(
        f"Wrote baseline judgments to "
        f"{BASELINE_JUDGMENTS_PATH}"
    )

    write_answer_metrics()


if __name__ == "__main__":
    main()
