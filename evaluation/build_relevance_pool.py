"""Build pooled retrieval candidates for evaluation questions.

Run from the project root:

    uv run python -m evaluation.build_relevance_pool
"""

import json
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from app.retrieval import (
    ELASTICSEARCH_URL,
    MODEL_NAME as EMBEDDING_MODEL_NAME,
    RRF_CONSTANT,
    bm25_search,
    vector_search,
)


# Configuration

QUESTIONS_PATH = Path("data/evaluation/questions.jsonl")
OUTPUT_PATH = Path("data/evaluation/relevance_pool.jsonl")

POOL_DEPTH = 10


# Data loading

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records

def hybrid_search(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    fused = {}

    for results in (bm25_results, vector_results):
        for rank, result in enumerate(results, start=1):
            document_id = result["_id"]

            if document_id not in fused:
                fused[document_id] = {
                    **result,
                    "rrf_score": 0.0,
                }

            fused[document_id]["rrf_score"] += (
                1 / (RRF_CONSTANT + rank)
            )

    return sorted(
        fused.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )[:limit]


def build_candidates(
    client: Elasticsearch,
    embedding_model: SentenceTransformer,
    question: str,
) -> list[dict[str, Any]]:
    bm25_results = bm25_search(client, question)
    vector_results = vector_search(
        client,
        embedding_model,
        question,
    )
    hybrid_results = hybrid_search(
        bm25_results,
        vector_results,
        POOL_DEPTH,
    )

    retrieval_runs = {
        "bm25": bm25_results[:POOL_DEPTH],
        "vector": vector_results[:POOL_DEPTH],
        "hybrid": hybrid_results,
    }

    candidates = {}

    for method, results in retrieval_runs.items():
        for rank, result in enumerate(results, start=1):
            document_id = result["_id"]
            document = result["_source"]

            if document_id not in candidates:
                candidates[document_id] = {
                    "document_id": document_id,
                    "title": document["title"],
                    "source": document["source"],
                    "jurisdiction": document["jurisdiction"],
                    "source_url": document["source_url"],
                    "content": document["content"],
                    "retrieved_by": {},
                }

            candidates[document_id]["retrieved_by"][method] = rank

    return list(candidates.values())

def main() -> None:
    questions = load_jsonl(QUESTIONS_PATH)

    client = Elasticsearch(ELASTICSEARCH_URL)
    if not client.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {ELASTICSEARCH_URL}"
        )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(f"Questions: {len(questions)}")
    print("Elasticsearch connected")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    pooled_questions = []

    for number, question_record in enumerate(questions, start=1):
        candidates = build_candidates(
            client,
            embedding_model,
            question_record["question"],
        )

        pooled_questions.append(
            {
                **question_record,
                "candidates": candidates,
            }
        )

        print(
            f"[{number}/{len(questions)}] "
            f"{len(candidates)} pooled candidates"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in pooled_questions:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Wrote {OUTPUT_PATH}")

if __name__ == "__main__":
    main()