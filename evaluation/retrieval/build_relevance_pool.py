"""Build pooled retrieval candidates for evaluation questions.

Run from the project root:

    uv run python -m evaluation.retrieval.build_relevance_pool
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from app.retrieval import (
    CANDIDATE_COUNT,
    EMBEDDING_MODEL_NAME,
    EvidenceRetriever,
    reciprocal_rank_fusion,
)


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = (
    PROJECT_ROOT / "data/evaluation/questions.jsonl"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/retrieval/relevance_pool.jsonl"
)

POOL_DEPTH = 10


# Data loading

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


def record_sha256(record: dict[str, Any]) -> str:
    """Return a deterministic hash for one JSON-compatible record."""

    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_candidates(
    retriever: EvidenceRetriever,
    question: str,
) -> list[dict[str, Any]]:
    bm25_results = retriever.bm25_search(question)
    vector_results = retriever.vector_search(question)
    hybrid_results = reciprocal_rank_fusion(
        [bm25_results, vector_results],
        limit=CANDIDATE_COUNT,
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
                    "source_document_id": document[
                        "source_document_id"
                    ],
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

    retriever = EvidenceRetriever.from_defaults(
        use_reranker=False,
    )
    index_document_sha256 = (
        retriever.index_document_sha256()
    )

    print(f"Questions: {len(questions)}")
    print("Elasticsearch connected")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    pooled_questions = []

    for number, question_record in enumerate(questions, start=1):
        candidates = build_candidates(
            retriever,
            question_record["question"],
        )

        pooled_record = {
            **question_record,
            "index_document_sha256": (
                index_document_sha256
            ),
            "candidates": candidates,
        }
        pooled_record["pool_hash"] = record_sha256(
            pooled_record
        )
        pooled_questions.append(pooled_record)

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
