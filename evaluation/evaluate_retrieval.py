"""Evaluate BM25, vector, and hybrid retrieval.

Run from the project root:

    uv run python -m evaluation.evaluate_retrieval
"""

import json
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from app.retrieval import (
    RERANKER_MODEL_NAME,
    rerank_results,
)

# Configuration

POOL_PATH = Path("data/evaluation/relevance_pool.jsonl")
JUDGMENTS_PATH = Path(
    "data/evaluation/relevance_judgments.jsonl"
)

METHODS = (
    "bm25",
    "vector",
    "hybrid",
    "hybrid_reranked",
)
CUTOFF = 5

OUTPUT_PATH = Path(
    "data/evaluation/retrieval_metrics.json"
)

# Data loading

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


def build_evaluation_records() -> list[dict[str, Any]]:
    pooled_records = load_jsonl(POOL_PATH)
    judgment_records = {
        record["question_id"]: record
        for record in load_jsonl(JUDGMENTS_PATH)
    }

    evaluation_records = []

    for pooled_record in pooled_records:
        question_id = pooled_record["question_id"]

        if question_id not in judgment_records:
            raise ValueError(
                f"Missing judgments for {question_id}"
            )

        judged_relevant_ids = judgment_records[
            question_id
        ]["relevant_document_ids"]

        relevant_document_ids = set(judged_relevant_ids)
        relevant_document_ids.add(
            pooled_record["seed_document_id"]
        )

        evaluation_records.append(
            {
                "question_id": question_id,
                "question": pooled_record["question"],
                "source": pooled_record["source"],
                "candidates": pooled_record["candidates"],
                "relevant_document_ids": relevant_document_ids,
            }
        )

    return evaluation_records


def ranked_document_ids(
    record: dict[str, Any],
    method: str,
) -> list[str]:
    if method == "hybrid_reranked":
        return record["reranked_document_ids"][:CUTOFF]

    ranked_candidates = [
        candidate
        for candidate in record["candidates"]
        if method in candidate["retrieved_by"]
    ]

    ranked_candidates.sort(
        key=lambda candidate: candidate["retrieved_by"][method]
    )

    return [
        candidate["document_id"]
        for candidate in ranked_candidates[:CUTOFF]
    ]

def add_reranked_results(
    records: list[dict[str, Any]],
    model: CrossEncoder,
) -> None:
    for number, record in enumerate(records, start=1):
        results = [
            {
                "_id": candidate["document_id"],
                "_source": {
                    "title": candidate["title"],
                    "content": candidate["content"],
                },
            }
            for candidate in record["candidates"]
        ]

        reranked = rerank_results(
            model,
            record["question"],
            results,
            limit=CUTOFF,
        )

        record["reranked_document_ids"] = [
            result["_id"]
            for result in reranked
        ]

        print(
            f"[{number}/{len(records)}] Reranked "
            f"{len(results)} candidates"
        )

def calculate_metrics(
    records: list[dict[str, Any]],
    method: str,
) -> dict[str, float]:
    hits = []
    reciprocal_ranks = []
    recalls = []

    for record in records:
        retrieved_ids = ranked_document_ids(
            record,
            method,
        )
        relevant_ids = record["relevant_document_ids"]

        relevant_ranks = [
            rank
            for rank, document_id in enumerate(
                retrieved_ids,
                start=1,
            )
            if document_id in relevant_ids
        ]

        hits.append(1.0 if relevant_ranks else 0.0)

        reciprocal_ranks.append(
            1 / relevant_ranks[0]
            if relevant_ranks
            else 0.0
        )

        retrieved_relevant_ids = (
            set(retrieved_ids) & relevant_ids
        )
        recalls.append(
            len(retrieved_relevant_ids)
            / len(relevant_ids)
        )

    question_count = len(records)

    return {
        "hit_rate@5": sum(hits) / question_count,
        "mrr@5": sum(reciprocal_ranks) / question_count,
        "pooled_recall@5": sum(recalls) / question_count,
    }

def evaluate_methods(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        method: calculate_metrics(records, method)
        for method in METHODS
    }


def print_metrics(
    title: str,
    metrics_by_method: dict[str, dict[str, float]],
) -> None:
    print(f"\n{title}")

    for method, metrics in metrics_by_method.items():
        print(f"\n  {method.upper()}")
        print(
            f"    Hit Rate@5: "
            f"{metrics['hit_rate@5']:.3f}"
        )
        print(f"    MRR@5: {metrics['mrr@5']:.3f}")
        print(
            f"    Pooled Recall@5: "
            f"{metrics['pooled_recall@5']:.3f}"
        )

def main() -> None:
    records = build_evaluation_records()

    reranker_model = CrossEncoder(
        RERANKER_MODEL_NAME
    )
    add_reranked_results(records, reranker_model)

    results = {
        "question_count": len(records),
        "cutoff": CUTOFF,
        "overall": evaluate_methods(records),
        "by_source": {},
    }

    sources = sorted(
        {record["source"] for record in records}
    )

    for source in sources:
        source_records = [
            record
            for record in records
            if record["source"] == source
        ]

        results["by_source"][source] = {
            "question_count": len(source_records),
            "metrics": evaluate_methods(source_records),
        }

    print_metrics("OVERALL", results["overall"])

    for source, source_result in results["by_source"].items():
        print_metrics(
            f"{source} ({source_result['question_count']} questions)",
            source_result["metrics"],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
