"""Generate RAG answers for the NIH ODS FAQ holdout dataset.

Run a small pilot:

    uv run python -m evaluation.generate_faq_answers \
        --ids nih_faq_019 nih_faq_067 nih_faq_073

Run all remaining questions:

    uv run python -m evaluation.generate_faq_answers
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.rag import (
    LLM_MODEL_NAME,
    PROJECT_ROOT,
    RAG_VERSION,
    answer_question,
)
from app.retrieval import (
    ELASTICSEARCH_URL,
    INDEX_NAME,
    MODEL_NAME as EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
)


FAQ_PATH = (
    PROJECT_ROOT
    / "data/evaluation/faq/nih_ods_faq_holdout.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "data/evaluation/faq"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON Lines records."""

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def parse_arguments() -> argparse.Namespace:
    """Parse optional pilot-selection arguments."""

    parser = argparse.ArgumentParser(
        description="Generate V1 answers for the NIH FAQ holdout set."
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        metavar="FAQ_ID",
        help="Generate only the listed FAQ IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate at most this many pending answers.",
    )
    parser.add_argument(
        "--run-name",
        default="v1",
        help="Output label, for example v1 or v2 (default: v1).",
    )
    return parser.parse_args()


def select_pending_records(
    records: list[dict[str, Any]],
    requested_ids: list[str] | None,
    completed_ids: set[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select requested records that have not already been generated."""

    records_by_id = {
        record["faq_id"]: record
        for record in records
    }

    if requested_ids:
        unknown_ids = set(requested_ids) - set(records_by_id)

        if unknown_ids:
            unknown = ", ".join(sorted(unknown_ids))
            raise ValueError(f"Unknown FAQ IDs: {unknown}")

        selected = [
            records_by_id[faq_id]
            for faq_id in requested_ids
        ]
    else:
        selected = records

    pending = [
        record
        for record in selected
        if record["faq_id"] not in completed_ids
    ]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")

        pending = pending[:limit]

    return pending


def build_output_record(
    faq_record: dict[str, Any],
    result: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    """Combine the holdout reference with the generated RAG result."""

    return {
        "faq_id": faq_record["faq_id"],
        "sections": faq_record["sections"],
        "question": faq_record["question"],
        "reference_answer": faq_record["reference_answer"],
        "reference_source_url": faq_record["source_url"],
        "reference_source_sha256": faq_record["source_sha256"],
        "answer": result["answer"],
        "response": result["response"],
        "search_queries": result["search_queries"],
        "contexts": result["contexts"],
        "baseline_version": run_name,
        "rag_version": RAG_VERSION,
        "answer_model": LLM_MODEL_NAME,
        "index_name": INDEX_NAME,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one completed answer so interrupted runs can resume."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def main() -> None:
    """Load shared resources once and generate all pending answers."""

    arguments = parse_arguments()
    load_dotenv(PROJECT_ROOT / ".env")
    if not arguments.run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("--run-name may contain only letters, numbers, - and _")
    output_path = OUTPUT_DIR / f"faq_{arguments.run_name}_answers.jsonl"

    faq_records = load_jsonl(FAQ_PATH)
    completed_records = (
        load_jsonl(output_path)
        if output_path.exists()
        else []
    )
    completed_ids = {
        record["faq_id"]
        for record in completed_records
    }
    pending_records = select_pending_records(
        faq_records,
        arguments.ids,
        completed_ids,
        arguments.limit,
    )

    print(f"FAQ records: {len(faq_records)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Pending in this run: {len(pending_records)}")

    if not pending_records:
        print("No pending answers.")
        return

    elasticsearch_client = Elasticsearch(ELASTICSEARCH_URL)

    if not elasticsearch_client.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {ELASTICSEARCH_URL}"
        )

    indexed_count = elasticsearch_client.count(
        index=INDEX_NAME
    )["count"]
    print(f"Indexed documents: {indexed_count}")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    openai_client = OpenAI()

    for position, faq_record in enumerate(
        pending_records,
        start=1,
    ):
        faq_id = faq_record["faq_id"]
        print(
            f"\n[{position}/{len(pending_records)}] "
            f"{faq_id}: {faq_record['question']}"
        )

        result = answer_question(
            faq_record["question"],
            elasticsearch_client,
            embedding_model,
            reranker_model,
            openai_client,
            return_trace=True,
        )

        if not isinstance(result, dict):
            raise TypeError(
                f"Expected traced result for {faq_id}"
            )

        output_record = build_output_record(
            faq_record,
            result,
            arguments.run_name,
        )
        append_record(output_path, output_record)
        print(
            f"Saved {faq_id}: "
            f"{len(result['search_queries'])} searches, "
            f"{len(result['contexts'])} cited contexts"
        )

    print(f"\nWrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
