"""Judge pooled document relevance for retrieval evaluation.

Run from the project root:

    uv run python -m evaluation.retrieval.judge_relevance
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from evaluation.structured_llm import generate_structured_output


# Configuration

INPUT_PATH = Path(
    "data/evaluation/retrieval/relevance_pool.jsonl"
)
OUTPUT_PATH = Path(
    "data/evaluation/retrieval/relevance_judgments.jsonl"
)

JUDGE_MODEL_NAME = "gpt-5.4-mini"
RELEVANCE_JUDGE_VERSION = "relevance_judge_v1"

JUDGING_INSTRUCTIONS = """
You are judging whether retrieved official-source excerpts are relevant to a
user's dietary supplement question.

Judge each candidate independently.

A candidate is relevant when it contains information that directly answers at
least one meaningful part of the question or provides evidence needed to answer
it.

A candidate is not relevant when it only:
- mentions the same ingredient without addressing the question;
- discusses a different effect, dose, population, or product;
- contains background information that would not help answer the question;
- lists references without useful answer content.

Do not judge whether the source itself is correct. Judge only whether its
content is useful for answering the question.

Return one judgment for every supplied document ID. Use only:
- relevant
- not_relevant
"""

RELEVANCE_JUDGMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_number": {
                        "type": "integer"
                    },
                    "relevance": {
                        "type": "string",
                        "enum": [
                            "relevant",
                            "not_relevant",
                        ],
                    },
                    "reason": {
                        "type": "string"
                    },
                },
                "required": [
                    "candidate_number",
                    "relevance",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgments"],
    "additionalProperties": False,
}


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


def judge_candidates(
    client: OpenAI,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates_for_prompt = []

    for number, candidate in enumerate(
        record["candidates"],
        start=1,
    ):
        candidates_for_prompt.append(
            {
                "candidate_number": number,
                "title": candidate["title"],
                "source": candidate["source"],
                "content": candidate["content"],
            }
        )

    prompt = json.dumps(
        {
            "question": record["question"],
            "candidates": candidates_for_prompt,
        },
        ensure_ascii=False,
    )

    result = generate_structured_output(
        client,
        model=JUDGE_MODEL_NAME,
        instructions=JUDGING_INSTRUCTIONS,
        input_text=prompt,
        schema_name="relevance_judgments",
        schema=RELEVANCE_JUDGMENTS_SCHEMA,
    )
    judgments_by_number = {
        judgment["candidate_number"]: judgment
        for judgment in result["judgments"]
    }

    expected_numbers = set(
        range(1, len(record["candidates"]) + 1)
    )

    if set(judgments_by_number) != expected_numbers:
        raise ValueError(
            f"Incomplete judgments for {record['question_id']}"
        )

    judgments = []

    for number, candidate in enumerate(
        record["candidates"],
        start=1,
    ):
        judgment = judgments_by_number[number]

        judgments.append(
            {
                "document_id": candidate["document_id"],
                "source_document_id": candidate[
                    "source_document_id"
                ],
                "relevance": judgment["relevance"],
                "reason": judgment["reason"],
            }
        )

    return judgments

def main() -> None:
    load_dotenv()

    records = load_jsonl(INPUT_PATH)
    records_by_id = {
        record["question_id"]: record
        for record in records
    }

    if len(records_by_id) != len(records):
        raise ValueError("Relevance pool contains duplicate question IDs")

    for record in records:
        unhashed_record = {
            key: value
            for key, value in record.items()
            if key != "pool_hash"
        }

        if record.get("pool_hash") != record_sha256(
            unhashed_record
        ):
            raise ValueError(
                f"Invalid pool hash for {record['question_id']}"
            )

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    completed_ids = set()

    if OUTPUT_PATH.exists():
        completed_records = load_jsonl(OUTPUT_PATH)
        completed_judge_versions = {
            record.get("judge_version")
            for record in completed_records
        }
        completed_judge_models = {
            record.get("judge_model")
            for record in completed_records
        }

        if (
            completed_judge_versions
            != {RELEVANCE_JUDGE_VERSION}
            or completed_judge_models != {JUDGE_MODEL_NAME}
        ):
            raise ValueError(
                f"{OUTPUT_PATH} contains judgments from a "
                "different judge or model version. Move or "
                "delete that file before continuing."
            )

        for record in completed_records:
            question_id = record["question_id"]

            if (
                question_id not in records_by_id
                or record.get("pool_hash")
                != records_by_id[question_id]["pool_hash"]
            ):
                raise ValueError(
                    f"{OUTPUT_PATH} contains judgments for a "
                    "different relevance pool. Move or delete "
                    "that file before continuing."
                )

        completed_ids = {
            record["question_id"]
            for record in completed_records
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("a", encoding="utf-8") as file:
        for number, record in enumerate(records, start=1):
            question_id = record["question_id"]

            if question_id in completed_ids:
                print(f"[{number}/{len(records)}] Skipped {question_id}")
                continue

            judgments = judge_candidates(client, record)
            relevant_document_ids = [
                judgment["source_document_id"]
                for judgment in judgments
                if judgment["relevance"] == "relevant"
            ]

            output_record = {
                "question_id": question_id,
                "question": record["question"],
                "pool_hash": record["pool_hash"],
                "judge_version": RELEVANCE_JUDGE_VERSION,
                "judge_model": JUDGE_MODEL_NAME,
                "judgments": judgments,
                "relevant_document_ids": relevant_document_ids,
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
                f"[{number}/{len(records)}] "
                f"{question_id}: "
                f"{len(relevant_document_ids)} relevant"
            )

    print(f"Wrote judgments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
