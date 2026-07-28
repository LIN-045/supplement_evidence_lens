"""Judge pooled document relevance for retrieval evaluation.

Run from the project root:

    uv run python -m evaluation.judge_relevance
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# Configuration

INPUT_PATH = Path("data/evaluation/relevance_pool.jsonl")
OUTPUT_PATH = Path("data/evaluation/relevance_judgments.jsonl")

MODEL_NAME = "gpt-5.4-mini"

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


# Data loading

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records

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

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=JUDGING_INSTRUCTIONS,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "relevance_judgments",
                "strict": True,
                "schema": {
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
                },
            }
        },
    )

    result = json.loads(response.output_text)
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
                "relevance": judgment["relevance"],
                "reason": judgment["reason"],
            }
        )

    return judgments

def main() -> None:
    load_dotenv()

    records = load_jsonl(INPUT_PATH)
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    completed_ids = set()

    if OUTPUT_PATH.exists():
        completed_records = load_jsonl(OUTPUT_PATH)
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
                judgment["document_id"]
                for judgment in judgments
                if judgment["relevance"] == "relevant"
            ]

            output_record = {
                "question_id": question_id,
                "question": record["question"],
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