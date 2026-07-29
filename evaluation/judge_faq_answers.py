"""Evaluate FAQ RAG answers against NIH official reference answers.

Run the pilot:

    uv run python -m evaluation.judge_faq_answers \
        --ids nih_faq_019 nih_faq_067 nih_faq_073

Run all remaining answers:

    uv run python -m evaluation.judge_faq_answers
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.rag import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "data/evaluation/faq"

JUDGE_MODEL_NAME = "gpt-5.4-mini"

JUDGE_INSTRUCTIONS = """
Evaluate a dietary-supplement RAG answer using the user's question, the
official NIH ODS reference answer, and the contexts cited by the RAG answer.

Score four dimensions from 1 to 5.

Correctness:
- 5: The answer's important conclusions agree with the official reference
  answer and contain no meaningful error or overstatement.
- 4: Correct overall, with only a minor imprecision.
- 3: Partly correct, but includes a meaningful overstatement, misleading
  framing, or questionable conclusion.
- 2: Contains major errors or substantially conflicts with the reference.
- 1: Fundamentally incorrect.
- Do not lower correctness merely because information is omitted; omissions
  belong under completeness.

Completeness:
- 5: Covers all important conclusions from the reference answer needed to
  answer the user's question.
- 4: Covers the main answer but omits a secondary qualification or safety
  detail.
- 3: Omits at least one important conclusion or limitation.
- 2: Omits several central conclusions.
- 1: Barely addresses the important content of the reference answer.
- Do not require contact details, navigation links, or incidental background
  unless they are necessary to answer the user's question.

Faithfulness:
- 5: All factual claims are supported by the cited contexts.
- 4: Nearly all claims are supported, with one minor unsupported inference.
- 3: Mostly supported, but includes a meaningful unsupported inference.
- 2: Several important claims are unsupported or contradict the contexts.
- 1: The answer is largely unsupported by the cited contexts.

Citation correctness:
- 5: Citations are attached to the appropriate claims and every cited context
  supports those claims.
- 4: Citations are useful overall, with a minor placement or support issue.
- 3: Some citations are weakly matched, incomplete, or ambiguously placed.
- 2: Several citations do not support the associated claims.
- 1: Citations are absent or seriously misleading.

Important judging rules:
- Treat the NIH ODS answer as the official reference for correctness and
  completeness, not as retrieved evidence.
- Judge faithfulness and citation correctness only against the supplied cited
  contexts.
- Watch for overpositive openings, research doses presented like practical
  recommendations, confusion between regulatory permission and clinical
  effectiveness, and missing safety qualifications.
- Give a concise evidence-based reason before each score.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON Lines records."""

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def parse_arguments() -> argparse.Namespace:
    """Parse optional FAQ ID selection."""

    parser = argparse.ArgumentParser(
        description="Judge FAQ answers against NIH references."
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        metavar="FAQ_ID",
        help="Judge only the listed FAQ IDs.",
    )
    parser.add_argument(
        "--run-name",
        default="v1",
        help="Answer/output label, for example v1 or v2 (default: v1).",
    )
    return parser.parse_args()


def select_pending_records(
    records: list[dict[str, Any]],
    requested_ids: list[str] | None,
    completed_ids: set[str],
) -> list[dict[str, Any]]:
    """Select requested answers that have not been judged."""

    records_by_id = {
        record["faq_id"]: record
        for record in records
    }

    if requested_ids:
        unknown_ids = set(requested_ids) - set(records_by_id)

        if unknown_ids:
            unknown = ", ".join(sorted(unknown_ids))
            raise ValueError(f"Unknown or unanswered FAQ IDs: {unknown}")

        selected = [
            records_by_id[faq_id]
            for faq_id in requested_ids
        ]
    else:
        selected = records

    return [
        record
        for record in selected
        if record["faq_id"] not in completed_ids
    ]


def judgment_schema() -> dict[str, Any]:
    """Return the structured-output schema for the four metrics."""

    metric_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
            },
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
        },
        "required": ["reason", "score"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "correctness": metric_schema,
            "completeness": metric_schema,
            "faithfulness": metric_schema,
            "citation_correctness": metric_schema,
        },
        "required": [
            "correctness",
            "completeness",
            "faithfulness",
            "citation_correctness",
        ],
        "additionalProperties": False,
    }


def judge_answer(
    client: OpenAI,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one generated answer."""

    prompt = json.dumps(
        {
            "question": record["question"],
            "official_reference_answer": (
                record["reference_answer"]
            ),
            "rag_answer": record["answer"],
            "cited_contexts": record["contexts"],
        },
        ensure_ascii=False,
    )

    response = client.responses.create(
        model=JUDGE_MODEL_NAME,
        instructions=JUDGE_INSTRUCTIONS,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "faq_answer_evaluation",
                "strict": True,
                "schema": judgment_schema(),
            }
        },
    )

    return json.loads(response.output_text)


def append_record(
    path: Path,
    record: dict[str, Any],
) -> None:
    """Append one judgment for resumable evaluation."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def main() -> None:
    """Judge all selected and pending FAQ answers."""

    arguments = parse_arguments()
    load_dotenv(PROJECT_ROOT / ".env")
    if not arguments.run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("--run-name may contain only letters, numbers, - and _")
    answers_path = OUTPUT_DIR / f"faq_{arguments.run_name}_answers.jsonl"
    output_path = (
        OUTPUT_DIR
        / f"faq_{arguments.run_name}_answer_judgments.jsonl"
    )

    answer_records = load_jsonl(answers_path)
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
        answer_records,
        arguments.ids,
        completed_ids,
    )

    print(f"Generated answers: {len(answer_records)}")
    print(f"Already judged: {len(completed_ids)}")
    print(f"Pending in this run: {len(pending_records)}")

    if not pending_records:
        print("No pending judgments.")
        return

    client = OpenAI()

    for position, answer_record in enumerate(
        pending_records,
        start=1,
    ):
        faq_id = answer_record["faq_id"]

        print(
            f"\n[{position}/{len(pending_records)}] "
            f"Judging {faq_id}"
        )

        judgment = judge_answer(client, answer_record)
        output_record = {
            "faq_id": faq_id,
            "question": answer_record["question"],
            **judgment,
            "needs_coverage_audit": (
                judgment["correctness"]["score"] < 4
                or judgment["completeness"]["score"] < 4
            ),
            "judge_model": JUDGE_MODEL_NAME,
            "judged_at": datetime.now(UTC).isoformat(),
        }

        append_record(output_path, output_record)

        print(
            "Scores: "
            f"correctness={judgment['correctness']['score']}, "
            f"completeness={judgment['completeness']['score']}, "
            f"faithfulness={judgment['faithfulness']['score']}, "
            "citation_correctness="
            f"{judgment['citation_correctness']['score']}"
        )
        print(
            "Coverage audit: "
            f"{output_record['needs_coverage_audit']}"
        )

    print(f"\nWrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
