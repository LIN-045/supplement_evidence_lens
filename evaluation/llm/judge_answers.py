"""Judge baseline and agentic RAG answers with one shared rubric.

Run from the project root:

    uv run python -m evaluation.llm.judge_answers
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from evaluation.structured_llm import generate_structured_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data/evaluation/llm"

BASELINE_ANSWERS_PATH = OUTPUT_DIR / "baseline_answers.jsonl"
AGENTIC_ANSWERS_PATH = OUTPUT_DIR / "agentic_answers.jsonl"
BASELINE_JUDGMENTS_PATH = (
    OUTPUT_DIR / "baseline_judgments.jsonl"
)
AGENTIC_JUDGMENTS_PATH = (
    OUTPUT_DIR / "agentic_judgments.jsonl"
)

JUDGE_MODEL_NAME = "gpt-5.4-mini"
ANSWER_JUDGE_VERSION = "answer_judge_v1"

ANSWER_JUDGE_INSTRUCTIONS = """
Evaluate a RAG answer to a dietary supplement question using only the supplied
question, answer, and cited official-source contexts.

Use the official reference answer and reference contexts to judge correctness
and completeness. Use the cited contexts to judge faithfulness and citation
correctness.

Score four dimensions from 1 to 5:

Correctness:
- 5: The answer's conclusions accurately reflect the official reference
  answer or reference contexts.
- 3: The core conclusion is reasonable but contains a meaningful error,
  overstatement, or misinterpretation.
- 1: The central conclusion contradicts or seriously misrepresents the
  evidence.

Completeness:
- 5: The answer addresses every important part of the question that the
  official reference answer or reference contexts can support and clearly
  identifies material uncertainty.
- 3: It answers the main point but omits a meaningful supported qualification
  or part of the question.
- 1: It leaves the main question substantially unanswered.

Faithfulness:
- 5: Every factual claim is supported by the supplied contexts.
- 3: Most claims are supported, but there is a meaningful unsupported
  inference.
- 1: Major claims are unsupported by or contradict the supplied contexts.

Citation correctness:
- 5: Citations are attached to the appropriate claims and each cited context
  supports those claims.
- 3: Citations are generally useful, but some are incomplete or weakly
  matched.
- 1: Citations are missing, misleading, or do not support the claims.

For each dimension, provide a brief evidence-based reason before assigning the
score. Do not use outside knowledge or judge the truth of the official sources.
"""


def _score_schema() -> dict[str, Any]:
    return {
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
    }


METRIC_NAMES = (
    "correctness",
    "completeness",
    "faithfulness",
    "citation_correctness",
)

ANSWER_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        metric: _score_schema()
        for metric in METRIC_NAMES
    },
    "required": list(METRIC_NAMES),
    "additionalProperties": False,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON Lines records."""

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def record_sha256(record: dict[str, Any]) -> str:
    """Return a deterministic hash for one JSON-compatible record."""

    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def judge_answer(
    client: OpenAI,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Judge one answer against its cited contexts."""

    prompt = json.dumps(
        {
            "question": record["question"],
            "official_reference_answer": record[
                "reference_answer"
            ],
            "reference_contexts": record[
                "reference_contexts"
            ],
            "rag_answer": record["answer"],
            "cited_contexts": record["contexts"],
        },
        ensure_ascii=False,
    )

    return generate_structured_output(
        client,
        model=JUDGE_MODEL_NAME,
        instructions=ANSWER_JUDGE_INSTRUCTIONS,
        input_text=prompt,
        schema_name="rag_answer_judgment",
        schema=ANSWER_JUDGMENT_SCHEMA,
    )


def judge_answers(
    client: OpenAI,
    answers_path: Path,
    judgments_path: Path,
) -> None:
    """Judge pending answers from one RAG workflow."""

    answers = load_jsonl(answers_path)
    answer_versions = {
        record.get("rag_version")
        for record in answers
    }

    if len(answer_versions) != 1 or None in answer_versions:
        raise ValueError(
            f"{answers_path} contains mixed or missing RAG versions"
        )

    answer_version = next(iter(answer_versions))
    answers_by_id = {
        record["question_id"]: record
        for record in answers
    }

    if len(answers_by_id) != len(answers):
        raise ValueError(
            f"{answers_path} contains duplicate question IDs"
        )

    for record in answers:
        unhashed_record = {
            key: value
            for key, value in record.items()
            if key != "answer_hash"
        }

        if record.get("answer_hash") != record_sha256(
            unhashed_record
        ):
            raise ValueError(
                f"Invalid answer hash for "
                f"{record['question_id']}"
            )

    completed_records = (
        load_jsonl(judgments_path)
        if judgments_path.exists()
        else []
    )
    completed_versions = {
        record.get("rag_version")
        for record in completed_records
    }
    completed_judge_versions = {
        record.get("judge_version")
        for record in completed_records
    }
    completed_judge_models = {
        record.get("judge_model")
        for record in completed_records
    }

    if completed_records and (
        completed_versions != {answer_version}
        or completed_judge_versions != {ANSWER_JUDGE_VERSION}
        or completed_judge_models != {JUDGE_MODEL_NAME}
    ):
        raise ValueError(
            f"{judgments_path} contains judgments from a "
            "different RAG, judge, or model version. Move or "
            "delete that file before judging the new answers."
        )

    for record in completed_records:
        question_id = record["question_id"]

        if (
            question_id not in answers_by_id
            or record.get("answer_hash")
            != answers_by_id[question_id]["answer_hash"]
        ):
            raise ValueError(
                f"{judgments_path} contains a judgment for a "
                "different answer dataset. Move or delete that "
                "file before continuing."
            )

    completed_ids = {
        record["question_id"]
        for record in completed_records
    }

    judgments_path.parent.mkdir(parents=True, exist_ok=True)

    with judgments_path.open("a", encoding="utf-8") as file:
        for number, answer_record in enumerate(
            answers,
            start=1,
        ):
            question_id = answer_record["question_id"]

            if question_id in completed_ids:
                print(
                    f"[{number}/{len(answers)}] "
                    f"Skipped judgment {question_id}"
                )
                continue

            judgment = judge_answer(client, answer_record)
            output_record = {
                "question_id": question_id,
                "source": answer_record["source"],
                "rag_version": answer_record["rag_version"],
                "answer_hash": answer_record["answer_hash"],
                "judge_version": ANSWER_JUDGE_VERSION,
                "judge_model": JUDGE_MODEL_NAME,
                **judgment,
            }
            file.write(
                json.dumps(output_record, ensure_ascii=False)
                + "\n"
            )
            file.flush()

            scores = ", ".join(
                f"{metric}={judgment[metric]['score']}"
                for metric in METRIC_NAMES
            )
            print(
                f"[{number}/{len(answers)}] "
                f"{question_id}: {scores}"
            )

    print(f"Wrote judgments to {judgments_path}")


def main() -> None:
    """Judge baseline and agentic answer files."""

    load_dotenv(PROJECT_ROOT / ".env")
    client = OpenAI()

    judge_answers(
        client,
        BASELINE_ANSWERS_PATH,
        BASELINE_JUDGMENTS_PATH,
    )
    judge_answers(
        client,
        AGENTIC_ANSWERS_PATH,
        AGENTIC_JUDGMENTS_PATH,
    )


if __name__ == "__main__":
    main()
