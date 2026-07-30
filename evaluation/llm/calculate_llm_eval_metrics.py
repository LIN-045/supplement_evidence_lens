"""Calculate metrics for baseline and agentic RAG answer judgments.

Run from the project root:

    uv run python -m evaluation.llm.calculate_llm_eval_metrics
"""

import json
import hashlib
from pathlib import Path
from typing import Any

from evaluation.llm.judge_answers import METRIC_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/evaluation/llm"

BASELINE_ANSWERS_PATH = DATA_DIR / "baseline_answers.jsonl"
AGENTIC_ANSWERS_PATH = DATA_DIR / "agentic_answers.jsonl"
BASELINE_JUDGMENTS_PATH = (
    DATA_DIR / "baseline_judgments.jsonl"
)
AGENTIC_JUDGMENTS_PATH = (
    DATA_DIR / "agentic_judgments.jsonl"
)
OUTPUT_PATH = DATA_DIR / "llm_eval_metrics.json"


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


def index_by_question_id(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index records and reject duplicate question IDs."""

    indexed = {
        record["question_id"]: record
        for record in records
    }

    if len(indexed) != len(records):
        raise ValueError("Duplicate question IDs found")

    return indexed


def summarize_judgments(
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate average scores and perfect-answer count."""

    if not judgments:
        raise ValueError("Cannot summarize an empty judgment set")

    return {
        "question_count": len(judgments),
        "average_scores": {
            metric: sum(
                record[metric]["score"]
                for record in judgments
            )
            / len(judgments)
            for metric in METRIC_NAMES
        },
        "perfect_answers": sum(
            all(
                record[metric]["score"] == 5
                for metric in METRIC_NAMES
            )
            for record in judgments
        ),
    }


def summarize_by_source(
    judgments: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calculate judgment summaries for each evidence source."""

    sources = sorted(
        {record["source"] for record in judgments}
    )
    return {
        source: summarize_judgments(
            [
                record
                for record in judgments
                if record["source"] == source
            ]
        )
        for source in sources
    }


def summarize_trajectory(
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize searches and cited contexts used to form answers."""

    if not answers:
        raise ValueError("Cannot summarize an empty answer set")

    search_counts = [
        len(record["search_queries"])
        for record in answers
    ]
    context_counts = [
        len(record["contexts"])
        for record in answers
    ]

    return {
        "average_searches": (
            sum(search_counts) / len(search_counts)
        ),
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
    baseline_judgments: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Compare paired scores for every metric and their total."""

    agentic_by_id = index_by_question_id(
        agentic_judgments
    )
    baseline_by_id = index_by_question_id(
        baseline_judgments
    )

    if set(agentic_by_id) != set(baseline_by_id):
        raise ValueError(
            "Agentic and baseline judgments contain "
            "different question IDs"
        )

    def compare_scores(
        agentic_score: int,
        baseline_score: int,
        counts: dict[str, int],
    ) -> None:
        if agentic_score > baseline_score:
            counts["agentic_wins"] += 1
        elif baseline_score > agentic_score:
            counts["baseline_wins"] += 1
        else:
            counts["ties"] += 1

    comparisons: dict[str, dict[str, int]] = {}

    for metric in METRIC_NAMES:
        counts = {
            "agentic_wins": 0,
            "baseline_wins": 0,
            "ties": 0,
        }

        for question_id in agentic_by_id:
            compare_scores(
                agentic_by_id[question_id][metric]["score"],
                baseline_by_id[question_id][metric]["score"],
                counts,
            )

        comparisons[metric] = counts

    combined_counts = {
        "agentic_wins": 0,
        "baseline_wins": 0,
        "ties": 0,
    }

    for question_id in agentic_by_id:
        agentic_total = sum(
            agentic_by_id[question_id][metric]["score"]
            for metric in METRIC_NAMES
        )
        baseline_total = sum(
            baseline_by_id[question_id][metric]["score"]
            for metric in METRIC_NAMES
        )
        compare_scores(
            agentic_total,
            baseline_total,
            combined_counts,
        )

    comparisons["combined_score"] = combined_counts
    return comparisons


def compare_by_source(
    agentic_judgments: list[dict[str, Any]],
    baseline_judgments: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Calculate paired comparisons separately for each source."""

    sources = sorted(
        {record["source"] for record in agentic_judgments}
        | {record["source"] for record in baseline_judgments}
    )
    return {
        source: compare_judgments(
            [
                record
                for record in agentic_judgments
                if record["source"] == source
            ],
            [
                record
                for record in baseline_judgments
                if record["source"] == source
            ],
        )
        for source in sources
    }


def validate_answer_coverage(
    answers: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    workflow: str,
) -> None:
    """Require one judgment for every generated answer."""

    answers_by_id = index_by_question_id(answers)
    judgments_by_id = index_by_question_id(judgments)
    answer_ids = set(answers_by_id)
    judgment_ids = set(judgments_by_id)

    if answer_ids != judgment_ids:
        raise ValueError(
            f"{workflow} answers and judgments contain "
            "different question IDs"
        )

    for question_id, answer in answers_by_id.items():
        unhashed_answer = {
            key: value
            for key, value in answer.items()
            if key != "answer_hash"
        }
        if answer.get("answer_hash") != record_sha256(
            unhashed_answer
        ):
            raise ValueError(
                f"{workflow} has an invalid answer hash for "
                f"{question_id}"
            )
        if (
            judgments_by_id[question_id].get("answer_hash")
            != answer["answer_hash"]
        ):
            raise ValueError(
                f"{workflow} judgment does not match the "
                f"answer for {question_id}"
            )


def main() -> None:
    """Load evaluation records and write the final metrics file."""

    baseline_answers = load_jsonl(BASELINE_ANSWERS_PATH)
    agentic_answers = load_jsonl(AGENTIC_ANSWERS_PATH)
    baseline_judgments = load_jsonl(
        BASELINE_JUDGMENTS_PATH
    )
    agentic_judgments = load_jsonl(
        AGENTIC_JUDGMENTS_PATH
    )

    validate_answer_coverage(
        baseline_answers,
        baseline_judgments,
        "Baseline",
    )
    validate_answer_coverage(
        agentic_answers,
        agentic_judgments,
        "Agentic",
    )

    index_hashes = {
        record.get("index_document_sha256")
        for record in [*baseline_answers, *agentic_answers]
    }
    if len(index_hashes) != 1 or None in index_hashes:
        raise ValueError(
            "Answers contain mixed or missing search index hashes"
        )

    results = {
        "index_document_sha256": next(iter(index_hashes)),
        "baseline_rag": {
            "rag_version": baseline_answers[0]["rag_version"],
            "answer_model": baseline_answers[0]["answer_model"],
            "judge_version": baseline_judgments[0][
                "judge_version"
            ],
            "judge_model": baseline_judgments[0]["judge_model"],
            "overall": summarize_judgments(
                baseline_judgments
            ),
            "by_source": summarize_by_source(
                baseline_judgments
            ),
            "trajectory": summarize_trajectory(
                baseline_answers
            ),
        },
        "agentic_rag": {
            "rag_version": agentic_answers[0]["rag_version"],
            "answer_model": agentic_answers[0]["answer_model"],
            "judge_version": agentic_judgments[0][
                "judge_version"
            ],
            "judge_model": agentic_judgments[0]["judge_model"],
            "overall": summarize_judgments(
                agentic_judgments
            ),
            "by_source": summarize_by_source(
                agentic_judgments
            ),
            "trajectory": summarize_trajectory(
                agentic_answers
            ),
        },
        "paired_comparison": {
            "overall": compare_judgments(
                agentic_judgments,
                baseline_judgments,
            ),
            "by_source": compare_by_source(
                agentic_judgments,
                baseline_judgments,
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote LLM evaluation metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
