"""Generate baseline and agentic RAG answers for LLM evaluation.

Run from the project root:

    uv run python -m evaluation.llm.generate_answers
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.agentic_rag import AgenticRAG
from app.base_rag import BaseRAG
from app.retrieval import EvidenceRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = (
    PROJECT_ROOT / "data/evaluation/questions.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "data/evaluation/llm"
BASELINE_ANSWERS_PATH = OUTPUT_DIR / "baseline_answers.jsonl"
AGENTIC_ANSWERS_PATH = OUTPUT_DIR / "agentic_answers.jsonl"


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


def generate_answers(
    rag: BaseRAG,
    questions: list[dict[str, Any]],
    output_path: Path,
    index_document_sha256: str,
) -> None:
    """Generate pending answers for one RAG workflow."""

    completed_records = (
        load_jsonl(output_path)
        if output_path.exists()
        else []
    )
    completed_versions = {
        record.get("rag_version")
        for record in completed_records
    }
    completed_models = {
        record.get("answer_model")
        for record in completed_records
    }
    completed_index_hashes = {
        record.get("index_document_sha256")
        for record in completed_records
    }

    if completed_records and (
        completed_versions != {rag.rag_version}
        or completed_models != {rag.model_name}
        or completed_index_hashes
        != {index_document_sha256}
    ):
        raise ValueError(
            f"{output_path} contains answers from a different "
            "RAG version, model, or search index. Move or "
            "delete that file before starting a new "
            "evaluation run."
        )

    questions_by_id = {
        record["question_id"]: record
        for record in questions
    }

    if len(questions_by_id) != len(questions):
        raise ValueError("Questions contain duplicate IDs")

    for record in completed_records:
        question_id = record["question_id"]

        if (
            question_id not in questions_by_id
            or record.get("question_hash")
            != questions_by_id[question_id].get("question_hash")
        ):
            raise ValueError(
                f"{output_path} contains an answer for a "
                "different question dataset. Move or delete "
                "that file before continuing."
            )

    completed_ids = {
        record["question_id"]
        for record in completed_records
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        for number, question_record in enumerate(
            questions,
            start=1,
        ):
            question_id = question_record["question_id"]

            if question_id in completed_ids:
                print(
                    f"[{number}/{len(questions)}] "
                    f"Skipped {rag.rag_version} {question_id}"
                )
                continue

            result = rag.answer(question_record["question"])

            output_record = {
                "question_id": question_id,
                "source": question_record["source"],
                "seed_document_id": question_record[
                    "seed_document_id"
                ],
                "reference_answer": question_record[
                    "reference_answer"
                ],
                "reference_contexts": question_record[
                    "reference_contexts"
                ],
                "question_hash": question_record[
                    "question_hash"
                ],
                "rag_version": rag.rag_version,
                "answer_model": rag.model_name,
                "index_document_sha256": (
                    index_document_sha256
                ),
                **result,
            }
            output_record["answer_hash"] = record_sha256(
                output_record
            )
            file.write(
                json.dumps(output_record, ensure_ascii=False)
                + "\n"
            )
            file.flush()

            print(
                f"[{number}/{len(questions)}] "
                f"Wrote {rag.rag_version} {question_id}: "
                f"{len(result['search_queries'])} searches, "
                f"{len(result['contexts'])} cited contexts"
            )

    print(f"Wrote {rag.rag_version} answers to {output_path}")


def main() -> None:
    """Load shared resources once and run both RAG workflows."""

    load_dotenv(PROJECT_ROOT / ".env")
    questions = load_jsonl(QUESTIONS_PATH)

    retriever = EvidenceRetriever.from_defaults()
    index_document_sha256 = (
        retriever.index_document_sha256()
    )
    openai_client = OpenAI()

    baseline_rag = BaseRAG(retriever, openai_client)
    agentic_rag = AgenticRAG(retriever, openai_client)

    print(f"Questions: {len(questions)}")
    generate_answers(
        baseline_rag,
        questions,
        BASELINE_ANSWERS_PATH,
        index_document_sha256,
    )
    generate_answers(
        agentic_rag,
        questions,
        AGENTIC_ANSWERS_PATH,
        index_document_sha256,
    )


if __name__ == "__main__":
    main()
