from pathlib import Path

import pytest

from evaluation.llm.generate_answers import generate_answers


class FakeRAG:
    rag_version = "baseline_rag_v1"
    model_name = "test-model"

    def answer(self, question: str) -> dict:
        return {
            "question": question,
            "answer": "Test answer.",
            "search_queries": [question],
            "contexts": [],
        }


def test_answers_cannot_resume_with_a_different_index(
    tmp_path: Path,
) -> None:
    questions = [
        {
            "question_id": "question-1",
            "question": "What does vitamin C do?",
            "source": "test_source",
            "seed_document_id": "document-1",
            "reference_answer": None,
            "reference_contexts": [],
            "question_hash": "question-hash",
        }
    ]
    output_path = tmp_path / "answers.jsonl"
    rag = FakeRAG()

    generate_answers(
        rag,
        questions,
        output_path,
        index_document_sha256="index-a",
    )

    with pytest.raises(
        ValueError,
        match="different RAG version, model, or search index",
    ):
        generate_answers(
            rag,
            questions,
            output_path,
            index_document_sha256="index-b",
        )


def test_answers_cannot_resume_with_changed_question_content(
    tmp_path: Path,
) -> None:
    original_questions = [
        {
            "question_id": "question-1",
            "question": "What does vitamin C do?",
            "source": "test_source",
            "seed_document_id": "document-1",
            "reference_answer": None,
            "reference_contexts": [],
            "question_hash": "original-question-hash",
        }
    ]
    output_path = tmp_path / "answers.jsonl"
    rag = FakeRAG()

    generate_answers(
        rag,
        original_questions,
        output_path,
        index_document_sha256="same-index",
    )

    changed_questions = [
        {
            **original_questions[0],
            "question": "Can vitamin C prevent colds?",
            "question_hash": "changed-question-hash",
        }
    ]

    with pytest.raises(
        ValueError,
        match="different question dataset",
    ):
        generate_answers(
            rag,
            changed_questions,
            output_path,
            index_document_sha256="same-index",
        )


def test_answers_cannot_resume_with_a_different_rag_version(
    tmp_path: Path,
) -> None:
    questions = [
        {
            "question_id": "question-1",
            "question": "What does vitamin C do?",
            "source": "test_source",
            "seed_document_id": "document-1",
            "reference_answer": None,
            "reference_contexts": [],
            "question_hash": "question-hash",
        }
    ]
    output_path = tmp_path / "answers.jsonl"

    original_rag = FakeRAG()
    generate_answers(
        original_rag,
        questions,
        output_path,
        index_document_sha256="same-index",
    )

    updated_rag = FakeRAG()
    updated_rag.rag_version = "baseline_rag_v2"

    with pytest.raises(
        ValueError,
        match="different RAG version, model, or search index",
    ):
        generate_answers(
            updated_rag,
            questions,
            output_path,
            index_document_sha256="same-index",
        )
