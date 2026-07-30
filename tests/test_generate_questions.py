from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from evaluation.generate_questions import (
    FAQ_PATH,
    GENERATED_SOURCE_NAMES,
    build_faq_questions,
    finalise_question,
    generate_source_questions,
    load_jsonl,
    sample_documents,
    validate_questions,
)
from ingestion.chunk_documents import prepare_documents


def test_faq_seed_document_ids_exist_before_chunking() -> None:
    documents = prepare_documents()
    document_ids = {
        document["document_id"]
        for document in documents
    }

    faq_records = load_jsonl(FAQ_PATH)
    questions = build_faq_questions(faq_records)

    assert len(questions) == 74
    assert all(
        question["seed_document_id"] in document_ids
        for question in questions
    )
    assert all(
        "::chunk-" not in question["seed_document_id"]
        for question in questions
    )


def test_generated_question_samples_use_ten_documents_per_source() -> None:
    documents = prepare_documents()
    samples = sample_documents(documents)

    assert set(samples) == set(GENERATED_SOURCE_NAMES)
    assert all(
        len(source_documents) == 10
        for source_documents in samples.values()
    )

    selected_documents = [
        document
        for source_documents in samples.values()
        for document in source_documents
    ]

    source_counts = Counter(
        document["source"]
        for document in selected_documents
    )

    assert source_counts == Counter(
        {
            source: 10
            for source in GENERATED_SOURCE_NAMES
        }
    )
    assert len(selected_documents) == 60
    assert len(
        {
            document["document_id"]
            for document in selected_documents
        }
    ) == 60
    assert all(
        "::chunk-" not in document["document_id"]
        for document in selected_documents
    )


def test_complete_question_set_contains_134_valid_questions(
    tmp_path: Path,
) -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.count = 0

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.count += 1
            return SimpleNamespace(
                output_text=f"Generated question {self.count}?"
            )

    documents = prepare_documents()
    samples = sample_documents(documents)
    client = SimpleNamespace(responses=FakeResponses())

    questions = [
        finalise_question(question)
        for question in [
            *build_faq_questions(load_jsonl(FAQ_PATH)),
            *generate_source_questions(
                client,
                samples,
                tmp_path / "checkpoint.jsonl",
            ),
        ]
    ]

    validate_questions(questions, documents)

    assert len(questions) == 134
    assert len({question["question_id"] for question in questions}) == 134
    assert all(question["question_hash"] for question in questions)
