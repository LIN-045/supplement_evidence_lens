import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.generate_questions import (
    EXPECTED_QUESTION_COUNT,
    EXPECTED_SOURCE_COUNTS,
    FAQ_SOURCE,
    QUESTIONS_PER_SOURCE,
    RANDOM_SEED,
    SOURCE_NAMES,
    clean_generated_question,
    finalise_question,
    generate_answerable_question,
    generate_question,
    generate_source_questions,
    generation_candidates,
    generated_question_is_valid,
    is_eligible_for_generation,
    sampling_group,
    validate_questions,
)
from ingestion.chunk_documents import prepare_documents


def test_question_samples_use_fifteen_documents_per_source() -> None:
    documents = prepare_documents()
    samples = {
        source: candidates[:QUESTIONS_PER_SOURCE]
        for source, candidates in generation_candidates(
            documents
        ).items()
    }

    assert set(samples) == set(SOURCE_NAMES)
    assert all(
        len(source_documents) == QUESTIONS_PER_SOURCE
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
        EXPECTED_SOURCE_COUNTS
    )
    assert len(selected_documents) == EXPECTED_QUESTION_COUNT
    assert len(
        {
            document["document_id"]
            for document in selected_documents
        }
    ) == EXPECTED_QUESTION_COUNT
    assert all(
        "::chunk-" not in document["document_id"]
        for document in selected_documents
    )


def test_sampling_is_reproducible_with_the_fixed_seed() -> None:
    documents = prepare_documents()

    first_samples = generation_candidates(documents)
    second_samples = generation_candidates(documents)

    assert RANDOM_SEED == 42
    assert {
        source: [
            document["document_id"]
            for document in source_documents
        ]
        for source, source_documents in first_samples.items()
    } == {
        source: [
            document["document_id"]
            for document in source_documents
        ]
        for source, source_documents in second_samples.items()
    }


def test_faq_generation_hides_the_original_question() -> None:
    faq_document = next(
        document
        for document in prepare_documents()
        if document["source"] == FAQ_SOURCE
    )

    class FakeResponses:
        def __init__(self) -> None:
            self.request: dict | None = None

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.request = kwargs
            return SimpleNamespace(
                output_text="Could supplements cause unwanted effects?"
            )

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)

    generate_question(client, faq_document)

    assert responses.request is not None
    prompt = str(responses.request["input"])
    assert faq_document["question"] not in prompt
    assert f"Title: {faq_document['title']}" not in prompt
    assert faq_document["content"] in prompt


def test_generation_prompt_does_not_reveal_source_metadata() -> None:
    document = next(
        document
        for document in prepare_documents()
        if document["source"] == "eu_health_claims_register"
    )

    class FakeResponses:
        def __init__(self) -> None:
            self.request: dict | None = None

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.request = kwargs
            return SimpleNamespace(
                output_text="Does this ingredient really support heart health?"
            )

    responses = FakeResponses()
    generate_question(
        SimpleNamespace(responses=responses),
        document,
    )

    assert responses.request is not None
    prompt = str(responses.request["input"])
    assert "Source:" not in prompt
    assert "Jurisdiction:" not in prompt
    assert "Evidence role:" not in prompt


def test_source_leakage_and_regulatory_wording_are_rejected() -> None:
    document = next(
        document
        for document in prepare_documents()
        if document["source"] == "eu_health_claims_register"
    )

    assert not generated_question_is_valid(
        "Is this claim authorised in the EU?",
        document,
    )
    assert not generated_question_is_valid(
        "Does Health Canada allow this claim?",
        document,
    )
    assert generated_question_is_valid(
        "Does this ingredient really support heart health?",
        document,
    )


def test_health_canada_sampling_uses_consumer_relevant_sections() -> None:
    documents = prepare_documents()
    samples = generation_candidates(documents)[
        "health_canada_nhpid"
    ][:QUESTIONS_PER_SOURCE]

    assert all(
        is_eligible_for_generation(document)
        for document in samples
    )
    assert all(
        not any(
            excluded in document["section_title"].casefold()
            for excluded in (
                "specifications",
                "version history",
                "foreword",
                "example of product facts",
                "drug facts table",
            )
        )
        for document in samples
    )

    dosage_form_document = next(
        document
        for document in documents
        if (
            document["source"] == "health_canada_nhpid"
            and "dosage form"
            in document.get("section_title", "").casefold()
            and len(document.get("content", "").strip()) >= 300
        )
    )
    assert is_eligible_for_generation(dosage_form_document)


def test_eu_sampling_groups_duplicate_claims_together() -> None:
    documents = [
        document
        for document in prepare_documents()
        if document["source"] == "eu_health_claims_register"
    ]
    olive_documents = [
        document
        for document in documents
        if document["document_id"] in {"POL-HC-7682", "POL-HC-7684"}
    ]

    assert len(olive_documents) == 2
    assert (
        sampling_group(olive_documents[0])
        == sampling_group(olive_documents[1])
    )


def test_generated_question_formatting_is_cleaned() -> None:
    assert (
        clean_generated_question(
            "  Question:  Can zinc cause side effects?  \n"
        )
        == "Can zinc cause side effects?"
    )


def test_faq_original_question_is_rejected() -> None:
    faq_document = next(
        document
        for document in prepare_documents()
        if document["source"] == FAQ_SOURCE
    )

    class FakeResponses:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                output_text=faq_document["question"]
            )

    client = SimpleNamespace(responses=FakeResponses())

    with pytest.raises(
        ValueError,
        match="Could not generate a new, valid question",
    ):
        generate_question(client, faq_document)


def test_complete_question_set_contains_105_valid_questions(
    tmp_path: Path,
) -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.count = 0

        def create(self, **kwargs: object) -> SimpleNamespace:
            if "text" in kwargs:
                prompt = json.loads(str(kwargs["input"]))
                document = prompt["official_document"]
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "answerable": True,
                            "reference_answer": (
                                "The supplied document answers "
                                "the question."
                            ),
                            "supporting_evidence": document[:80],
                            "reason": "Directly supported.",
                        }
                    )
                )

            self.count += 1
            return SimpleNamespace(
                output_text=f"Generated question {self.count}?"
            )

    documents = prepare_documents()
    candidates = generation_candidates(documents)
    client = SimpleNamespace(responses=FakeResponses())

    questions = [
        finalise_question(question)
        for question in generate_source_questions(
            client,
            candidates,
            tmp_path / "checkpoint.jsonl",
        )
    ]

    validate_questions(questions, documents)

    assert len(questions) == EXPECTED_QUESTION_COUNT
    assert len(
        {question["question_id"] for question in questions}
    ) == EXPECTED_QUESTION_COUNT
    assert all(question["question_hash"] for question in questions)
    assert all(
        question["reference_contexts"][0]["evidence_role"]
        for question in questions
    )
    assert all(
        question["reference_answer"]
        and question["supporting_evidence"]
        and question["answerability_prompt_sha256"]
        for question in questions
    )

    faq_questions = [
        question
        for question in questions
        if question["source"] == FAQ_SOURCE
    ]
    assert len(faq_questions) == QUESTIONS_PER_SOURCE
    assert all(
        question["reference_answer"]
        for question in faq_questions
    )


def test_unanswerable_question_is_rejected() -> None:
    document = next(iter(prepare_documents()))

    class FakeResponses:
        def create(self, **kwargs: object) -> SimpleNamespace:
            if "text" in kwargs:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "answerable": False,
                            "reference_answer": "",
                            "supporting_evidence": "",
                            "reason": "The requested value is missing.",
                        }
                    )
                )
            return SimpleNamespace(
                output_text="What exact value should I take?"
            )

    with pytest.raises(
        ValueError,
        match="requested value is missing",
    ):
        generate_answerable_question(
            SimpleNamespace(responses=FakeResponses()),
            document,
            disallowed_questions=set(),
        )
