import pytest

from ingestion.sources.us_ods_faq import (
    DEFAULT_RAW_PATH,
    extract_records,
)


def test_identical_duplicate_questions_are_merged() -> None:
    html = b"""
    <article>
        <h2 id="section-one">Section One</h2>
        <p><strong>Q. What does vitamin C do?</strong></p>
        <p>Vitamin C acts as an antioxidant.</p>

        <h2 id="section-two">Section Two</h2>
        <p><strong>Q. What does vitamin C do?</strong></p>
        <p>Vitamin C acts as an antioxidant.</p>
    </article>
    """

    records = extract_records(
        html,
        source_sha256="test-sha256",
        retrieved_at="2026-07-30T00:00:00+00:00",
    )

    assert len(records) == 1

    record = records[0]
    assert record["question"] == "What does vitamin C do?"
    assert record["text"] == "Vitamin C acts as an antioxidant."
    assert record["duplicate_count"] == 2
    assert record["sections"] == ["Section One", "Section Two"]



def test_duplicate_question_with_different_answers_fails() -> None:
    html = b"""
    <article>
        <h2>Section One</h2>
        <p><strong>Q. What does vitamin C do?</strong></p>
        <p>Vitamin C acts as an antioxidant.</p>

        <h2>Section Two</h2>
        <p><strong>Q. What does vitamin C do?</strong></p>
        <p>Vitamin C prevents every illness.</p>
    </article>
    """

    with pytest.raises(
        RuntimeError,
        match="Duplicate FAQ question has different answers",
    ):
        extract_records(
            html,
            source_sha256="test-sha256",
            retrieved_at="2026-07-30T00:00:00+00:00",
        )

    

def test_current_raw_faq_has_expected_record_counts() -> None:
    html = DEFAULT_RAW_PATH.read_bytes()

    records = extract_records(
        html,
        source_sha256="test-sha256",
        retrieved_at="2026-07-30T00:00:00+00:00",
    )

    duplicate_occurrences = sum(
        record["duplicate_count"] - 1
        for record in records
    )

    assert len(records) == 74
    assert duplicate_occurrences == 19
    assert len({record["document_id"] for record in records}) == 74
    assert len({record["question"] for record in records}) == 74