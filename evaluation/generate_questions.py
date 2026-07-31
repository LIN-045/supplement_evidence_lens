"""Build the final 105-question evaluation dataset.

Run from the project root:

    uv run python -m evaluation.generate_questions
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from evaluation.structured_llm import generate_structured_output
from ingestion.chunk_documents import prepare_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT / "data/evaluation/questions.jsonl"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/question_generation_checkpoint.jsonl"
)

QUESTION_MODEL_NAME = "gpt-5.4-mini"
ANSWERABILITY_MODEL_NAME = "gpt-5.4-mini"
RANDOM_SEED = 42
QUESTIONS_PER_SOURCE = 15
MAX_GENERATION_ATTEMPTS = 3

FAQ_SOURCE = "us_nih_ods_faq"
SOURCE_NAMES = {
    "eu_health_claims_register": "EU Register",
    "health_canada_nhpid": "Health Canada",
    "nih_ods": "NIH ODS Professional Fact Sheets",
    "nih_ods_guidance": "NIH ODS Consumer Guidance",
    "us_dri_tables": "US Dietary Reference Intakes",
    "nccih_herbs": "NCCIH Herbs at a Glance",
    FAQ_SOURCE: "NIH ODS Consumer FAQ",
}
EXPECTED_SOURCE_COUNTS = {
    source: QUESTIONS_PER_SOURCE
    for source in SOURCE_NAMES
}
EXPECTED_QUESTION_COUNT = sum(
    EXPECTED_SOURCE_COUNTS.values()
)

QUESTION_GENERATION_INSTRUCTIONS = """
Write one question in the way a real person might type it into a dietary
supplement evidence assistant.

Requirements:
- The supplied official document must contain enough information to answer it.
- Use natural, everyday English rather than formal, academic, database, or
  regulatory wording.
- Keep it concise, but include enough detail to understand it without seeing
  the document.
- Prefer direct phrasing such as "Does...", "Can...", "Is it safe...",
  "How much...", "What are the risks...", or "Is this claim supported..."
  when appropriate. Do not force every question into the same pattern.
- Name the relevant ingredient, supplement, or product when needed.
- Use the ingredient name a consumer would recognise. Omit long chemical,
  botanical, strain, or product identifiers unless they are necessary to
  distinguish the substance.
- Prefer practical questions about whether a claim is true or supported,
  what an ingredient may do, dose information, safety, restrictions, or who
  the information applies to.
- Do not ask whether wording is allowed, authorised, approved, permitted, or
  what its regulatory status is. A consumer is more likely to ask whether a
  claimed benefit is true, supported by evidence, or actually works.
- Do not name or reveal the document's source, publisher, country,
  jurisdiction, regulator, or evidence role. Do not use phrases such as
  "according to", "in the EU", "in Canada", or "NIH says".
- Do not ask about authors, publication dates, section names, database fields,
  or citation numbers.
- Do not assume that a marketing claim is authorised or scientifically proven.
- Do not include a personal medical history, laboratory result, medication
  regimen, or request for diagnosis or personalised treatment.
- Return only the question.
"""
QUESTION_PROMPT_SHA256 = hashlib.sha256(
    QUESTION_GENERATION_INSTRUCTIONS.encode()
).hexdigest()

ANSWERABILITY_INSTRUCTIONS = """
Determine whether the question can be answered completely and accurately using
only the supplied official document.

Mark it answerable only when:
- the document contains the information needed for every meaningful part of
  the question;
- the reference answer can be written without outside knowledge;
- the answer respects the document's evidence boundary. A regulatory claim
  record can establish regulatory support or lack of support, but not prove
  that a biological effect is universally true or false. A regulatory
  monograph can establish recognised or traditional uses, doses, and warnings,
  but does not by itself prove clinical effectiveness.

Reject questions that ask for a specific value, use, risk, comparison, or
conclusion that the document merely links to, suggests looking up elsewhere,
or does not state.

When answerable, write a concise reference answer and copy the shortest passage
from the document that supports it. When not answerable, leave both fields
empty and briefly explain what is missing.
"""
ANSWERABILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "reference_answer": {"type": "string"},
        "supporting_evidence": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "answerable",
        "reference_answer",
        "supporting_evidence",
        "reason",
    ],
    "additionalProperties": False,
}
ANSWERABILITY_PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "instructions": ANSWERABILITY_INSTRUCTIONS,
            "schema": ANSWERABILITY_SCHEMA,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def parse_arguments() -> argparse.Namespace:
    """Parse question-generation controls."""

    parser = argparse.ArgumentParser(
        description="Build the final 105-question evaluation dataset."
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Discard an existing generation checkpoint and "
            "regenerate all 105 source-grounded questions."
        ),
    )
    return parser.parse_args()


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


def finalise_question(
    question: dict[str, Any],
) -> dict[str, Any]:
    """Attach a hash covering the complete question and reference data."""

    return {
        **question,
        "question_hash": record_sha256(question),
    }


def normalise_question_text(question: str) -> str:
    """Return a stable representation for comparison and validation."""

    return " ".join(question.casefold().split())


def clean_generated_question(question: str) -> str:
    """Normalize harmless model formatting around one generated question."""

    question = " ".join(question.strip().split())

    if question.casefold().startswith("question:"):
        question = question.split(":", 1)[1].strip()

    return question


def generated_question_is_valid(
    question: str,
    document: dict[str, Any],
) -> bool:
    """Reject empty questions and direct copies of source prompts."""

    normalized_question = normalise_question_text(question)
    if not normalized_question:
        return False

    disallowed_texts = {
        normalise_question_text(document.get("title", "")),
        normalise_question_text(document.get("question", "")),
    }
    disallowed_texts.discard("")
    if normalized_question in disallowed_texts:
        return False

    source_leakage_patterns = (
        r"\beu\b",
        r"\beuropean union\b",
        r"\bcanada\b",
        r"\bhealth canada\b",
        r"\bnih\b",
        r"\bods\b",
        r"\bnccih\b",
        r"\bevidence role\b",
    )
    if any(
        re.search(pattern, normalized_question)
        for pattern in source_leakage_patterns
    ):
        return False

    regulatory_wording_patterns = (
        r"\bofficially allowed\b",
        r"\b(?:allow|allowed|authorise|authorised|authorize|authorized|"
        r"permit|permitted)\b",
        r"\bregulatory status\b",
    )
    return not any(
        re.search(pattern, normalized_question)
        for pattern in regulatory_wording_patterns
    )


def is_eligible_for_generation(
    document: dict[str, Any],
) -> bool:
    """Return whether a processed document is suitable for generation."""

    section_title = document.get(
        "section_title",
        "",
    ).casefold()
    excluded_sections = {
        "date",
        "references cited",
        "references reviewed",
    }

    if section_title in excluded_sections:
        return False

    if document["source"] == "eu_health_claims_register":
        return bool(document.get("claim_text"))

    if document["source"] == "us_dri_tables":
        return bool(document.get("nutrient"))

    if document["source"] == "health_canada_nhpid":
        section_title = document.get(
            "section_title",
            "",
        ).casefold()
        administrative_sections = (
            "specifications",
            "version history",
            "foreword",
            "example of product facts",
            "drug facts table",
        )
        return (
            len(document.get("content", "").strip()) >= 300
            and not any(
                section in section_title
                for section in administrative_sections
            )
        )

    return len(document.get("content", "").strip()) >= 300


def sampling_group(document: dict[str, Any]) -> str:
    """Return the parent entity used to diversify sampled chunks."""

    source = document["source"]
    parent_fields = {
        "health_canada_nhpid": "monograph_id",
        "nih_ods": "fact_sheet_id",
        "us_dri_tables": "nutrient",
        "nccih_herbs": "herb_id",
    }

    if source == "eu_health_claims_register":
        return normalise_question_text(
            document.get("claim_text", document["title"])
        )

    if source in parent_fields:
        return str(document[parent_fields[source]])

    return str(document["document_id"])


def sample_documents(
    documents: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Select fifteen deterministic, diverse documents per source."""

    candidates = generation_candidates(documents)

    return {
        source: source_documents[:QUESTIONS_PER_SOURCE]
        for source, source_documents in candidates.items()
    }


def generation_candidates(
    documents: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic candidate queues with unique source entities."""

    rng = random.Random(RANDOM_SEED)
    candidates_by_source = {}

    for source in SOURCE_NAMES:
        candidates = [
            document
            for document in documents
            if document["source"] == source
            and is_eligible_for_generation(document)
        ]
        candidates.sort(
            key=lambda document: document["document_id"]
        )
        rng.shuffle(candidates)

        selected = []
        selected_groups = set()

        for document in candidates:
            group = sampling_group(document)

            if group in selected_groups:
                continue

            selected.append(document)
            selected_groups.add(group)

        if len(selected) < QUESTIONS_PER_SOURCE:
            raise ValueError(
                f"Not enough eligible documents for {source}: "
                f"{len(selected)}"
            )

        candidates_by_source[source] = selected

    return candidates_by_source


def evidence_boundary(document: dict[str, Any]) -> str | None:
    """Return private generation guidance for evidence-limited sources."""

    if document["evidence_role"] == "regulatory_claim":
        return (
            "Ask whether the stated benefit is supported by the supplied "
            "official evidence. Do not ask for a definitive biological or "
            "clinical verdict that this regulatory record cannot establish."
        )

    if document["evidence_role"] == "regulatory_monograph":
        return (
            "Treat listed uses as recognised or traditional monograph uses, "
            "not automatic proof of clinical effectiveness. Dose, identity, "
            "and warning questions may be asked directly when stated."
        )

    return None


def generation_input(document: dict[str, Any]) -> str:
    """Build the complete document-specific question-generation input."""

    prompt_parts = []
    if document["source"] != FAQ_SOURCE:
        prompt_parts.append(f"Subject: {document['title']}")
    boundary = evidence_boundary(document)
    if boundary:
        prompt_parts.append(f"Evidence boundary: {boundary}")
    prompt_parts.append(f"Official document:\n{document['content']}")
    return "\n".join(prompt_parts)


def answerability_input(
    document: dict[str, Any],
    question: str,
) -> str:
    """Build the complete document-specific answerability input."""

    return json.dumps(
        {
            "question": question,
            "evidence_role": document["evidence_role"],
            "official_document": document["content"],
        },
        ensure_ascii=False,
    )


def generate_question(
    client: OpenAI,
    document: dict[str, Any],
    *,
    disallowed_questions: set[str] | None = None,
) -> str:
    """Generate one source-grounded consumer question."""

    disallowed_questions = disallowed_questions or set()
    prompt = generation_input(document)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        response = client.responses.create(
            model=QUESTION_MODEL_NAME,
            instructions=QUESTION_GENERATION_INSTRUCTIONS,
            input=prompt,
        )
        question = clean_generated_question(
            response.output_text
        )

        if (
            generated_question_is_valid(question, document)
            and normalise_question_text(question)
            not in disallowed_questions
        ):
            return question

    raise ValueError(
        f"Could not generate a new, valid question for "
        f"{document['document_id']} after "
        f"{MAX_GENERATION_ATTEMPTS} attempts"
    )


def validate_answerability(
    client: OpenAI,
    document: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Validate one question and produce its grounded reference answer."""

    result = generate_structured_output(
        client,
        model=ANSWERABILITY_MODEL_NAME,
        instructions=ANSWERABILITY_INSTRUCTIONS,
        input_text=answerability_input(
            document,
            question,
        ),
        schema_name="question_answerability",
        schema=ANSWERABILITY_SCHEMA,
    )

    if result["answerable"] and (
        not result["reference_answer"].strip()
        or not result["supporting_evidence"].strip()
    ):
        raise ValueError(
            "Answerability validator accepted a question without "
            "a reference answer and supporting evidence"
        )

    return result


def generate_answerable_question(
    client: OpenAI,
    document: dict[str, Any],
    *,
    disallowed_questions: set[str],
) -> dict[str, str]:
    """Generate and semantically validate a question for one document."""

    rejection_reason = "no valid question generated"

    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            question = generate_question(
                client,
                document,
                disallowed_questions=disallowed_questions,
            )
        except ValueError as error:
            rejection_reason = str(error)
            break

        validation = validate_answerability(
            client,
            document,
            question,
        )
        if validation["answerable"]:
            return {
                "question": question,
                "reference_answer": validation[
                    "reference_answer"
                ].strip(),
                "supporting_evidence": validation[
                    "supporting_evidence"
                ].strip(),
            }

        rejection_reason = validation["reason"].strip()

    raise ValueError(rejection_reason)


def generate_source_questions(
    client: OpenAI,
    candidates_by_source: dict[str, list[dict[str, Any]]],
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    """Generate or resume the 105 source-grounded questions."""

    candidate_documents = [
        document
        for documents in candidates_by_source.values()
        for document in documents
    ]
    candidates_by_id = {
        document["document_id"]: document
        for document in candidate_documents
    }
    completed_records = (
        load_jsonl(checkpoint_path)
        if checkpoint_path.exists()
        else []
    )
    completed_by_seed_id = {}
    normalized_questions: set[str] = set()

    for record in completed_records:
        seed_document_id = record.get("seed_document_id")

        if (
            record.get("question_model") != QUESTION_MODEL_NAME
            or record.get("answerability_model")
            != ANSWERABILITY_MODEL_NAME
            or record.get("random_seed") != RANDOM_SEED
            or record.get("prompt_sha256")
            != QUESTION_PROMPT_SHA256
            or record.get("answerability_prompt_sha256")
            != ANSWERABILITY_PROMPT_SHA256
        ):
            raise ValueError(
                f"{checkpoint_path} was created with a "
                "different generator configuration. Run with "
                "--restart to generate a new question set."
            )

        if (
            seed_document_id not in candidates_by_id
            or seed_document_id in completed_by_seed_id
        ):
            raise ValueError(
                f"{checkpoint_path} contains unexpected or "
                "duplicate seed document IDs."
            )

        document = candidates_by_id[seed_document_id]
        content_sha256 = hashlib.sha256(
            document["content"].encode()
        ).hexdigest()

        generation_input_sha256 = hashlib.sha256(
            generation_input(document).encode()
        ).hexdigest()

        if (
            record.get("seed_content_sha256") != content_sha256
            or record.get("generation_input_sha256")
            != generation_input_sha256
        ):
            raise ValueError(
                f"Processed content changed for "
                f"{seed_document_id}. Run with --restart."
            )

        normalized_question = normalise_question_text(
            record.get("question", "")
        )
        if (
            not generated_question_is_valid(
                record.get("question", ""),
                document,
            )
            or normalized_question in normalized_questions
            or not record.get("reference_answer", "").strip()
            or not record.get("supporting_evidence", "").strip()
        ):
            raise ValueError(
                f"{checkpoint_path} contains an invalid or "
                "duplicate generated question. Run with "
                "--restart."
            )

        normalized_questions.add(normalized_question)
        completed_by_seed_id[seed_document_id] = record

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    selected_documents = []

    for source, candidates in candidates_by_source.items():
        completed_for_source = [
            record
            for record in completed_records
            if candidates_by_id[
                record["seed_document_id"]
            ]["source"] == source
        ]
        if len(completed_for_source) > QUESTIONS_PER_SOURCE:
            raise ValueError(
                f"{checkpoint_path} contains too many questions "
                f"for {source}"
            )
        selected_documents.extend(
            candidates_by_id[record["seed_document_id"]]
            for record in completed_for_source
        )

        selected_ids = {
            record["seed_document_id"]
            for record in completed_for_source
        }
        for document in candidates:
            if len(completed_for_source) == QUESTIONS_PER_SOURCE:
                break
            if document["document_id"] in selected_ids:
                continue

            try:
                generated = generate_answerable_question(
                    client,
                    document,
                    disallowed_questions=normalized_questions,
                )
            except ValueError as error:
                print(
                    f"Rejected {document['document_id']}: {error}"
                )
                continue

            checkpoint_record = {
                "question_model": QUESTION_MODEL_NAME,
                "answerability_model": ANSWERABILITY_MODEL_NAME,
                "random_seed": RANDOM_SEED,
                "prompt_sha256": QUESTION_PROMPT_SHA256,
                "answerability_prompt_sha256": (
                    ANSWERABILITY_PROMPT_SHA256
                ),
                "seed_content_sha256": hashlib.sha256(
                    document["content"].encode()
                ).hexdigest(),
                "generation_input_sha256": hashlib.sha256(
                    generation_input(document).encode()
                ).hexdigest(),
                "seed_document_id": document["document_id"],
                **generated,
            }

            with checkpoint_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        checkpoint_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                file.flush()

            completed_records.append(checkpoint_record)
            completed_for_source.append(checkpoint_record)
            completed_by_seed_id[
                document["document_id"]
            ] = checkpoint_record
            selected_documents.append(document)
            normalized_questions.add(
                normalise_question_text(generated["question"])
            )
            print(
                f"[{len(selected_documents)}/"
                f"{EXPECTED_QUESTION_COUNT}] "
                f"{source}: {generated['question']}"
            )

        if len(completed_for_source) != QUESTIONS_PER_SOURCE:
            raise ValueError(
                f"Could not generate {QUESTIONS_PER_SOURCE} "
                f"answerable questions for {source}"
            )

    return [
        {
            "question_id": f"source-{number:03d}",
            "question": completed_by_seed_id[
                document["document_id"]
            ]["question"],
            "question_group": "document_generated",
            "seed_document_id": document["document_id"],
            "source": document["source"],
            "title": document["title"],
            "source_url": document["source_url"],
            "reference_answer": completed_by_seed_id[
                document["document_id"]
            ]["reference_answer"],
            "supporting_evidence": completed_by_seed_id[
                document["document_id"]
            ]["supporting_evidence"],
            "reference_contexts": [
                {
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "jurisdiction": document["jurisdiction"],
                    "evidence_role": document["evidence_role"],
                    "source_url": document["source_url"],
                    "content": document["content"],
                }
            ],
            "question_model": QUESTION_MODEL_NAME,
            "question_prompt_sha256": QUESTION_PROMPT_SHA256,
            "answerability_model": ANSWERABILITY_MODEL_NAME,
            "answerability_prompt_sha256": (
                ANSWERABILITY_PROMPT_SHA256
            ),
            "random_seed": RANDOM_SEED,
        }
        for number, document in enumerate(
            selected_documents,
            start=1,
        )
    ]


def validate_questions(
    questions: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> None:
    """Validate final counts, IDs, and source documents."""

    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUESTION_COUNT} questions, "
            f"found {len(questions)}"
        )

    question_ids = [
        question["question_id"]
        for question in questions
    ]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Question IDs must be unique")

    normalized_questions = [
        normalise_question_text(question["question"])
        for question in questions
    ]
    if (
        not all(normalized_questions)
        or len(set(normalized_questions))
        != len(normalized_questions)
    ):
        raise ValueError(
            "Questions must be non-empty and unique after normalization"
        )

    documents_by_id = {
        document["document_id"]: document
        for document in documents
    }

    for question in questions:
        stored_hash = question.get("question_hash")
        unhashed_question = {
            key: value
            for key, value in question.items()
            if key != "question_hash"
        }

        if stored_hash != record_sha256(unhashed_question):
            raise ValueError(
                f"Invalid question hash for "
                f"{question['question_id']}"
            )

        if (
            not question.get("reference_answer", "").strip()
            or not question.get("supporting_evidence", "").strip()
            or question.get("answerability_model")
            != ANSWERABILITY_MODEL_NAME
            or question.get("answerability_prompt_sha256")
            != ANSWERABILITY_PROMPT_SHA256
        ):
            raise ValueError(
                f"{question['question_id']} is missing a valid "
                "answerability result"
            )

        document = documents_by_id.get(
            question["seed_document_id"]
        )
        if (
            document is not None
            and not generated_question_is_valid(
                question["question"],
                document,
            )
        ):
            raise ValueError(
                f"{question['question_id']} copies its source "
                "question or title"
            )

    source_counts = Counter(
        question["source"]
        for question in questions
    )
    if source_counts != Counter(EXPECTED_SOURCE_COUNTS):
        raise ValueError(
            f"Unexpected source distribution: {source_counts}"
        )

    source_document_ids = {
        document["document_id"]
        for document in documents
    }
    missing_seed_ids = sorted(
        {
            question["seed_document_id"]
            for question in questions
        }
        - source_document_ids
    )
    if missing_seed_ids:
        raise ValueError(
            f"Seed documents are missing from processed data: "
            f"{missing_seed_ids}"
        )


def write_questions(
    questions: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write the validated question set."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(
                json.dumps(question, ensure_ascii=False)
                + "\n"
            )


def main() -> None:
    """Build and validate the final evaluation question set."""

    arguments = parse_arguments()
    load_dotenv(PROJECT_ROOT / ".env")
    documents = prepare_documents()
    candidates_by_source = generation_candidates(documents)

    if arguments.restart:
        CHECKPOINT_PATH.unlink(missing_ok=True)

    client = OpenAI()

    questions = [
        finalise_question(question)
        for question in generate_source_questions(
            client,
            candidates_by_source,
            CHECKPOINT_PATH,
        )
    ]
    validate_questions(questions, documents)
    write_questions(questions, OUTPUT_PATH)
    CHECKPOINT_PATH.unlink(missing_ok=True)

    print(
        f"Wrote {len(questions)} validated questions "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
