"""Build the final 134-question evaluation dataset.

Run from the project root:

    uv run python -m evaluation.generate_questions
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ingestion.chunk_documents import prepare_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAQ_PATH = (
    PROJECT_ROOT
    / "data/processed/sources/us_ods_faq_answers.jsonl"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data/evaluation/questions.jsonl"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/question_generation_checkpoint.jsonl"
)

QUESTION_MODEL_NAME = "gpt-5.4-mini"
QUESTION_GENERATOR_VERSION = "question_generator_v1"
RANDOM_SEED = 42
QUESTIONS_PER_GENERATED_SOURCE = 10

FAQ_SOURCE = "us_nih_ods_faq"
GENERATED_SOURCE_NAMES = {
    "eu_health_claims_register": "EU Register",
    "health_canada_nhpid": "Health Canada",
    "nih_ods": "NIH ODS Professional Fact Sheets",
    "nih_ods_guidance": "NIH ODS Consumer Guidance",
    "us_dri_tables": "US Dietary Reference Intakes",
    "nccih_herbs": "NCCIH Herbs at a Glance",
}
EXPECTED_SOURCE_COUNTS = {
    FAQ_SOURCE: 74,
    **{
        source: QUESTIONS_PER_GENERATED_SOURCE
        for source in GENERATED_SOURCE_NAMES
    },
}
EXPECTED_QUESTION_COUNT = sum(
    EXPECTED_SOURCE_COUNTS.values()
)

QUESTION_GENERATION_INSTRUCTIONS = """
Generate one realistic question that an ordinary consumer might ask a dietary
supplement evidence assistant after seeing a product label, advertisement, or
supplement ingredient.

Requirements:
- The question must be answerable from the supplied excerpt.
- Write in clear, everyday English and avoid database or regulatory jargon.
- Make the question understandable without showing the excerpt.
- Name the relevant ingredient, supplement, or product when needed.
- Use the ingredient name a consumer would recognise. Omit long chemical,
  botanical, strain, or product identifiers unless they are necessary to
  distinguish the substance.
- Prefer practical questions about whether a claim is supported or allowed,
  what an ingredient may do, dose information, safety, restrictions, or who
  the information applies to.
- Regulatory questions may mention the relevant jurisdiction, but avoid
  formulaic phrases such as "What is the regulatory status of the health
  claim regarding".
- Do not ask about authors, publication dates, section names, database fields,
  or citation numbers.
- Do not assume that a marketing claim is authorised or scientifically proven.
- Return only the question.
"""
QUESTION_PROMPT_SHA256 = hashlib.sha256(
    QUESTION_GENERATION_INSTRUCTIONS.encode()
).hexdigest()


def parse_arguments() -> argparse.Namespace:
    """Parse question-generation controls."""

    parser = argparse.ArgumentParser(
        description="Build the final 134-question evaluation dataset."
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Discard an existing generation checkpoint and "
            "regenerate all 60 source-grounded questions."
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


def is_eligible_for_generation(
    document: dict[str, Any],
) -> bool:
    """Return whether a chunk is suitable for question generation."""

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

    if source in parent_fields:
        return str(document[parent_fields[source]])

    return str(document["document_id"])


def sample_documents(
    documents: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Select ten deterministic, diverse source documents."""

    rng = random.Random(RANDOM_SEED)
    samples = {}

    for source in GENERATED_SOURCE_NAMES:
        candidates = [
            document
            for document in documents
            if document["source"] == source
            and is_eligible_for_generation(document)
        ]
        rng.shuffle(candidates)

        selected = []
        selected_groups = set()

        for document in candidates:
            group = sampling_group(document)

            if group in selected_groups:
                continue

            selected.append(document)
            selected_groups.add(group)

            if (
                len(selected)
                == QUESTIONS_PER_GENERATED_SOURCE
            ):
                break

        if len(selected) < QUESTIONS_PER_GENERATED_SOURCE:
            raise ValueError(
                f"Not enough eligible documents for {source}: "
                f"{len(selected)}"
            )

        samples[source] = selected

    return samples


def build_faq_questions(
    faq_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy the 74 original NIH ODS FAQ questions."""

    return [
        {
            "question_id": f"faq-{number:03d}",
            "question": record["question"],
            "question_group": "faq_original",
            "seed_document_id": record["document_id"],
            "source": FAQ_SOURCE,
            "title": record["title"],
            "source_url": record["source_url"],
            "reference_answer": record["text"],
            "reference_contexts": [],
            "question_generator_version": None,
            "question_model": None,
            "question_prompt_sha256": None,
            "random_seed": None,
        }
        for number, record in enumerate(
            faq_records,
            start=1,
        )
    ]


def generate_question(
    client: OpenAI,
    document: dict[str, Any],
) -> str:
    """Generate one source-grounded consumer question."""

    prompt = (
        f"Source: {GENERATED_SOURCE_NAMES[document['source']]}\n"
        f"Title: {document['title']}\n\n"
        f"Excerpt:\n{document['content']}"
    )
    response = client.responses.create(
        model=QUESTION_MODEL_NAME,
        instructions=QUESTION_GENERATION_INSTRUCTIONS,
        input=prompt,
    )
    question = response.output_text.strip()

    if not question:
        raise ValueError(
            f"No question generated for {document['document_id']}"
        )

    return question


def generate_source_questions(
    client: OpenAI,
    samples: dict[str, list[dict[str, Any]]],
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    """Generate or resume the 60 source-grounded questions."""

    sampled_documents = [
        document
        for documents in samples.values()
        for document in documents
    ]
    sampled_by_id = {
        document["document_id"]: document
        for document in sampled_documents
    }
    completed_records = (
        load_jsonl(checkpoint_path)
        if checkpoint_path.exists()
        else []
    )
    completed_by_seed_id = {}

    for record in completed_records:
        seed_document_id = record.get("seed_document_id")

        if (
            record.get("generator_version")
            != QUESTION_GENERATOR_VERSION
            or record.get("question_model")
            != QUESTION_MODEL_NAME
            or record.get("random_seed") != RANDOM_SEED
            or record.get("prompt_sha256")
            != QUESTION_PROMPT_SHA256
        ):
            raise ValueError(
                f"{checkpoint_path} was created with a "
                "different generator configuration. Run with "
                "--restart to generate a new question set."
            )

        if (
            seed_document_id not in sampled_by_id
            or seed_document_id in completed_by_seed_id
        ):
            raise ValueError(
                f"{checkpoint_path} contains unexpected or "
                "duplicate seed document IDs."
            )

        document = sampled_by_id[seed_document_id]
        content_sha256 = hashlib.sha256(
            document["content"].encode()
        ).hexdigest()

        if record.get("seed_content_sha256") != content_sha256:
            raise ValueError(
                f"Processed content changed for "
                f"{seed_document_id}. Run with --restart."
            )

        completed_by_seed_id[seed_document_id] = record

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    total = (
        QUESTIONS_PER_GENERATED_SOURCE
        * len(GENERATED_SOURCE_NAMES)
    )

    for question_number, document in enumerate(
        sampled_documents,
        start=1,
    ):
        seed_document_id = document["document_id"]

        if seed_document_id in completed_by_seed_id:
            print(
                f"[{question_number}/{total}] "
                f"Skipped {seed_document_id}"
            )
            continue

        question = generate_question(client, document)
        checkpoint_record = {
            "generator_version": QUESTION_GENERATOR_VERSION,
            "question_model": QUESTION_MODEL_NAME,
            "random_seed": RANDOM_SEED,
            "prompt_sha256": QUESTION_PROMPT_SHA256,
            "seed_content_sha256": hashlib.sha256(
                document["content"].encode()
            ).hexdigest(),
            "seed_document_id": seed_document_id,
            "question": question,
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

        completed_by_seed_id[seed_document_id] = (
            checkpoint_record
        )
        print(
            f"[{question_number}/{total}] "
            f"{document['source']}: {question}"
        )

    if len(completed_by_seed_id) != total:
        raise ValueError(
            f"Expected {total} generated questions, found "
            f"{len(completed_by_seed_id)}"
        )

    return [
        {
            "question_id": f"source-{number:03d}",
            "question": completed_by_seed_id[
                document["document_id"]
            ]["question"],
            "question_group": "source_generated",
            "seed_document_id": document["document_id"],
            "source": document["source"],
            "title": document["title"],
            "source_url": document["source_url"],
            "reference_answer": None,
            "reference_contexts": [
                {
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "jurisdiction": document["jurisdiction"],
                    "source_url": document["source_url"],
                    "content": document["content"],
                }
            ],
            "question_generator_version": (
                QUESTION_GENERATOR_VERSION
            ),
            "question_model": QUESTION_MODEL_NAME,
            "question_prompt_sha256": QUESTION_PROMPT_SHA256,
            "random_seed": RANDOM_SEED,
        }
        for number, document in enumerate(
            sampled_documents,
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
    faq_records = load_jsonl(FAQ_PATH)

    if len(faq_records) != EXPECTED_SOURCE_COUNTS[FAQ_SOURCE]:
        raise ValueError(
            f"Expected 74 FAQ records, found {len(faq_records)}"
        )

    samples = sample_documents(documents)

    if arguments.restart:
        CHECKPOINT_PATH.unlink(missing_ok=True)

    client = OpenAI()

    questions = [
        finalise_question(question)
        for question in [
            *build_faq_questions(faq_records),
            *generate_source_questions(
                client,
                samples,
                CHECKPOINT_PATH,
            ),
        ]
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
