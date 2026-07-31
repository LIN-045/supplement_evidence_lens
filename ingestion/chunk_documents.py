"""Create document chunks from the processed source datasets.

Run from the project root:

    uv run python ingestion/chunk_documents.py
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any

from ingestion.ingestion_io import read_jsonl, write_jsonl


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SOURCE_DOCUMENTS_DIR = PROCESSED_DIR / "sources"

EU_PATH = SOURCE_DOCUMENTS_DIR / "eu_health_claims.jsonl"
CA_PATH = SOURCE_DOCUMENTS_DIR / "ca_monograph_sections.jsonl"
NIH_PATH = SOURCE_DOCUMENTS_DIR / "us_nih_ods_sections.jsonl"
ODS_GUIDANCE_PATH = (
    SOURCE_DOCUMENTS_DIR / "us_ods_guidance_sections.jsonl"
)
DRI_PATH = SOURCE_DOCUMENTS_DIR / "us_dri_reference_values.jsonl"
NCCIH_HERBS_PATH = (
    SOURCE_DOCUMENTS_DIR / "us_nccih_herb_sections.jsonl"
)
ODS_FAQ_PATH = SOURCE_DOCUMENTS_DIR / "us_ods_faq_answers.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "document_chunks.jsonl"

CHUNK_SIZE = 2_000
CHUNK_STEP = 1_500

SECTION_PATHS = (
    CA_PATH,
    NIH_PATH,
    ODS_GUIDANCE_PATH,
    NCCIH_HERBS_PATH,
)

EXCLUDED_SECTION_TITLES = {
    "references",
    "references cited",
    "references reviewed",
    "key references",
    "for more information",
    "table of contents",
    "tcm referenced texts",
}

NCCIH_EXCLUDED_SECTION_TITLES = {
    "keep in mind",
}


# Document preparation

def build_eu_content(record: dict[str, Any]) -> str:
    fields = (
        ("Ingredient", record.get("ingredient")),
        ("Official claim", record.get("claim_text")),
        ("Status", record.get("status")),
        ("Status code", record.get("status_code")),
        ("Claim type", record.get("claim_type")),
        ("Health relationship", record.get("health_relationship")),
        ("Conditions of use", record.get("conditions_of_use")),
        ("Restrictions", record.get("restriction_of_use")),
    )
    parts = [f"{label}: {value}" for label, value in fields if value]

    reasons = record.get("reasons_for_non_authorisation") or []
    if reasons:
        parts.append(f"Reasons for non-authorisation: {'; '.join(reasons)}")

    entry_ids = record.get("entry_ids") or []
    if entry_ids:
        parts.append(f"Entry IDs: {'; '.join(entry_ids)}")

    question_numbers = record.get("efsa_question_numbers") or []
    if question_numbers:
        parts.append(f"EFSA question numbers: {'; '.join(question_numbers)}")

    return "\n".join(parts)


def normalise_section_title(value: str) -> str:
    title = " ".join(value.lower().split())
    title = re.sub(
        r"^\d+(?:\.\d+)*\s*[-:.)]?\s*",
        "",
        title,
    )
    return title.strip()


def should_exclude_section(record: dict[str, Any]) -> bool:
    section_title = normalise_section_title(
        str(record.get("section_title") or "")
    )

    if section_title in EXCLUDED_SECTION_TITLES:
        return True
    return (
        record.get("source") == "nccih_herbs"
        and section_title in NCCIH_EXCLUDED_SECTION_TITLES
    )


def prepare_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for record in read_jsonl(EU_PATH):
        documents.append(
            {
                **record,
                "title": record.get("ingredient") or "EU health claim",
                "content": build_eu_content(record),
            }
        )

    for path in SECTION_PATHS:
        for record in read_jsonl(path):
            if should_exclude_section(record):
                continue

            document = {
                key: value
                for key, value in record.items()
                if key != "text"
            }
            documents.append(
                {
                    **document,
                    "title": (
                        f"{record['name']} — {record['section_title']}"
                    ),
                    "content": record["text"],
                }
            )

    for record in read_jsonl(DRI_PATH):
        document = {
            key: value
            for key, value in record.items()
            if key != "text"
        }
        population = ", ".join(
            part
            for part in (
                record.get("population_group"),
                record.get("life_stage"),
            )
            if part and part != "General"
        )
        title_parts = [
            str(record["nutrient"]),
            str(record["reference_type"]),
        ]
        if population:
            title_parts.append(population)

        documents.append(
            {
                **document,
                "title": " — ".join(title_parts),
                "content": record["text"],
            }
        )

    for record in read_jsonl(ODS_FAQ_PATH):
        document = {
            key: value
            for key, value in record.items()
            if key != "text"
        }
        documents.append(
            {
                **document,
                "title": record["question"],
                "content": record["text"],
            }
        )

    return documents


# Chunking

def chunk_documents(
    documents: list[dict[str, Any]],
    size: int = CHUNK_SIZE,
    step: int = CHUNK_STEP,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for document in documents:
        content = document["content"]
        pieces: list[str] = []
        prefix = ""
        piece_size = size

        if document.get("document_type") == "consumer_faq_answer":
            prefix = (
                f"Question: {document['question']}\n"
                "Answer: "
            )
            piece_size = size - len(prefix)
            if piece_size <= 0:
                raise ValueError(
                    "FAQ question is too long for the configured chunk size: "
                    f"{document['document_id']}"
                )

        for start in range(0, len(content), step):
            piece = content[start:start + piece_size].strip()

            if piece:
                pieces.append(f"{prefix}{piece}")
            if start + piece_size >= len(content):
                break

        for index, piece in enumerate(pieces):
            source_document_id = document["document_id"]
            chunks.append(
                {
                    **document,
                    "document_id": f"{source_document_id}::chunk-{index + 1}",
                    "source_document_id": source_document_id,
                    "content": piece,
                    "chunk_index": index,
                    "chunk_count": len(pieces),
                }
            )

    return chunks


# Validation and execution

def validate_documents(documents: list[dict[str, Any]]) -> None:
    if not documents:
        raise ValueError("No document chunks were created")

    document_ids = [document["document_id"] for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document chunk IDs found")

    for document in documents:
        if (
            not document.get("title")
            or not document.get("content")
            or not document.get("source_url")
            or not document.get("publisher")
            or not document.get("document_type")
            or not document.get("evidence_role")
            or not document.get("retrieved_at")
        ):
            raise ValueError(
                f"Incomplete document: {document['document_id']}"
            )


def run() -> dict[str, Any]:
    """Build all searchable chunks and return a pipeline-friendly summary."""

    documents = prepare_documents()
    chunks = chunk_documents(documents)
    validate_documents(chunks)
    write_jsonl(OUTPUT_PATH, chunks)

    source_counts = Counter(chunk["source"] for chunk in chunks)
    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "source_chunk_counts": dict(sorted(source_counts.items())),
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    result = run()

    print(f"Documents: {result['document_count']}")
    print(f"Chunks: {result['chunk_count']}")
    for source, count in result["source_chunk_counts"].items():
        print(f"{source}: {count}")
    print(f"Wrote {result['output_path']}")


if __name__ == "__main__":
    main()
