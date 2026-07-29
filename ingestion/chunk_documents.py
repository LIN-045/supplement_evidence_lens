"""Create document chunks from the processed source datasets.

Run from the project root:

    uv run python ingestion/chunk_documents.py
"""

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EU_PATH = PROCESSED_DIR / "eu_health_claims.jsonl"
CA_PATH = PROCESSED_DIR / "ca_monograph_sections.jsonl"
NIH_PATH = PROCESSED_DIR / "us_nih_ods_sections.jsonl"
ODS_GUIDANCE_PATH = PROCESSED_DIR / "us_ods_guidance_sections.jsonl"
DRI_PATH = PROCESSED_DIR / "us_dri_reference_values.jsonl"
NCCIH_HERBS_PATH = PROCESSED_DIR / "us_nccih_herb_sections.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "document_chunks_v2.jsonl"

CHUNK_SIZE = 2_000
CHUNK_STEP = 1_500

SECTION_PATHS = (
    CA_PATH,
    NIH_PATH,
    ODS_GUIDANCE_PATH,
    NCCIH_HERBS_PATH,
)

SOURCE_DEFAULTS: dict[str, dict[str, Any]] = {
    "eu_health_claims_register": {
        "publisher": "European Commission",
        "document_type": "health_claim",
        "evidence_role": "regulatory_claim",
        "updated_date": None,
    },
    "health_canada_nhpid": {
        "publisher": "Health Canada",
        "document_type": "monograph_section",
        "evidence_role": "regulatory_monograph",
        "updated_date": None,
    },
    "nih_ods": {
        "publisher": "NIH Office of Dietary Supplements",
        "document_type": "professional_fact_sheet_section",
        "evidence_role": "clinical_evidence_summary",
    },
    "nih_ods_guidance": {
        "publisher": "NIH Office of Dietary Supplements",
    },
    "us_dri_tables": {
        "publisher": (
            "National Academies of Sciences, Engineering, and Medicine"
        ),
        "document_type": "nutrient_reference_value",
        "evidence_role": "nutrient_reference",
        "updated_date": None,
    },
    "nccih_herbs": {
        "publisher": (
            "National Center for Complementary and Integrative Health"
        ),
        "document_type": "herb_fact_sheet_section",
        "evidence_role": "clinical_evidence_summary",
    },
}

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


# File handling

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def serialise_jsonl(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


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


def source_retrieved_at(path: Path) -> str:
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=UTC,
    ).isoformat()


def add_common_metadata(
    record: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    document = {**record}
    defaults = SOURCE_DEFAULTS.get(str(record.get("source")), {})

    for field, value in defaults.items():
        document.setdefault(field, value)

    # Existing V1 processed snapshots predate ``retrieved_at``. Their file
    # modification time is a deterministic migration fallback; future source
    # runs write the actual retrieval timestamp into each record.
    document.setdefault("retrieved_at", source_retrieved_at(path))
    document.setdefault("updated_date", None)
    return document


def prepare_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for record in read_jsonl(EU_PATH):
        document = add_common_metadata(record, EU_PATH)
        documents.append(
            {
                **document,
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
                for key, value in add_common_metadata(record, path).items()
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
            for key, value in add_common_metadata(
                record,
                DRI_PATH,
            ).items()
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

        for start in range(0, len(content), step):
            piece = content[start:start + size].strip()

            if piece:
                pieces.append(piece)
            if start + size >= len(content):
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


def main() -> None:
    documents = prepare_documents()
    chunks = chunk_documents(documents)
    validate_documents(chunks)
    atomic_write(OUTPUT_PATH, serialise_jsonl(chunks))

    source_counts = Counter(chunk["source"] for chunk in chunks)
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count}")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
