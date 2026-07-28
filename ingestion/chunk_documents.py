"""Create document chunks from the processed source datasets.

Run from the project root:

    uv run python ingestion/chunk_documents.py
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EU_PATH = PROCESSED_DIR / "eu_health_claims.jsonl"
CA_PATH = PROCESSED_DIR / "ca_monograph_sections.jsonl"
NIH_PATH = PROCESSED_DIR / "us_nih_ods_sections.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "document_chunks.jsonl"

CHUNK_SIZE = 2_000
CHUNK_STEP = 1_500


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

    for path in (CA_PATH, NIH_PATH):
        for record in read_jsonl(path):
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
