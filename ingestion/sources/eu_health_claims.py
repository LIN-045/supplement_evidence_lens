"""Ingest health claims from the European Commission EU Register.

Run from the project root:

    uv run python -m ingestion.sources.eu_health_claims

The script saves the official API responses under ``data/raw`` and writes one
normalised JSON object per line under ``data/processed``.
"""

import hashlib
import html
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingestion.ingestion_io import (
    atomic_write,
    download_json,
    write_jsonl,
)

# Source configuration

API_BASE = "https://ec.europa.eu/food/food-feed-portal/backend/api"
CLAIMS_URL = (
    f"{API_BASE}/policy-items?foodDomain=nut&authorisationType=nut_auth"
)
STATUSES_URL = f"{API_BASE}/keyed-reference-data?key=hc_claim_status"
REGISTER_URL = (
    "https://ec.europa.eu/food/food-feed-portal/"
    "screen/health-claims/eu-register"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "eu_health_claims"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_CLAIMS_PATH = RAW_DIR / "health_claims.json"
RAW_STATUSES_PATH = RAW_DIR / "claim_statuses.json"
OUTPUT_PATH = PROCESSED_DIR / "sources" / "eu_health_claims.jsonl"

HTML_TAG = re.compile(r"<[^>]+>")


# Record parsing

def collect_values(node: dict[str, Any], identifier: str) -> list[Any]:
    """Recursively collect matching fields from a nested portal record."""
    values: list[Any] = []

    if node.get("valueIdentifier") == identifier and node.get("value") is not None:
        values.append(node["value"])

    for child in node.get("childrenValues", []):
        values.extend(collect_values(child, identifier))

    return values


def first_value(node: dict[str, Any], identifier: str) -> Any:
    values = collect_values(node, identifier)
    return values[0] if values else None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = html.unescape(HTML_TAG.sub(" ", str(value)))
    text = " ".join(text.split())
    return text or None


def clean_values(values: list[Any]) -> list[str]:
    cleaned = [clean_text(value) for value in values]
    return [value for value in cleaned if value is not None]


def normalise_record(
    record: dict[str, Any],
    status_labels: dict[str, str],
    retrieved_at: str,
) -> dict[str, Any]:
    """Convert one nested portal record to the project's EU claim schema."""
    document_id = clean_text(first_value(record, "policyItemCode"))
    status_code = clean_text(first_value(record, "hcClaimStatus"))

    return {
        "document_id": document_id,
        "source": "eu_health_claims_register",
        "publisher": "European Commission",
        "jurisdiction": "EU",
        "document_type": "health_claim",
        "evidence_role": "regulatory_claim",
        "updated_date": None,
        "retrieved_at": retrieved_at,
        "ingredient": clean_text(first_value(record, "hcNutSubFoodCat")),
        "claim_text": clean_text(first_value(record, "hcClaim")),
        "status": status_labels.get(status_code, status_code),
        "status_code": status_code,
        "claim_type": clean_text(first_value(record, "hcClaimType")),
        "health_relationship": clean_text(
            first_value(record, "hcHealthRelationship")
        ),
        "conditions_of_use": clean_text(first_value(record, "hcCondOfUse")),
        "restriction_of_use": clean_text(
            first_value(record, "hcRestrictionOfUse")
        ),
        "reasons_for_non_authorisation": clean_values(
            collect_values(record, "hcReasonsForNonAuth")
        ),
        "entry_ids": clean_values(collect_values(record, "hcEntryId")),
        "efsa_question_numbers": clean_values(
            collect_values(record, "hcEfsaQuestionNbr")
        ),
        "source_url": f"{REGISTER_URL}/details/{document_id}",
    }


# Validation and execution

def validate_records(records: list[dict[str, Any]]) -> None:
    """Reject incomplete output before it is written to the processed dataset."""
    if not records:
        raise ValueError("The EU Register returned no records")

    required_fields = ("document_id", "ingredient", "claim_text", "status")
    for index, record in enumerate(records):
        missing = [field for field in required_fields if not record.get(field)]
        if missing:
            raise ValueError(
                f"Record {index} is missing required fields: {', '.join(missing)}"
            )

    document_ids = [record["document_id"] for record in records]
    if len(document_ids) != len(set(document_ids)):
        duplicates = [
            document_id
            for document_id, count in Counter(document_ids).items()
            if count > 1
        ]
        raise ValueError(f"Duplicate document IDs: {duplicates[:10]}")


def run() -> dict[str, Any]:
    """Ingest the EU Register and return a Prefect-friendly summary."""

    claims_content, raw_claims = download_json(CLAIMS_URL)
    statuses_content, raw_statuses = download_json(STATUSES_URL)
    retrieved_at = datetime.now(UTC).isoformat()

    if not isinstance(raw_claims, list):
        raise ValueError("Claims response must be a JSON list")
    if not isinstance(raw_statuses, list):
        raise ValueError("Claim statuses response must be a JSON list")

    status_labels = {
        item["value"]: item["description"]
        for item in raw_statuses
        if item.get("value") and item.get("description")
    }
    records = [
        normalise_record(record, status_labels, retrieved_at)
        for record in raw_claims
    ]
    validate_records(records)

    atomic_write(RAW_CLAIMS_PATH, claims_content)
    atomic_write(RAW_STATUSES_PATH, statuses_content)
    write_jsonl(OUTPUT_PATH, records)

    status_counts = Counter(record["status"] for record in records)
    return {
        "source": "eu_health_claims_register",
        "record_count": len(records),
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "raw_sha256": hashlib.sha256(claims_content).hexdigest(),
        "status_counts": dict(sorted(status_counts.items())),
    }


def main() -> None:
    result = run()

    print(
        f"Downloaded {result['record_count']:,} "
        "EU health-claim records"
    )
    for status, count in result["status_counts"].items():
        print(f"{status}: {count:,}")
    print(f"Raw SHA-256: {result['raw_sha256']}")
    print(f"Wrote {result['output_path']}")


if __name__ == "__main__":
    main()
