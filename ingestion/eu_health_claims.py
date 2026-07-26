"""Ingest health claims from the European Commission EU Register.

Run from the project root:

    uv run python ingestion/eu_health_claims.py

The script saves the official API responses under ``data/raw`` and writes one
normalised JSON object per line under ``data/processed``.
"""

import hashlib
import html
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "eu_health_claims"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_CLAIMS_PATH = RAW_DIR / "health_claims.json"
RAW_STATUSES_PATH = RAW_DIR / "claim_statuses.json"
OUTPUT_PATH = PROCESSED_DIR / "eu_health_claims.jsonl"

USER_AGENT = "SupplementEvidenceLens/0.1"
REQUEST_TIMEOUT_SECONDS = 180
MAX_RETRIES = 3
HTML_TAG = re.compile(r"<[^>]+>")


# Download and file handling

def download_json(url: str) -> tuple[bytes, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                content = response.read()
            break
        except (HTTPError, URLError, TimeoutError):
            if attempt == MAX_RETRIES:
                raise

            time.sleep(attempt)

    try:
        return content, json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Expected JSON from {url}") from error


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


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
) -> dict[str, Any]:
    """Convert one nested portal record to the project's EU claim schema."""
    document_id = clean_text(first_value(record, "policyItemCode"))
    status_code = clean_text(first_value(record, "hcClaimStatus"))

    return {
        "document_id": document_id,
        "source": "eu_health_claims_register",
        "jurisdiction": "EU",
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


def serialise_jsonl(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> None:
    claims_content, raw_claims = download_json(CLAIMS_URL)
    statuses_content, raw_statuses = download_json(STATUSES_URL)

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
        normalise_record(record, status_labels)
        for record in raw_claims
    ]
    validate_records(records)

    output_content = serialise_jsonl(records)
    atomic_write(RAW_CLAIMS_PATH, claims_content)
    atomic_write(RAW_STATUSES_PATH, statuses_content)
    atomic_write(OUTPUT_PATH, output_content)

    status_counts = Counter(record["status"] for record in records)
    print(f"Downloaded {len(records):,} EU health-claim records")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count:,}")
    print(f"Raw SHA-256: {hashlib.sha256(claims_content).hexdigest()}")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
