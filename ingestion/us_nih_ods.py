"""Ingest NIH Office of Dietary Supplements fact sheets.

Run from the project root:

    uv run python ingestion/us_nih_ods.py

The script discovers all health-professional fact sheets from the official
index, saves the source HTML, and writes their main sections as JSON Lines.
"""

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag


# Source configuration

BASE_URL = "https://ods.od.nih.gov"
INDEX_URL = f"{BASE_URL}/factsheets/list-all/"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "us_nih_ods"
DETAILS_DIR = RAW_DIR / "details"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_INDEX_PATH = RAW_DIR / "fact_sheet_index.html"
INDEX_OUTPUT_PATH = RAW_DIR / "fact_sheet_index.json"
ERRORS_PATH = RAW_DIR / "download_errors.json"
OUTPUT_PATH = PROCESSED_DIR / "us_nih_ods_sections.jsonl"

USER_AGENT = "SupplementEvidenceLens/0.1"
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3


# Download and file handling

def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError):
            if attempt == MAX_RETRIES:
                raise

            time.sleep(attempt)

    raise RuntimeError(f"Failed to download {url}")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


# Page parsing

def document_id_from_url(url: str) -> str:
    endpoint = urlparse(url).path.rstrip("/").split("/")[-1]
    endpoint = re.sub(
        r"-HealthProfessional$",
        "",
        endpoint,
        flags=re.IGNORECASE,
    )
    endpoint = re.sub(r"(?<=[a-z])(?=[A-Z])", "-", endpoint)
    endpoint = re.sub(r"(?<=[A-Za-z])(?=\d)", "-", endpoint)
    endpoint = re.sub(r"(?<=\d)(?=[A-Za-z])", "-", endpoint)
    return endpoint.lower()


def parse_fact_sheet_index(content: bytes) -> list[dict[str, str]]:
    """Discover unique NIH-hosted health-professional fact sheets."""
    soup = BeautifulSoup(content, "html.parser")
    fact_sheets_by_url: dict[str, dict[str, str]] = {}

    for link in soup.find_all("a", href=True):
        url = urljoin(INDEX_URL, str(link["href"]))
        parsed_url = urlparse(url)

        if parsed_url.netloc.lower() != "ods.od.nih.gov":
            continue

        if not parsed_url.path.rstrip("/").lower().endswith(
            "-healthprofessional"
        ):
            continue

        canonical_url = urljoin(url, parsed_url.path.rstrip("/") + "/")
        fact_sheets_by_url[canonical_url] = {
            "document_id": document_id_from_url(canonical_url),
            "url": canonical_url,
        }

    fact_sheets = sorted(
        fact_sheets_by_url.values(),
        key=lambda item: item["document_id"],
    )

    if not fact_sheets:
        raise ValueError("No NIH health-professional fact sheets found")

    return fact_sheets


def parse_fact_sheet(
    content: bytes,
    fact_sheet: dict[str, str],
    retrieved_at: str,
) -> list[dict[str, Any]]:
    """Convert one NIH fact sheet into main-section retrieval documents."""
    soup = BeautifulSoup(content, "html.parser")
    fact_sheet_content = soup.select_one("#fact-sheet")

    if fact_sheet_content is None:
        raise ValueError(f"Fact-sheet content not found for {fact_sheet['url']}")

    title_element = soup.select_one("h1.fsTitle")
    if title_element is None:
        raise ValueError(f"Fact-sheet title not found for {fact_sheet['url']}")

    name = title_element.get_text(" ", strip=True)
    updated_element = fact_sheet_content.select_one(
        "[id$='lblUpdatedDate']"
    )
    updated_date = (
        updated_element.get_text(" ", strip=True)
        if updated_element is not None
        else None
    )

    sections: list[dict[str, Any]] = []

    for heading in fact_sheet_content.find_all("h2", id=True):
        section_id = str(heading["id"])
        section_title = heading.get_text(" ", strip=True)
        section_parts = [section_title]

        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag) and sibling.name == "h2":
                break

            if isinstance(sibling, Tag):
                text = sibling.get_text(" ", strip=True)
                if text:
                    section_parts.append(text)

        section_text = " ".join(" ".join(section_parts).split())

        sections.append(
            {
                "document_id": (
                    f"{fact_sheet['document_id']}#{section_id}"
                ),
                "fact_sheet_id": fact_sheet["document_id"],
                "name": name,
                "section_id": section_id,
                "section_title": section_title,
                "text": section_text,
                "updated_date": updated_date,
                "source": "nih_ods",
                "publisher": "NIH Office of Dietary Supplements",
                "jurisdiction": "US",
                "document_type": "professional_fact_sheet_section",
                "evidence_role": "clinical_evidence_summary",
                "retrieved_at": retrieved_at,
                "source_url": urljoin(
                    fact_sheet["url"],
                    f"#{section_id}",
                ),
            }
        )

    if not sections:
        raise ValueError(f"No sections found for {fact_sheet['url']}")

    return sections


def serialise_jsonl(records: list[dict[str, str]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def validate_records(
    fact_sheets: list[dict[str, str]],
    sections: list[dict[str, Any]],
) -> None:
    """Reject empty or duplicate source and section records."""
    if not fact_sheets:
        raise ValueError("No NIH fact sheets found")
    if not sections:
        raise ValueError("No NIH fact-sheet sections found")

    for records, label in (
        (fact_sheets, "fact-sheet"),
        (sections, "section"),
    ):
        document_ids = [record["document_id"] for record in records]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(f"Duplicate NIH {label} IDs found")


# Execution

def main() -> None:
    index_content = download(INDEX_URL)
    fact_sheets = parse_fact_sheet_index(index_content)
    retrieved_at = datetime.now(UTC).isoformat()

    all_sections: list[dict[str, Any]] = []
    download_errors: list[dict[str, str]] = []

    for position, fact_sheet in enumerate(fact_sheets, start=1):
        raw_path = DETAILS_DIR / f"{fact_sheet['document_id']}.html"

        try:
            if raw_path.exists():
                content = raw_path.read_bytes()
                action = "Read"
            else:
                content = download(fact_sheet["url"])
                atomic_write(raw_path, content)
                time.sleep(REQUEST_DELAY_SECONDS)
                action = "Wrote"

            sections = parse_fact_sheet(
                content,
                fact_sheet,
                retrieved_at,
            )
            all_sections.extend(sections)

            print(
                f"[{position}/{len(fact_sheets)}] {action} "
                f"{raw_path.relative_to(PROJECT_ROOT)}: "
                f"{len(sections)} sections"
            )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as error:
            download_errors.append(
                {
                    "document_id": fact_sheet["document_id"],
                    "url": fact_sheet["url"],
                    "error": str(error),
                }
            )
            print(
                f"[{position}/{len(fact_sheets)}] "
                f"Failed {fact_sheet['document_id']}: {error}"
            )

    validate_records(fact_sheets, all_sections)
    atomic_write(OUTPUT_PATH, serialise_jsonl(all_sections))

    index_output = json.dumps(
        fact_sheets,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    errors_output = json.dumps(
        download_errors,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    atomic_write(RAW_INDEX_PATH, index_content)
    atomic_write(INDEX_OUTPUT_PATH, index_output)
    atomic_write(ERRORS_PATH, errors_output)

    print(f"Fact sheets: {len(fact_sheets)}")
    print(f"Parsed sections: {len(all_sections)}")
    print(f"Download errors: {len(download_errors)}")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {ERRORS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
