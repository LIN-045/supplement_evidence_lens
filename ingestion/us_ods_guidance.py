"""Ingest selected NIH ODS consumer-guidance and nutrient-reference pages.

Run from the project root:

    uv run python ingestion/us_ods_guidance.py

The script downloads three deliberately selected ODS pages, preserves their
HTML under ``data/raw``, and writes one processed record per useful section.
It does not ingest the NIH Consumer FAQ or any linked DRI reports.
"""

import json
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag


# Source configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "us_ods_guidance"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "us_ods_guidance_sections.jsonl"
ERRORS_PATH = RAW_DIR / "download_errors.json"

USER_AGENT = "SupplementEvidenceLens/0.1"
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3

PAGE_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "page_id": "what-you-need-to-know",
        "url": "https://ods.od.nih.gov/factsheets/WYNTK-Consumer/",
        "document_type": "consumer_fact_sheet_section",
        "evidence_role": "consumer_guidance",
        "content_selector": "#fact-sheet",
        "included_headings": None,
    },
    {
        "page_id": "botanical-background",
        "url": (
            "https://ods.od.nih.gov/factsheets/"
            "BotanicalBackground-Consumer/"
        ),
        "document_type": "consumer_fact_sheet_section",
        "evidence_role": "consumer_guidance",
        "content_selector": "#fact-sheet",
        "included_headings": None,
    },
    {
        "page_id": "nutrient-recommendations",
        "url": (
            "https://ods.od.nih.gov/HealthInformation/"
            "nutrientrecommendations.aspx"
        ),
        "document_type": "nutrient_reference_section",
        "evidence_role": "nutrient_reference",
        "content_selector": "main",
        "included_headings": {
            "Nutrient Recommendations: Dietary Reference Intakes (DRI)",
            "Online DRI Tool",
            "Daily Values",
            "USDA FoodData Central",
            "USDA Databases",
        },
    },
)

EXCLUDED_HEADINGS = {
    "Table of Contents",
    "Disclaimer",
}


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


def serialise_jsonl(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


# Page parsing

def clean_text(value: str) -> str:
    return " ".join(value.split())


def updated_date_from_page(soup: BeautifulSoup) -> str | None:
    match = re.search(
        r"\b(?:Updated|Last Updated):\s*([A-Za-z]+ \d{1,2},? \d{4})",
        soup.get_text(" ", strip=True),
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def iter_section_parts(heading: Tag) -> Iterable[str]:
    yield heading.get_text(" ", strip=True)

    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag) and sibling.name == "h2":
            break
        if isinstance(sibling, Tag):
            text = sibling.get_text(" ", strip=True)
            if text:
                yield text


def parse_page(
    content: bytes,
    config: dict[str, Any],
    retrieved_at: str,
) -> list[dict[str, Any]]:
    """Convert one selected ODS page into section-level records."""
    soup = BeautifulSoup(content, "html.parser")
    root = soup.select_one(config["content_selector"])

    if root is None:
        raise ValueError(
            f"Content root {config['content_selector']!r} not found for "
            f"{config['url']}"
        )

    title_element = soup.select_one("h1.fsTitle") or root.find("h1")
    if title_element is None:
        raise ValueError(f"Page title not found for {config['url']}")

    page_title = clean_text(title_element.get_text(" ", strip=True))
    included_headings = config["included_headings"]
    updated_date = updated_date_from_page(soup)
    sections: list[dict[str, Any]] = []

    for position, heading in enumerate(root.find_all("h2"), start=1):
        section_title = clean_text(heading.get_text(" ", strip=True))

        section_title_lower = section_title.lower()
        if (
            section_title in EXCLUDED_HEADINGS
            or "information sources" in section_title_lower
            or "additional sources of information" in section_title_lower
        ):
            continue
        if (
            included_headings is not None
            and section_title not in included_headings
        ):
            continue

        section_id = str(
            heading.get("id")
            or f"section-{position}"
        )
        section_text = clean_text(" ".join(iter_section_parts(heading)))

        if not section_text:
            continue

        sections.append(
            {
                "document_id": f"ods-guidance:{config['page_id']}#{section_id}",
                "page_id": config["page_id"],
                "name": page_title,
                "section_id": section_id,
                "section_title": section_title,
                "text": section_text,
                "source": "nih_ods_guidance",
                "publisher": "NIH Office of Dietary Supplements",
                "jurisdiction": "US",
                "document_type": config["document_type"],
                "evidence_role": config["evidence_role"],
                "updated_date": updated_date,
                "retrieved_at": retrieved_at,
                "source_url": urljoin(
                    config["url"],
                    f"#{section_id}",
                ),
            }
        )

    if not sections:
        raise ValueError(f"No selected sections found for {config['url']}")

    return sections


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No ODS guidance sections were parsed")

    document_ids = [record["document_id"] for record in records]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate ODS guidance document IDs found")

    required_fields = (
        "document_id",
        "name",
        "section_title",
        "text",
        "source_url",
        "publisher",
        "document_type",
        "evidence_role",
        "retrieved_at",
    )
    for record in records:
        missing = [field for field in required_fields if not record.get(field)]
        if missing:
            raise ValueError(
                f"{record.get('document_id')} is missing: {', '.join(missing)}"
            )


# Execution

def main() -> None:
    retrieved_at = datetime.now(UTC).isoformat()
    all_sections: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for position, config in enumerate(PAGE_CONFIGS, start=1):
        raw_path = RAW_DIR / f"{config['page_id']}.html"

        try:
            content = download(config["url"])
            atomic_write(raw_path, content)
            sections = parse_page(content, config, retrieved_at)
            all_sections.extend(sections)
            print(
                f"[{position}/{len(PAGE_CONFIGS)}] "
                f"{config['page_id']}: {len(sections)} sections"
            )
            time.sleep(REQUEST_DELAY_SECONDS)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as error:
            errors.append(
                {
                    "page_id": config["page_id"],
                    "url": config["url"],
                    "error": str(error),
                }
            )
            print(f"Failed {config['page_id']}: {error}")

    if errors:
        atomic_write(
            ERRORS_PATH,
            json.dumps(errors, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        raise RuntimeError(f"Failed ODS guidance pages: {len(errors)}")

    validate_records(all_sections)
    atomic_write(OUTPUT_PATH, serialise_jsonl(all_sections))
    atomic_write(ERRORS_PATH, b"[]\n")

    print(f"Parsed sections: {len(all_sections)}")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
