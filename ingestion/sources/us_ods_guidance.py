"""Ingest selected NIH ODS consumer-guidance and nutrient-reference pages.

Run from the project root:

    uv run python -m ingestion.sources.us_ods_guidance

The script downloads three deliberately selected ODS pages, preserves their
HTML under ``data/raw``, and writes one processed record per useful section.
It does not ingest the NIH Consumer FAQ or any linked DRI reports.
"""

import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from ingestion.ingestion_io import (
    atomic_write,
    download_bytes,
    write_json,
    write_jsonl,
)

# Source configuration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "us_ods_guidance"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = (
    PROCESSED_DIR / "sources" / "us_ods_guidance_sections.jsonl"
)
ERRORS_PATH = RAW_DIR / "download_errors.json"

REQUEST_DELAY_SECONDS = 0.5

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

def run() -> dict[str, Any]:
    """Ingest selected ODS guidance pages and return a summary."""

    retrieved_at = datetime.now(UTC).isoformat()
    all_sections: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for position, config in enumerate(PAGE_CONFIGS, start=1):
        raw_path = RAW_DIR / f"{config['page_id']}.html"

        try:
            content = download_bytes(config["url"])
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
        write_json(ERRORS_PATH, errors)
        raise RuntimeError(f"Failed ODS guidance pages: {len(errors)}")

    validate_records(all_sections)
    write_jsonl(OUTPUT_PATH, all_sections)
    write_json(ERRORS_PATH, [])

    return {
        "source": "nih_ods_guidance",
        "record_count": len(all_sections),
        "page_count": len(PAGE_CONFIGS),
        "download_error_count": len(errors),
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "errors_path": str(ERRORS_PATH.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    result = run()

    print(f"Parsed sections: {result['record_count']}")
    print(f"Wrote {result['output_path']}")


if __name__ == "__main__":
    main()
