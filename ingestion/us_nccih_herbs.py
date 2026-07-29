"""Ingest NCCIH Herbs at a Glance fact sheets.

Run from the project root:

    uv run python ingestion/us_nccih_herbs.py

The script discovers herb pages from the official collection, preserves the
source HTML, and writes one processed record per page section. Reference and
resource-list sections remain available in processed data and can be excluded
later when searchable chunks are constructed.
"""

import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag


# Source configuration

BASE_URL = "https://www.nccih.nih.gov"
INDEX_URL = f"{BASE_URL}/health/herbsataglance"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "us_nccih_herbs"
DETAILS_DIR = RAW_DIR / "details"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_INDEX_PATH = RAW_DIR / "herbs_at_a_glance.html"
INDEX_OUTPUT_PATH = RAW_DIR / "herb_index.json"
ERRORS_PATH = RAW_DIR / "download_errors.json"
OUTPUT_PATH = PROCESSED_DIR / "us_nccih_herb_sections.jsonl"

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


def serialise_jsonl(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


# Index and page parsing

def clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "unknown"


def parse_herb_index(content: bytes) -> list[dict[str, str]]:
    """Discover the bounded list of herb pages before the next index section."""
    soup = BeautifulSoup(content, "html.parser")
    heading = next(
        (
            candidate
            for candidate in soup.find_all("h1")
            if clean_text(candidate.get_text(" ", strip=True))
            == "Herbs at a Glance"
        ),
        None,
    )
    if heading is None:
        raise ValueError("NCCIH Herbs at a Glance heading not found")

    herbs_by_url: dict[str, dict[str, str]] = {}

    for element in heading.next_elements:
        if isinstance(element, Tag) and element.name == "h2":
            break
        if not isinstance(element, Tag) or element.name != "a":
            continue
        if not element.get("href"):
            continue

        url = urljoin(INDEX_URL, str(element["href"]))
        parsed = urlparse(url)
        name = clean_text(element.get_text(" ", strip=True))

        if parsed.netloc.lower() != "www.nccih.nih.gov":
            continue
        if not parsed.path.lower().startswith("/health/"):
            continue
        if parsed.path.rstrip("/").lower() == "/health/herbsataglance":
            continue
        if not name:
            continue

        canonical_url = urljoin(url, parsed.path.rstrip("/"))
        herb_id = slug(parsed.path.rstrip("/").split("/")[-1])
        herbs_by_url[canonical_url] = {
            "document_id": herb_id,
            "name": name,
            "url": canonical_url,
        }

    herbs = sorted(
        herbs_by_url.values(),
        key=lambda herb: herb["document_id"],
    )
    if not herbs:
        raise ValueError("No NCCIH herb pages discovered")
    return herbs


def updated_date_from_page(soup: BeautifulSoup) -> str | None:
    match = re.search(
        r"\bLast Updated:\s*([A-Za-z]+(?: \d{1,2},)? \d{4})",
        soup.get_text(" ", strip=True),
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def labelled_value(soup: BeautifulSoup, label: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(label)}\s*:", flags=re.IGNORECASE)
    text_node = soup.find(string=pattern)
    if text_node is None:
        return None

    parent = text_node.parent
    if parent is None:
        return None

    container = parent.find_parent(["p", "div"]) or parent
    value = pattern.sub("", container.get_text(" ", strip=True))
    value = clean_text(value)
    return value or None


def section_text(heading: Tag) -> str:
    parts = [heading.get_text(" ", strip=True)]
    heading_container = heading.find_parent(
        id=re.compile(r"(?:^|-)heading(?:-|$)")
    )

    if heading_container is None:
        heading_container = heading.parent

    if isinstance(heading_container, Tag):
        heading_row: Tag = heading
        while (
            isinstance(heading_row.parent, Tag)
            and heading_row.parent is not heading_container
        ):
            heading_row = heading_row.parent

        content_container = heading_row.find_next_sibling()
        if isinstance(content_container, Tag):
            text = content_container.get_text(" ", strip=True)
            if text:
                parts.append(text)

    return clean_text(" ".join(parts))


def parse_herb_page(
    content: bytes,
    herb: dict[str, str],
    retrieved_at: str,
) -> list[dict[str, Any]]:
    """Convert one NCCIH herb page into section-level records."""
    soup = BeautifulSoup(content, "html.parser")
    title_element = soup.find("h1")
    if title_element is None:
        raise ValueError(f"Page title not found for {herb['url']}")

    page_title = clean_text(title_element.get_text(" ", strip=True))
    common_names = labelled_value(soup, "Common Names")
    latin_names = labelled_value(soup, "Latin Names")
    updated_date = updated_date_from_page(soup)
    sections: list[dict[str, Any]] = []
    section_id_counts: Counter[str] = Counter()

    for position, heading in enumerate(
        title_element.find_all_next("h2"),
        start=1,
    ):
        section_title = clean_text(heading.get_text(" ", strip=True))

        # Stop before site-wide footer headings, if the page template adds any.
        if section_title in {"Follow NIH", "Follow NCCIH"}:
            break

        base_section_id = slug(
            str(heading.get("id") or section_title or f"section-{position}")
        )
        section_id_counts[base_section_id] += 1
        occurrence = section_id_counts[base_section_id]
        section_id = (
            base_section_id
            if occurrence == 1
            else f"{base_section_id}-{occurrence}"
        )
        text = section_text(heading)

        if not text:
            continue

        sections.append(
            {
                "document_id": f"nccih-herb:{herb['document_id']}#{section_id}",
                "herb_id": herb["document_id"],
                "name": page_title,
                "common_names": common_names,
                "latin_names": latin_names,
                "section_id": section_id,
                "section_title": section_title,
                "text": text,
                "source": "nccih_herbs",
                "publisher": (
                    "National Center for Complementary and Integrative Health"
                ),
                "jurisdiction": "US",
                "document_type": "herb_fact_sheet_section",
                "evidence_role": "clinical_evidence_summary",
                "updated_date": updated_date,
                "retrieved_at": retrieved_at,
                "source_url": f"{herb['url']}#{section_id}",
            }
        )

    if not sections:
        raise ValueError(f"No sections parsed for {herb['url']}")
    return sections


def validate_records(
    herbs: list[dict[str, str]],
    sections: list[dict[str, Any]],
) -> None:
    if not herbs:
        raise ValueError("No NCCIH herbs discovered")
    if not sections:
        raise ValueError("No NCCIH herb sections parsed")

    herb_ids = [herb["document_id"] for herb in herbs]
    if len(herb_ids) != len(set(herb_ids)):
        raise ValueError("Duplicate NCCIH herb IDs found")

    document_ids = [section["document_id"] for section in sections]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate NCCIH herb section IDs found")

    parsed_herb_ids = {section["herb_id"] for section in sections}
    missing_herbs = sorted(set(herb_ids) - parsed_herb_ids)
    if missing_herbs:
        raise ValueError(f"Herbs without parsed sections: {missing_herbs}")

    required_fields = (
        "document_id",
        "herb_id",
        "name",
        "section_title",
        "text",
        "source_url",
        "publisher",
        "document_type",
        "evidence_role",
        "retrieved_at",
    )
    for section in sections:
        missing = [field for field in required_fields if not section.get(field)]
        if missing:
            raise ValueError(
                f"{section.get('document_id')} is missing: "
                f"{', '.join(missing)}"
            )


# Execution

def main() -> None:
    retrieved_at = datetime.now(UTC).isoformat()
    index_content = download(INDEX_URL)
    herbs = parse_herb_index(index_content)

    all_sections: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for position, herb in enumerate(herbs, start=1):
        raw_path = DETAILS_DIR / f"{herb['document_id']}.html"

        try:
            content = download(herb["url"])
            atomic_write(raw_path, content)
            sections = parse_herb_page(content, herb, retrieved_at)
            all_sections.extend(sections)
            print(
                f"[{position}/{len(herbs)}] {herb['name']}: "
                f"{len(sections)} sections"
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
                    "document_id": herb["document_id"],
                    "name": herb["name"],
                    "url": herb["url"],
                    "error": str(error),
                }
            )
            print(f"Failed {herb['name']}: {error}")

    atomic_write(RAW_INDEX_PATH, index_content)
    atomic_write(
        INDEX_OUTPUT_PATH,
        json.dumps(herbs, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    atomic_write(
        ERRORS_PATH,
        json.dumps(errors, ensure_ascii=False, indent=2).encode("utf-8"),
    )

    if errors:
        raise RuntimeError(f"Failed NCCIH herb pages: {len(errors)}")

    validate_records(herbs, all_sections)
    atomic_write(OUTPUT_PATH, serialise_jsonl(all_sections))

    print(f"Herb pages: {len(herbs)}")
    print(f"Parsed sections: {len(all_sections)}")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
