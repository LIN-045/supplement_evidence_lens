"""Ingest Health Canada NHPID monographs.

Run from the project root:

    uv run python -m ingestion.sources.ca_nhpid_monographs

The script discovers all single-ingredient and product monographs, saves the
source HTML, and writes their main sections as JSON Lines.
"""

import argparse
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from ingestion.ingestion_io import (
    atomic_write,
    download_bytes,
    write_json,
    write_jsonl,
)

# Source configuration

SINGLE_MONOGRAPHS_URL = (
    "https://webprod.hc-sc.gc.ca/nhpid-bdipsn/"
    "monosReq?monotype=single"
)

PRODUCT_MONOGRAPHS_URL = (
    "https://webprod.hc-sc.gc.ca/nhpid-bdipsn/"
    "monosReq?monotype=product"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ca_nhpid_monographs"

SINGLE_OUTPUT_PATH = RAW_DIR / "single_monographs.html"
PRODUCT_OUTPUT_PATH = RAW_DIR / "product_monographs.html"

REQUEST_DELAY_SECONDS = 0.5

BASE_URL = "https://webprod.hc-sc.gc.ca/nhpid-bdipsn/"
INDEX_OUTPUT_PATH = RAW_DIR / "monograph_index.json"

DETAILS_DIR = RAW_DIR / "details"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = (
    PROCESSED_DIR / "sources" / "ca_monograph_sections.jsonl"
)
ERRORS_PATH = RAW_DIR / "download_errors.json"

def parse_monograph_index(
    content: bytes,
    monograph_type: str,
) -> list[dict[str, str]]:
    """Extract monograph metadata from an official NHPID list page."""
    soup = BeautifulSoup(content, "html.parser")
    monographs: list[dict[str, str]] = []

    for link in soup.select("a.at[href]"):
        detail_href = str(link["href"]).replace("atReq.do?", "atReq?")
        detail_url = urljoin(BASE_URL, detail_href)
        query = parse_qs(urlparse(detail_url).query)
        monograph_id = query.get("atid", [None])[0]

        if monograph_id is None:
            continue

        monographs.append(
            {
                "document_id": monograph_id,
                "name": link.get_text(" ", strip=True),
                "monograph_type": monograph_type,
                "detail_url": detail_url,
            }
        )

    return monographs


def parse_monograph_sections(
    content: bytes,
    monograph: dict[str, str],
    retrieved_at: str,
) -> list[dict[str, Any]]:
    """Convert one NHPID monograph into main-section documents."""
    soup = BeautifulSoup(content, "html.parser")
    sections: list[dict[str, Any]] = []
    section_id_counts: Counter[str] = Counter()

    for section in soup.find_all("section"):
        source_section_id_value = section.get("aria-labelledby")

        if not source_section_id_value:
            continue

        source_section_id = str(source_section_id_value)
        section_id_counts[source_section_id] += 1
        occurrence = section_id_counts[source_section_id]
        section_id = (
            source_section_id
            if occurrence == 1
            else f"{source_section_id}-{occurrence}"
        )

        heading = section.find(
            ["h2", "h3", "h4"],
            recursive=False,
        )

        if heading is None:
            continue

        section_title = heading.get_text(" ", strip=True)
        section_text = " ".join(section.get_text(" ", strip=True).split())

        sections.append(
            {
                "document_id": (
                    f"{monograph['document_id']}#{section_id}"
                ),
                "monograph_id": monograph["document_id"],
                "name": monograph["name"],
                "monograph_type": monograph["monograph_type"],
                "section_id": section_id,
                "section_title": section_title,
                "text": section_text,
                "source": "health_canada_nhpid",
                "publisher": "Health Canada",
                "jurisdiction": "CA",
                "document_type": "monograph_section",
                "evidence_role": "regulatory_monograph",
                "updated_date": None,
                "retrieved_at": retrieved_at,
                "source_url": monograph["detail_url"],
            }
        )

    return sections


def validate_records(
    monographs: list[dict[str, str]],
    sections: list[dict[str, Any]],
) -> None:
    """Reject empty or duplicate source and section records."""
    if not monographs:
        raise ValueError("No Health Canada monographs found")
    if not sections:
        raise ValueError("No Health Canada monograph sections found")

    for records, label in (
        (monographs, "monograph"),
        (sections, "section"),
    ):
        document_ids = [record["document_id"] for record in records]
        duplicates = [
            document_id
            for document_id, count in Counter(document_ids).items()
            if count > 1
        ]
        if duplicates:
            raise ValueError(f"Duplicate {label} IDs: {duplicates[:10]}")


# Execution

def run(*, use_existing_raw: bool = False) -> dict[str, Any]:
    """Ingest Health Canada monographs and return a run summary."""

    if use_existing_raw:
        single_content = SINGLE_OUTPUT_PATH.read_bytes()
        product_content = PRODUCT_OUTPUT_PATH.read_bytes()
    else:
        single_content = download_bytes(SINGLE_MONOGRAPHS_URL)
        product_content = download_bytes(PRODUCT_MONOGRAPHS_URL)
        atomic_write(SINGLE_OUTPUT_PATH, single_content)
        atomic_write(PRODUCT_OUTPUT_PATH, product_content)

    retrieved_at = datetime.now(UTC).isoformat()

    single_monographs = parse_monograph_index(
        single_content,
        "single_ingredient",
    )

    product_monographs = parse_monograph_index(
        product_content,
        "product",
    )

    monographs = single_monographs + product_monographs
    all_sections: list[dict[str, Any]] = []
    download_errors: list[dict[str, str]] = []

    for position, monograph in enumerate(monographs, start=1):
        document_id = monograph["document_id"]
        output_path = DETAILS_DIR / f"{document_id}.html"

        try:
            if use_existing_raw:
                detail_content = output_path.read_bytes()
                action = "Read"
            else:
                detail_content = download_bytes(monograph["detail_url"])
                atomic_write(output_path, detail_content)
                time.sleep(REQUEST_DELAY_SECONDS)
                action = "Wrote"

            print(
                f"[{position}/{len(monographs)}] "
                f"{action} {output_path.relative_to(PROJECT_ROOT)}"
            )

            all_sections.extend(
                parse_monograph_sections(
                    detail_content,
                    monograph,
                    retrieved_at,
                )
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
                    "document_id": document_id,
                    "name": monograph["name"],
                    "detail_url": monograph["detail_url"],
                    "error": str(error),
                }
            )
            print(
                f"[{position}/{len(monographs)}] "
                f"Failed {document_id}: {error}"
            )

    write_json(ERRORS_PATH, download_errors)
    if download_errors:
        raise RuntimeError(
            f"Failed to ingest {len(download_errors)} "
            "Health Canada monographs"
        )

    validate_records(monographs, all_sections)
    write_jsonl(OUTPUT_PATH, all_sections)
    write_json(INDEX_OUTPUT_PATH, monographs)

    return {
        "source": "health_canada_nhpid",
        "record_count": len(all_sections),
        "monograph_count": len(monographs),
        "single_ingredient_count": len(single_monographs),
        "product_count": len(product_monographs),
        "download_error_count": len(download_errors),
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "errors_path": str(ERRORS_PATH.relative_to(PROJECT_ROOT)),
        "index_path": str(INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-existing-raw",
        action="store_true",
        help="Parse existing raw files without downloading.",
    )
    arguments = parser.parse_args()
    result = run(use_existing_raw=arguments.use_existing_raw)

    print(f"Parsed sections: {result['record_count']}")
    print(f"Download errors: {result['download_error_count']}")
    print(f"Wrote {result['output_path']}")
    print(f"Wrote {result['errors_path']}")
    print(
        "Single ingredient monographs: "
        f"{result['single_ingredient_count']}"
    )
    print(f"Product monographs: {result['product_count']}")
    print(f"Total monographs: {result['monograph_count']}")
    print(f"Wrote {result['index_path']}")
    print(f"Wrote {SINGLE_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {PRODUCT_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
