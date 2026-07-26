"""Ingest Health Canada NHPID monographs.

Run from the project root:

    uv run python ingestion/ca_nhpid_monographs.py

The script discovers all single-ingredient and product monographs, saves the
source HTML, and writes their main sections as JSON Lines.
"""

import json
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

# Source configuration

SINGLE_MONOGRAPHS_URL = (
    "https://webprod.hc-sc.gc.ca/nhpid-bdipsn/"
    "monosReq?monotype=single"
)

PRODUCT_MONOGRAPHS_URL = (
    "https://webprod.hc-sc.gc.ca/nhpid-bdipsn/"
    "monosReq?monotype=product"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ca_nhpid_monographs"

SINGLE_OUTPUT_PATH = RAW_DIR / "single_monographs.html"
PRODUCT_OUTPUT_PATH = RAW_DIR / "product_monographs.html"

USER_AGENT = "SupplementEvidenceLens/0.1"
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3

BASE_URL = "https://webprod.hc-sc.gc.ca/nhpid-bdipsn/"
INDEX_OUTPUT_PATH = RAW_DIR / "monograph_index.json"

DETAILS_DIR = RAW_DIR / "details"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "ca_monograph_sections.jsonl"
ERRORS_PATH = RAW_DIR / "download_errors.json"

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
) -> list[dict[str, str]]:
    """Convert one NHPID monograph into main-section documents."""
    soup = BeautifulSoup(content, "html.parser")
    sections: list[dict[str, str]] = []
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
                "jurisdiction": "CA",
                "source_url": monograph["detail_url"],
            }
        )

    return sections


def validate_records(
    monographs: list[dict[str, str]],
    sections: list[dict[str, str]],
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


def serialise_jsonl(records: list[dict[str, str]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


# Execution

def main() -> None:
    single_content = download(SINGLE_MONOGRAPHS_URL)
    product_content = download(PRODUCT_MONOGRAPHS_URL)

    atomic_write(SINGLE_OUTPUT_PATH, single_content)
    atomic_write(PRODUCT_OUTPUT_PATH, product_content)

    single_monographs = parse_monograph_index(
        single_content,
        "single_ingredient",
    )

    product_monographs = parse_monograph_index(
        product_content,
        "product",
    )

    monographs = single_monographs + product_monographs
    all_sections: list[dict[str, str]] = []
    download_errors: list[dict[str, str]] = []

    for position, monograph in enumerate(monographs, start=1):
        document_id = monograph["document_id"]
        output_path = DETAILS_DIR / f"{document_id}.html"

        try:
            if output_path.exists():
                detail_content = output_path.read_bytes()
                action = "Read"
            else:
                detail_content = download(monograph["detail_url"])
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

    validate_records(monographs, all_sections)
    atomic_write(OUTPUT_PATH, serialise_jsonl(all_sections))

    errors_content = json.dumps(
        download_errors,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    atomic_write(ERRORS_PATH, errors_content)

    print(f"Parsed sections: {len(all_sections)}")
    print(f"Download errors: {len(download_errors)}")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {ERRORS_PATH.relative_to(PROJECT_ROOT)}")

    index_content = json.dumps(
        monographs,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    atomic_write(INDEX_OUTPUT_PATH, index_content)

    print(f"Single ingredient monographs: {len(single_monographs)}")
    print(f"Product monographs: {len(product_monographs)}")
    print(f"Total monographs: {len(monographs)}")
    print(f"Wrote {INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    print(f"Wrote {SINGLE_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {PRODUCT_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
