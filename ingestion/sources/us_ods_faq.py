"""NIH ODS consumer FAQ source-ingestion adapter."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from ingestion.ingestion_io import (
    atomic_write,
    download_bytes,
    write_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = (
    "https://ods.od.nih.gov/HealthInformation/"
    "ODS_Frequently_Asked_Questions.aspx"
)
DEFAULT_RAW_PATH = (
    PROJECT_ROOT / "data/raw/us_ods_faq/consumer_faq.html"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/sources/us_ods_faq_answers.jsonl"
)

EXCLUDED_SECTIONS = {
    "dietary supplement sales and market data",
    "ods website materials and link requests",
    "media inquiries",
    "table of contents",
}


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalise_question(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"^q\.\s*", "", value)
    return re.sub(r"[^\w]+", " ", value).strip()


def section_fragment(heading: Tag | None) -> str | None:
    if heading is None:
        return None
    heading_id = heading.get("id")
    if heading_id:
        return str(heading_id)
    anchor = heading.find("a", attrs={"name": True})
    return str(anchor["name"]) if anchor else None


def extract_records(
    html: bytes,
    source_sha256: str,
    retrieved_at: str,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.find(id="ctl00_ContentPlaceHolder1")
    if article is None:
        raise RuntimeError("Could not locate the FAQ article in the downloaded page")

    records_by_question: OrderedDict[str, dict[str, Any]] = OrderedDict()
    current_heading: Tag | None = None
    current_section = ""
    current_record: dict[str, Any] | None = None

    for element in article.find_all(["h2", "h3", "p", "ul", "ol"], recursive=True):
        if element.find_parent(["p", "ul", "ol"]) is not None:
            continue

        if element.name in {"h2", "h3"}:
            current_heading = element
            current_section = clean_text(element.get_text(" ", strip=True))
            current_record = None
            continue

        section_key = current_section.lower()
        if section_key in EXCLUDED_SECTIONS:
            current_record = None
            continue

        text = clean_text(element.get_text(" ", strip=True))
        strong = element.find("strong", recursive=False) if element.name == "p" else None
        strong_text = clean_text(strong.get_text(" ", strip=True)) if strong else ""
        is_question = element.name == "p" and strong_text.lower().startswith("q.")

        if is_question:
            question = re.sub(r"^Q\.\s*", "", strong_text, flags=re.IGNORECASE)
            question_key = normalise_question(question)
            if not question_key:
                current_record = None
                continue

            fragment = section_fragment(current_heading)
            source_url = f"{SOURCE_URL}#{fragment}" if fragment else SOURCE_URL
            candidate = {
                "question": question,
                "sections": [current_section] if current_section else [],
                "answer_parts": [],
                "answer_urls": [],
                "source_url": source_url,
                "source_sha256": source_sha256,
                "duplicate_count": 1,
            }

            if question_key in records_by_question:
                existing = records_by_question[question_key]
                existing["duplicate_count"] += 1
                if current_section and current_section not in existing["sections"]:
                    existing["sections"].append(current_section)
                current_record = existing
            else:
                records_by_question[question_key] = candidate
                current_record = candidate
            continue

        if current_record is None or not text:
            continue

        current_record["answer_parts"].append(text)
        for anchor in element.find_all("a", href=True):
            absolute_url = urljoin(SOURCE_URL, anchor["href"])
            if absolute_url not in current_record["answer_urls"]:
                current_record["answer_urls"].append(absolute_url)

    records: list[dict[str, Any]] = []
    for question_key, item in records_by_question.items():
        answer = clean_text(" ".join(item.pop("answer_parts")))
        if not answer:
            raise RuntimeError(f"FAQ question has no answer: {item['question']}")

        stable_id = hashlib.sha256(question_key.encode("utf-8")).hexdigest()[:16]
        title = f"NIH ODS Consumer FAQ — {item['question']}"
        records.append(
            {
                "document_id": f"us-ods-faq:{stable_id}",
                "title": title,
                "question": item["question"],
                "text": answer,
                "source": "us_nih_ods_faq",
                "publisher": "NIH Office of Dietary Supplements",
                "jurisdiction": "US",
                "source_url": item["source_url"],
                "document_type": "consumer_faq_answer",
                "evidence_role": "consumer_guidance",
                "updated_date": None,
                "retrieved_at": retrieved_at,
                "sections": item["sections"],
                "answer_urls": item["answer_urls"],
                "source_sha256": item["source_sha256"],
                "duplicate_count": item["duplicate_count"],
            }
        )

    return records


def run(
    *,
    url: str = SOURCE_URL,
    raw_path: Path = DEFAULT_RAW_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    use_existing_raw: bool = False,
) -> dict[str, Any]:
    """Ingest the NIH ODS consumer FAQ and return a run summary."""

    if use_existing_raw:
        html = raw_path.read_bytes()
    else:
        html = download_bytes(url, timeout_seconds=60)
        atomic_write(raw_path, html)

    source_sha256 = hashlib.sha256(html).hexdigest()
    retrieved_at = datetime.now(UTC).isoformat()
    records = extract_records(
        html,
        source_sha256,
        retrieved_at,
    )
    if len(records) != 74:
        raise RuntimeError(
            f"Expected 74 unique FAQ questions, but parsed {len(records)}. "
            "Review the source page or parser before indexing."
        )

    write_jsonl(output_path, records)

    def summary_path(path: Path) -> str:
        resolved_path = path.resolve()
        try:
            return str(resolved_path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(resolved_path)

    return {
        "source": "us_nih_ods_faq",
        "record_count": len(records),
        "output_path": summary_path(output_path),
        "raw_path": summary_path(raw_path),
        "raw_sha256": source_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--use-existing-raw",
        action="store_true",
        help="Parse the existing raw HTML instead of downloading it again.",
    )
    args = parser.parse_args()
    result = run(
        url=args.url,
        raw_path=args.raw_path,
        output_path=args.output,
        use_existing_raw=args.use_existing_raw,
    )

    print(
        f"Wrote {result['record_count']} FAQ documents "
        f"to {result['output_path']}"
    )
    print(f"Source SHA-256: {result['raw_sha256']}")


if __name__ == "__main__":
    main()
