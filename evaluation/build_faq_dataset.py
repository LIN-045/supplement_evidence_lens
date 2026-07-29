"""Build a deduplicated NIH ODS FAQ evaluation dataset."""

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from bs4.element import Tag


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data/raw/us_nih_ods/consumer_faq.html"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/faq/nih_ods_faq_holdout.jsonl"
)

SOURCE_URL = (
    "https://ods.od.nih.gov/HealthInformation/"
    "ODS_Frequently_Asked_Questions.aspx"
)

EXCLUDED_SECTIONS = {
    "Dietary Supplement Sales and Market Data",
    "ODS Website Materials and Link Requests",
    "Media Inquiries",
}


def clean_text(value: str) -> str:
    """Collapse HTML whitespace into readable plain text."""

    return re.sub(r"\s+", " ", value).strip()


def normalise_question(question: str) -> str:
    """Create a stable key for duplicate detection."""

    return clean_text(question).casefold()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def node_text(node: Tag) -> str:
    """Extract readable text from one answer node."""

    return clean_text(node.get_text(" ", strip=True))


def node_urls(node: Tag) -> list[str]:
    """Extract absolute URLs from one answer node."""

    urls = []

    for link in node.find_all("a", href=True):
        url = urljoin(SOURCE_URL, link["href"])

        if url not in urls:
            urls.append(url)

    return urls


def build_dataset() -> list[dict]:
    """Parse, filter, and deduplicate FAQ entries."""

    html = INPUT_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")

    if article is None:
        raise ValueError("Could not find the FAQ article")

    source_sha256 = file_sha256(INPUT_PATH)
    records: OrderedDict[str, dict] = OrderedDict()

    current_section: str | None = None
    current_section_id: str | None = None
    current_question: str | None = None
    answer_nodes: list[Tag] = []
    included_occurrences = 0

    def save_current_question() -> None:
        nonlocal included_occurrences

        if current_question is None or current_section is None:
            return

        answer_parts = [
            node_text(node)
            for node in answer_nodes
            if node_text(node)
        ]
        reference_answer = "\n\n".join(answer_parts)

        if not reference_answer:
            raise ValueError(
                f"Question has no answer: {current_question}"
            )

        answer_urls = []

        for node in answer_nodes:
            for url in node_urls(node):
                if url not in answer_urls:
                    answer_urls.append(url)

        key = normalise_question(current_question)
        included_occurrences += 1

        if key in records:
            record = records[key]

            existing_answer = clean_text(
                record["reference_answer"]
            )
            current_answer = clean_text(reference_answer)

            if existing_answer != current_answer:
                raise ValueError(
                    "Duplicate question has different answers: "
                    f"{current_question}"
                )

            if current_section not in record["sections"]:
                record["sections"].append(current_section)

            for url in answer_urls:
                if url not in record["answer_urls"]:
                    record["answer_urls"].append(url)

            record["duplicate_count"] += 1
            return

        fragment = (
            f"#{current_section_id}"
            if current_section_id
            else ""
        )

        records[key] = {
            "faq_id": "",
            "sections": [current_section],
            "question": current_question,
            "reference_answer": reference_answer,
            "answer_urls": answer_urls,
            "source_url": SOURCE_URL + fragment,
            "source_sha256": source_sha256,
            "duplicate_count": 1,
            "included_in_holdout": True,
            "exclusion_reason": None,
            "topics": [],
            "risk_tags": [],
        }

    relevant_tags = article.find_all(
        ["h2", "p", "ul", "ol"],
    )

    for element in relevant_tags:
        if element.name == "h2":
            save_current_question()
            current_question = None
            answer_nodes = []

            section = clean_text(
                element.get_text(" ", strip=True)
            )

            if (
                section == "Table of Contents"
                or section in EXCLUDED_SECTIONS
            ):
                current_section = None
                current_section_id = None
            else:
                current_section = section
                current_section_id = element.get("id")

            continue

        if current_section is None:
            continue

        strong = (
            element.find("strong", recursive=False)
            if element.name == "p"
            else None
        )
        strong_text = (
            clean_text(strong.get_text(" ", strip=True))
            if strong
            else ""
        )

        if strong_text.startswith("Q."):
            save_current_question()
            text = clean_text(
                element.get_text(" ", strip=True)
            )
            current_question = text.removeprefix("Q.").strip()
            answer_nodes = []
            continue

        if current_question is not None:
            answer_nodes.append(element)

    save_current_question()

    dataset = list(records.values())

    for number, record in enumerate(dataset, start=1):
        record["faq_id"] = f"nih_faq_{number:03d}"

    print(
        f"Parsed {included_occurrences} FAQ occurrences "
        f"into {len(dataset)} unique records."
    )
    print(f"Excluded sections: {len(EXCLUDED_SECTIONS)}")

    return dataset


def main() -> None:
    dataset = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in dataset:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
