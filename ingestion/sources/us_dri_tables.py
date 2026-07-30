"""Ingest the seven DRI summary tables linked by NIH ODS.

Run from the project root:

    uv run python -m ingestion.sources.us_dri_tables

The script intentionally ingests the compact summary tables, not the full
National Academies DRI reports and not the interactive DRI calculator.
"""

import re
import time
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
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "us_dri_tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = (
    PROCESSED_DIR / "sources" / "us_dri_reference_values.jsonl"
)
ERRORS_PATH = RAW_DIR / "download_errors.json"

ODS_PAGE_URL = (
    "https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx"
)
ODS_INDEX_PATH = RAW_DIR / "nutrient_recommendations.html"

REQUEST_DELAY_SECONDS = 0.5

TABLE_CONFIGS: tuple[dict[str, str], ...] = (
    {
        "table_id": "rda-ai-elements",
        "link_text": (
            "Recommended Dietary Allowances and Adequate Intakes, Elements"
        ),
        "reference_type": "RDA_OR_AI",
        "layout": "life_stage_rows",
    },
    {
        "table_id": "rda-ai-vitamins",
        "link_text": (
            "Recommended Dietary Allowances and Adequate Intakes, Vitamins"
        ),
        "reference_type": "RDA_OR_AI",
        "layout": "life_stage_rows",
    },
    {
        "table_id": "rda-ai-water-macronutrients",
        "link_text": (
            "Recommended Dietary Allowances and Adequate Intakes, "
            "Total Water and Macronutrients"
        ),
        "reference_type": "RDA_OR_AI",
        "layout": "life_stage_rows",
    },
    {
        "table_id": "ear",
        "link_text": "Estimated Average Requirements",
        "reference_type": "EAR",
        "layout": "life_stage_rows",
    },
    {
        "table_id": "amdr",
        "link_text": "Acceptable Macronutrient Distribution Ranges",
        "reference_type": "AMDR",
        "layout": "nutrient_rows",
    },
    {
        "table_id": "ul-vitamins",
        "link_text": "Tolerable Upper Intake Levels, Vitamins",
        "reference_type": "UL",
        "layout": "life_stage_rows",
    },
    {
        "table_id": "ul-elements",
        "link_text": "Tolerable Upper Intake Levels, Elements",
        "reference_type": "UL",
        "layout": "life_stage_rows",
    },
)

POPULATION_GROUPS = {
    "Infants",
    "Children",
    "Males",
    "Females",
    "Pregnancy",
    "Lactation",
}


# Table discovery and parsing

def clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "unknown"


def discover_table_urls(content: bytes) -> dict[str, str]:
    soup = BeautifulSoup(content, "html.parser")
    urls: dict[str, str] = {}

    for config in TABLE_CONFIGS:
        wanted = clean_text(config["link_text"])
        link = next(
            (
                candidate
                for candidate in soup.find_all("a", href=True)
                if clean_text(candidate.get_text(" ", strip=True)) == wanted
            ),
            None,
        )
        if link is None:
            raise ValueError(f"DRI table link not found: {wanted}")
        urls[config["table_id"]] = urljoin(
            ODS_PAGE_URL,
            str(link["href"]),
        )

    return urls


def cell_text(cell: Tag) -> str:
    clone = BeautifulSoup(str(cell), "html.parser")
    for element in clone.select("a[href^='#'], sup a"):
        element.decompose()
    return clean_text(clone.get_text(" ", strip=True))


def expand_rows(rows: list[Tag]) -> list[list[str]]:
    """Expand HTML rowspan and colspan cells into a rectangular grid."""
    grid: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}

    for row in rows:
        values: list[str] = []
        column = 0

        def fill_pending() -> None:
            nonlocal column
            while column in pending:
                value, remaining = pending[column]
                values.append(value)
                if remaining <= 1:
                    del pending[column]
                else:
                    pending[column] = (value, remaining - 1)
                column += 1

        fill_pending()
        for cell in row.find_all(["th", "td"], recursive=False):
            fill_pending()
            value = cell_text(cell)
            rowspan = int(str(cell.get("rowspan", "1")))
            colspan = int(str(cell.get("colspan", "1")))

            for _ in range(colspan):
                values.append(value)
                if rowspan > 1:
                    pending[column] = (value, rowspan - 1)
                column += 1
            fill_pending()

        grid.append(values)

    width = max((len(row) for row in grid), default=0)
    return [row + [""] * (width - len(row)) for row in grid]


def table_title(table: Tag) -> str:
    caption = table.find("caption")
    if caption is not None:
        return clean_text(caption.get_text(" ", strip=True))

    heading = table.find_previous(["h1", "h2", "h3"])
    if heading is not None:
        return clean_text(heading.get_text(" ", strip=True))

    return "Dietary Reference Intake summary table"


def combined_headers(table: Tag, width: int) -> list[str]:
    thead = table.find("thead")
    if thead is None:
        first_row = table.find("tr")
        if first_row is None:
            return []
        header_grid = expand_rows([first_row])
    else:
        header_grid = expand_rows(thead.find_all("tr", recursive=False))

    headers: list[str] = []
    for column in range(width):
        parts: list[str] = []
        for row in header_grid:
            if column >= len(row):
                continue
            value = row[column]
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        headers.append(clean_text(" ".join(parts)))

    return headers


def split_name_and_unit(header: str) -> tuple[str, str | None]:
    cleaned = clean_text(header)
    match = re.match(
        r"^(.*?)\s*\(([^()]*)\)\s*[,;]*\s*$",
        cleaned,
    )
    if not match:
        name = cleaned
        unit = None
    else:
        name = clean_text(match.group(1))
        unit = clean_text(match.group(2))

    name = re.sub(r"\b([A-Za-z])\s+(\d+)\b", r"\1\2", name)
    return name, unit


def normalise_value(value: str) -> str:
    return clean_text(value).replace(" *", "*")


def reference_type_for_value(base_type: str, value: str) -> str:
    if base_type != "RDA_OR_AI":
        return base_type
    return "AI" if "*" in value else "RDA"


def build_record(
    *,
    config: dict[str, str],
    title: str,
    source_url: str,
    nutrient: str,
    unit: str | None,
    population_group: str,
    life_stage: str,
    value: str,
    retrieved_at: str,
) -> dict[str, Any]:
    reference_type = reference_type_for_value(
        config["reference_type"],
        value,
    )
    clean_value = normalise_value(value).rstrip("*").strip()
    established = clean_value.upper() not in {"ND", "N/A", "—", "-"}

    if established:
        unit_phrase = f" {unit}" if unit else ""
        content = (
            f"The {reference_type} for {nutrient} for "
            f"{population_group}, {life_stage}, is "
            f"{clean_value}{unit_phrase}."
        )
    else:
        content = (
            f"No {reference_type} has been determined for {nutrient} for "
            f"{population_group}, {life_stage}."
        )

    identity = ":".join(
        (
            config["table_id"],
            reference_type,
            nutrient,
            population_group,
            life_stage,
        )
    )
    return {
        "document_id": f"dri:{slug(identity)}",
        "table_id": config["table_id"],
        "name": title,
        "section_title": reference_type,
        "nutrient": nutrient,
        "reference_type": reference_type,
        "population_group": population_group,
        "life_stage": life_stage,
        "value": clean_value,
        "unit": unit,
        "value_established": established,
        "text": content,
        "source": "us_dri_tables",
        "publisher": (
            "National Academies of Sciences, Engineering, and Medicine"
        ),
        "jurisdiction": "US",
        "document_type": "nutrient_reference_value",
        "evidence_role": "nutrient_reference",
        "updated_date": None,
        "retrieved_at": retrieved_at,
        "source_url": source_url,
    }


def parse_life_stage_rows(
    table: Tag,
    config: dict[str, str],
    source_url: str,
    retrieved_at: str,
) -> list[dict[str, Any]]:
    body = table.find("tbody")
    rows = (
        body.find_all("tr", recursive=False)
        if body is not None
        else table.find_all("tr", recursive=False)[1:]
    )
    grid = expand_rows(rows)
    if not grid:
        return []

    headers = combined_headers(table, len(grid[0]))
    if len(headers) < 2:
        raise ValueError(f"Could not parse headers for {config['table_id']}")

    title = table_title(table)
    population_group = "General"
    records: list[dict[str, Any]] = []

    for row in grid:
        first = clean_text(row[0])
        remaining = [normalise_value(value) for value in row[1:]]

        if first in POPULATION_GROUPS:
            population_group = first
            continue
        if first and not any(remaining):
            continue
        if not first:
            continue

        for column, value in enumerate(remaining, start=1):
            if not value or column >= len(headers):
                continue
            nutrient, unit = split_name_and_unit(headers[column])
            if not nutrient:
                continue
            records.append(
                build_record(
                    config=config,
                    title=title,
                    source_url=source_url,
                    nutrient=nutrient,
                    unit=unit,
                    population_group=population_group,
                    life_stage=first,
                    value=value,
                    retrieved_at=retrieved_at,
                )
            )

    return records


def parse_nutrient_rows(
    table: Tag,
    config: dict[str, str],
    source_url: str,
    retrieved_at: str,
) -> list[dict[str, Any]]:
    body = table.find("tbody")
    rows = (
        body.find_all("tr", recursive=False)
        if body is not None
        else table.find_all("tr", recursive=False)[1:]
    )
    grid = expand_rows(rows)
    if not grid:
        return []

    headers = combined_headers(table, len(grid[0]))
    title = table_title(table)
    records: list[dict[str, Any]] = []

    for row in grid:
        nutrient = clean_text(row[0])
        if not nutrient:
            continue

        for column, value in enumerate(row[1:], start=1):
            value = normalise_value(value)
            if not value or column >= len(headers):
                continue
            life_stage = re.sub(
                r"^Range \(percent of energy\)\s*",
                "",
                headers[column],
                flags=re.IGNORECASE,
            )
            records.append(
                build_record(
                    config=config,
                    title=title,
                    source_url=source_url,
                    nutrient=nutrient,
                    unit="percent of energy",
                    population_group="General",
                    life_stage=life_stage,
                    value=value,
                    retrieved_at=retrieved_at,
                )
            )

    return records


def parse_table(
    content: bytes,
    config: dict[str, str],
    source_url: str,
    retrieved_at: str,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(content, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError(f"No HTML table found for {source_url}")

    if config["layout"] == "nutrient_rows":
        records = parse_nutrient_rows(
            table,
            config,
            source_url,
            retrieved_at,
        )
    else:
        records = parse_life_stage_rows(
            table,
            config,
            source_url,
            retrieved_at,
        )

    if not records:
        raise ValueError(f"No DRI values parsed for {config['table_id']}")
    return records


def validate_records(
    records: list[dict[str, Any]],
    expected_table_ids: set[str],
) -> None:
    if not records:
        raise ValueError("No DRI reference values were parsed")

    actual_table_ids = {record["table_id"] for record in records}
    if actual_table_ids != expected_table_ids:
        missing = sorted(expected_table_ids - actual_table_ids)
        raise ValueError(f"Missing parsed DRI tables: {missing}")

    document_ids = [record["document_id"] for record in records]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate DRI reference-value document IDs found")

    for record in records:
        for field in (
            "document_id",
            "nutrient",
            "reference_type",
            "life_stage",
            "text",
            "source_url",
            "retrieved_at",
        ):
            if not record.get(field):
                raise ValueError(
                    f"{record.get('document_id')} is missing {field}"
                )


# Execution

def run() -> dict[str, Any]:
    """Ingest the seven DRI summary tables and return a summary."""

    retrieved_at = datetime.now(UTC).isoformat()
    index_content = download_bytes(ODS_PAGE_URL)
    atomic_write(ODS_INDEX_PATH, index_content)
    table_urls = discover_table_urls(index_content)

    all_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for position, config in enumerate(TABLE_CONFIGS, start=1):
        table_id = config["table_id"]
        source_url = table_urls[table_id]
        raw_path = RAW_DIR / f"{table_id}.html"

        try:
            content = download_bytes(source_url)
            atomic_write(raw_path, content)
            records = parse_table(
                content,
                config,
                source_url,
                retrieved_at,
            )
            all_records.extend(records)
            print(
                f"[{position}/{len(TABLE_CONFIGS)}] "
                f"{table_id}: {len(records)} values"
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
                    "table_id": table_id,
                    "url": source_url,
                    "error": str(error),
                }
            )
            print(f"Failed {table_id}: {error}")

    if errors:
        write_json(ERRORS_PATH, errors)
        raise RuntimeError(f"Failed DRI tables: {len(errors)}")

    validate_records(
        all_records,
        {config["table_id"] for config in TABLE_CONFIGS},
    )
    write_jsonl(OUTPUT_PATH, all_records)
    write_json(ERRORS_PATH, [])

    return {
        "source": "us_dri_tables",
        "record_count": len(all_records),
        "table_count": len(TABLE_CONFIGS),
        "download_error_count": len(errors),
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "errors_path": str(ERRORS_PATH.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    result = run()

    print(f"Parsed values: {result['record_count']}")
    print(f"Wrote {result['output_path']}")


if __name__ == "__main__":
    main()
