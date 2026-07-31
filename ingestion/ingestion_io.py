"""Network and file I/O shared by ingestion source adapters."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_USER_AGENT = "SupplementEvidenceLens/0.1"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_MAX_ATTEMPTS = 3


def download_bytes(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> bytes:
    """Download one URL with the ingestion retry policy."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    request_headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        **(headers or {}),
    }
    request = Request(url, headers=request_headers)

    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError):
            if attempt == max_attempts:
                raise
            time.sleep(attempt)

    raise RuntimeError(f"Failed to download {url}")


def download_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[bytes, Any]:
    """Download a JSON response and retain both raw and parsed forms."""

    content = download_bytes(
        url,
        headers={"Accept": "application/json", **(headers or {})},
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )

    try:
        return content, json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Expected JSON from {url}") from error


def atomic_write(path: Path, content: bytes) -> None:
    """Replace a file atomically after writing it beside the target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def serialise_json(
    value: Any,
    *,
    indent: int | None = 2,
) -> bytes:
    """Serialize one JSON value as deterministic UTF-8 bytes."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_json(
    path: Path,
    value: Any,
    *,
    indent: int | None = 2,
) -> None:
    """Serialize and atomically replace a JSON file."""

    atomic_write(path, serialise_json(value, indent=indent))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records and require JSON objects."""

    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line "
                    f"{line_number}"
                )

            records.append(record)

    return records


def serialise_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize records as deterministic UTF-8 JSON Lines."""

    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Serialize records and atomically replace a JSONL file."""

    atomic_write(path, serialise_jsonl(records))
