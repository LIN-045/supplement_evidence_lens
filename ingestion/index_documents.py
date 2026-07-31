"""Embed document chunks and index them in Elasticsearch.

Run from the project root:

    uv run python -m ingestion.index_documents
"""

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from app.retrieval import (
    ELASTICSEARCH_URL,
    EMBEDDING_MODEL_NAME,
    INDEX_NAME,
)
from ingestion.ingestion_io import read_jsonl


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "document_chunks.jsonl"
)

EMBEDDING_DIMENSIONS = 384
BATCH_SIZE = 32

logger = logging.getLogger(__name__)


# Elasticsearch setup

def create_index(
    client: Elasticsearch,
    index_name: str,
    document_sha256: str,
) -> None:
    client.indices.create(
        index=index_name,
        mappings={
            "_meta": {
                "document_sha256": document_sha256,
                "embedding_model": EMBEDDING_MODEL_NAME,
            },
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "english",
                },
                "title": {
                    "type": "text",
                    "analyzer": "english",
                },
                "question": {
                    "type": "text",
                    "analyzer": "english",
                },
                "source": {
                    "type": "keyword",
                },
                "source_document_id": {
                    "type": "keyword",
                },
                "jurisdiction": {
                    "type": "keyword",
                },
                "publisher": {
                    "type": "keyword",
                },
                "document_type": {
                    "type": "keyword",
                },
                "evidence_role": {
                    "type": "keyword",
                },
                "section_title": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                        },
                    },
                },
                "updated_date": {
                    "type": "keyword",
                },
                "retrieved_at": {
                    "type": "date",
                },
                "nutrient": {
                    "type": "keyword",
                },
                "reference_type": {
                    "type": "keyword",
                },
                "population_group": {
                    "type": "keyword",
                },
                "life_stage": {
                    "type": "keyword",
                },
                "value": {
                    "type": "keyword",
                },
                "unit": {
                    "type": "keyword",
                },
                "value_established": {
                    "type": "boolean",
                },
                "source_url": {
                    "type": "keyword",
                    "index": False,
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it at once."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def switch_alias(
    client: Elasticsearch,
    alias_name: str,
    new_index_name: str,
) -> list[str]:
    """Atomically point the public alias at a completed physical index."""

    old_indices: list[str] = []
    actions: list[dict[str, Any]] = []

    if client.indices.exists_alias(name=alias_name):
        aliases = client.indices.get_alias(name=alias_name)
        old_indices = list(aliases)
        actions.extend(
            {
                "remove": {
                    "index": old_index,
                    "alias": alias_name,
                }
            }
            for old_index in old_indices
        )
    elif client.indices.exists(index=alias_name):
        # The first safe rebuild may replace the old concrete index with
        # an alias. remove_index and add happen in one cluster-state update.
        old_indices = [alias_name]
        actions.append(
            {"remove_index": {"index": alias_name}}
        )

    actions.append(
        {
            "add": {
                "index": new_index_name,
                "alias": alias_name,
                "is_write_index": True,
            }
        }
    )
    client.indices.update_aliases(actions=actions)
    return [
        index_name
        for index_name in old_indices
        if index_name != alias_name
    ]


def matching_physical_index(
    client: Elasticsearch,
    alias_name: str,
    document_sha256: str,
    expected_count: int,
) -> str | None:
    """Return the live physical index when its input is unchanged."""

    if not client.indices.exists_alias(name=alias_name):
        return None

    aliases = client.indices.get_alias(name=alias_name)

    for physical_index_name in aliases:
        mapping = client.indices.get_mapping(
            index=physical_index_name
        )[physical_index_name]
        metadata = mapping.get("mappings", {}).get("_meta", {})

        if (
            metadata.get("document_sha256") == document_sha256
            and metadata.get("embedding_model")
            == EMBEDDING_MODEL_NAME
            and client.count(index=physical_index_name)["count"]
            == expected_count
        ):
            return physical_index_name

    return None


# Embedding and indexing

def build_actions(
    documents: list[dict[str, Any]],
    embeddings: list[list[float]],
    index_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            "_index": index_name,
            "_id": document["document_id"],
            "_source": {
                **document,
                "embedding": embedding,
            },
        }
        for document, embedding in zip(
            documents,
            embeddings,
            strict=True,
        )
    ]


def run(
    *,
    input_path: Path = INPUT_PATH,
    index_name: str = INDEX_NAME,
    elasticsearch_url: str = ELASTICSEARCH_URL,
) -> dict[str, Any]:
    """Embed chunks, validate a new index, then atomically switch its alias."""

    if not index_name.strip():
        raise ValueError("index_name must not be empty")

    documents = read_jsonl(input_path)
    if not documents:
        raise ValueError(f"No documents found in {input_path}")

    document_sha256 = file_sha256(input_path)
    print(f"Loaded documents: {len(documents)}")

    client = Elasticsearch(elasticsearch_url)
    if not client.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {elasticsearch_url}"
        )

    current_index_name = matching_physical_index(
        client,
        index_name,
        document_sha256,
        len(documents),
    )
    if current_index_name is not None:
        print(
            f"Index is already current: {current_index_name}"
        )
        return {
            "document_count": len(documents),
            "embedding_count": 0,
            "indexed_count": 0,
            "stored_count": len(documents),
            "input_path": str(input_path),
            "index_name": index_name,
            "physical_index_name": current_index_name,
            "document_sha256": document_sha256,
            "elasticsearch_url": elasticsearch_url,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "skipped": True,
        }

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        [document["content"] for document in documents],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()
    print(f"Created embeddings: {len(embeddings)}")

    physical_index_name = (
        f"{index_name}-{uuid4().hex[:12]}"
    )
    create_index(
        client,
        physical_index_name,
        document_sha256,
    )

    alias_switched = False

    try:
        actions = build_actions(
            documents,
            embeddings,
            physical_index_name,
        )
        indexed_count, errors = helpers.bulk(
            client.options(request_timeout=120),
            actions,
            chunk_size=200,
        )

        if errors:
            raise RuntimeError(
                f"Failed indexing operations: {len(errors)}"
            )

        client.indices.refresh(index=physical_index_name)

        stored_count = client.count(
            index=physical_index_name
        )["count"]
        if stored_count != len(documents):
            raise ValueError(
                f"Expected {len(documents)} documents, "
                f"found {stored_count}"
            )

        old_physical_indices = switch_alias(
            client,
            index_name,
            physical_index_name,
        )
        alias_switched = True

        for old_index in old_physical_indices:
            try:
                client.indices.delete(index=old_index)
            except Exception as error:
                logger.warning(
                    "The new index is live, but old index %s "
                    "could not be deleted: %s",
                    old_index,
                    error,
                )
    except Exception:
        if not alias_switched:
            client.indices.delete(
                index=physical_index_name,
                ignore_unavailable=True,
            )
        raise

    return {
        "document_count": len(documents),
        "embedding_count": len(embeddings),
        "indexed_count": indexed_count,
        "stored_count": stored_count,
        "input_path": str(input_path),
        "index_name": index_name,
        "physical_index_name": physical_index_name,
        "document_sha256": document_sha256,
        "elasticsearch_url": elasticsearch_url,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "skipped": False,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunks and rebuild an Elasticsearch index."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--index-name", default=INDEX_NAME)
    parser.add_argument(
        "--elasticsearch-url",
        default=ELASTICSEARCH_URL,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result = run(
        input_path=arguments.input,
        index_name=arguments.index_name,
        elasticsearch_url=arguments.elasticsearch_url,
    )

    if result["skipped"]:
        print("Skipped embedding and indexing; input is unchanged")
    else:
        print(f"Indexed documents: {result['indexed_count']}")
    print(f"Elasticsearch index: {result['index_name']}")


if __name__ == "__main__":
    main()
