"""Embed document chunks and index them in Elasticsearch.

Run from the project root:

    uv run python ingestion/index_documents.py
"""

import json
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"

ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "supplement_evidence"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
BATCH_SIZE = 32


# File handling

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# Elasticsearch setup

def create_index(client: Elasticsearch) -> None:
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)

    client.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "english",
                },
                "title": {
                    "type": "text",
                    "analyzer": "english",
                },
                "source": {
                    "type": "keyword",
                },
                "jurisdiction": {
                    "type": "keyword",
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


# Embedding and indexing

def build_actions(
    documents: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    return [
        {
            "_index": INDEX_NAME,
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


def main() -> None:
    documents = read_jsonl(INPUT_PATH)
    print(f"Loaded documents: {len(documents)}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [document["content"] for document in documents],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()
    print(f"Created embeddings: {len(embeddings)}")

    client = Elasticsearch(ELASTICSEARCH_URL)
    if not client.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {ELASTICSEARCH_URL}"
        )

    create_index(client)

    actions = build_actions(documents, embeddings)
    indexed_count, errors = helpers.bulk(
        client,
        actions,
        chunk_size=200,
        request_timeout=120,
    )

    if errors:
        raise RuntimeError(f"Failed indexing operations: {len(errors)}")

    client.indices.refresh(index=INDEX_NAME)

    stored_count = client.count(index=INDEX_NAME)["count"]
    if stored_count != len(documents):
        raise ValueError(
            f"Expected {len(documents)} documents, found {stored_count}"
        )

    print(f"Indexed documents: {indexed_count}")
    print(f"Elasticsearch index: {INDEX_NAME}")


if __name__ == "__main__":
    main()
