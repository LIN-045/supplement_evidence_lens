"""Embed document chunks and index them in Elasticsearch.

Run from the project root:

    uv run python -m ingestion.index_documents
"""

import argparse
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from ingestion.ingestion_io import read_jsonl


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "document_chunks.jsonl"
)

ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "supplement_evidence"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
BATCH_SIZE = 32


# Elasticsearch setup

def create_index(
    client: Elasticsearch,
    index_name: str,
) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    client.indices.create(
        index=index_name,
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
    """Embed all chunks, replace one index, and return a run summary."""

    if not index_name.strip():
        raise ValueError("index_name must not be empty")

    documents = read_jsonl(input_path)
    print(f"Loaded documents: {len(documents)}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        [document["content"] for document in documents],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()
    print(f"Created embeddings: {len(embeddings)}")

    client = Elasticsearch(elasticsearch_url)
    if not client.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {elasticsearch_url}"
        )

    create_index(client, index_name)

    actions = build_actions(documents, embeddings, index_name)
    indexed_count, errors = helpers.bulk(
        client.options(request_timeout=120),
        actions,
        chunk_size=200,
    )

    if errors:
        raise RuntimeError(f"Failed indexing operations: {len(errors)}")

    client.indices.refresh(index=index_name)

    stored_count = client.count(index=index_name)["count"]
    if stored_count != len(documents):
        raise ValueError(
            f"Expected {len(documents)} documents, found {stored_count}"
        )

    return {
        "document_count": len(documents),
        "embedding_count": len(embeddings),
        "indexed_count": indexed_count,
        "stored_count": stored_count,
        "input_path": str(input_path),
        "index_name": index_name,
        "elasticsearch_url": elasticsearch_url,
        "embedding_model": EMBEDDING_MODEL_NAME,
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

    print(f"Indexed documents: {result['indexed_count']}")
    print(f"Elasticsearch index: {result['index_name']}")


if __name__ == "__main__":
    main()
