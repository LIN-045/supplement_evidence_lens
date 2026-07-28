"""Search the Elasticsearch index with BM25 and vector retrieval.

Run from the project root:

    uv run python app/retrieval.py "What are the risks of high-dose zinc?"
"""

import argparse
from typing import Any

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


# Configuration

ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "supplement_evidence"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CANDIDATE_COUNT = 20
RESULT_COUNT = 5
RRF_CONSTANT = 60


# Search

def bm25_search(
    client: Elasticsearch,
    query: str,
) -> list[dict[str, Any]]:
    response = client.search(
        index=INDEX_NAME,
        size=CANDIDATE_COUNT,
        query={
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"],
            }
        },
        source_excludes=["embedding"],
    )
    return response["hits"]["hits"]


def vector_search(
    client: Elasticsearch,
    model: SentenceTransformer,
    query: str,
) -> list[dict[str, Any]]:
    query_embedding = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    response = client.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": CANDIDATE_COUNT,
            "num_candidates": 100,
        },
        source_excludes=["embedding"],
    )
    return response["hits"]["hits"]


# Result fusion

def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    fused_results: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            document_id = result["_id"]

            if document_id not in fused_results:
                fused_results[document_id] = {
                    **result,
                    "rrf_score": 0.0,
                }

            fused_results[document_id]["rrf_score"] += (
                1 / (RRF_CONSTANT + rank)
            )

    return sorted(
        fused_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )[:RESULT_COUNT]


def search(
    client: Elasticsearch,
    model: SentenceTransformer,
    query: str,
) -> list[dict[str, Any]]:
    bm25_results = bm25_search(client, query)
    vector_results = vector_search(client, model, query)

    return reciprocal_rank_fusion(
        [bm25_results, vector_results]
    )


# Command-line interface

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search official supplement evidence sources."
    )
    parser.add_argument(
        "query",
        help="Question or search query",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    client = Elasticsearch(ELASTICSEARCH_URL)
    if not client.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {ELASTICSEARCH_URL}"
        )

    model = SentenceTransformer(MODEL_NAME)
    results = search(client, model, arguments.query)

    for rank, result in enumerate(results, start=1):
        document = result["_source"]

        print(f"\n{rank}. {document['title']}")
        print(f"   Source: {document['source']}")
        print(f"   URL: {document['source_url']}")
        print(f"   Score: {result['rrf_score']:.6f}")
        print(f"   {document['content'][:500]}")


if __name__ == "__main__":
    main()