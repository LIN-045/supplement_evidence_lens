"""Search the Elasticsearch index with BM25 and vector retrieval.

Run from the project root:

    uv run python app/retrieval.py "What are the risks of high-dose zinc?"
"""

import argparse
import os
from typing import Any

from elasticsearch import Elasticsearch
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

# Configuration

ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = os.getenv(
    "SUPPLEMENT_EVIDENCE_INDEX",
    "supplement_evidence",
)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CANDIDATE_COUNT = 20
RESULT_COUNT = 5
RRF_CONSTANT = 60

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Search

def bm25_search(
    client: Elasticsearch,
    query: str,
    *,
    index_name: str = INDEX_NAME,
) -> list[dict[str, Any]]:
    response = client.search(
        index=index_name,
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
    *,
    index_name: str = INDEX_NAME,
) -> list[dict[str, Any]]:
    query_embedding = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    response = client.search(
        index=index_name,
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
    limit: int = RESULT_COUNT,
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
    )[:limit]

def rerank_results(
    model: CrossEncoder,
    query: str,
    results: list[dict[str, Any]],
    limit: int = RESULT_COUNT,
) -> list[dict[str, Any]]:
    pairs = [
        (
            query,
            (
                f"{result['_source']['title']}\n"
                f"{result['_source']['content']}"
            ),
        )
        for result in results
    ]

    scores = model.predict(pairs)

    reranked_results = [
        {
            **result,
            "reranker_score": float(score),
        }
        for result, score in zip(
            results,
            scores,
            strict=True,
        )
    ]

    return sorted(
        reranked_results,
        key=lambda result: result["reranker_score"],
        reverse=True,
    )[:limit]

class EvidenceRetriever:
    """Provide hybrid retrieval through one reusable object."""

    def __init__(
        self,
        client: Elasticsearch,
        embedding_model: SentenceTransformer,
        reranker_model: CrossEncoder | None = None,
        index_name: str = INDEX_NAME,
    ) -> None:
        self.client = client
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index_name = index_name

    @classmethod
    def from_defaults(
        cls,
        *,
        use_reranker: bool = True,
    ) -> "EvidenceRetriever":
        """Create the default Elasticsearch client and search models."""

        client = Elasticsearch(ELASTICSEARCH_URL)
        if not client.ping():
            raise ConnectionError(
                f"Cannot connect to Elasticsearch at {ELASTICSEARCH_URL}"
            )

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        reranker_model = (
            CrossEncoder(RERANKER_MODEL_NAME)
            if use_reranker
            else None
        )

        return cls(
            client,
            embedding_model,
            reranker_model,
            INDEX_NAME,
        )

    def index_document_sha256(self) -> str:
        """Return the processed-data hash stored on the live index."""

        mappings = self.client.indices.get_mapping(
            index=self.index_name,
        )
        hashes = {
            index_mapping.get("mappings", {})
            .get("_meta", {})
            .get("document_sha256")
            for index_mapping in mappings.values()
        }

        if len(hashes) != 1 or None in hashes:
            raise ValueError(
                f"Index {self.index_name!r} has missing or "
                "inconsistent document_sha256 metadata. "
                "Rebuild it with ingestion.index_documents."
            )

        return next(iter(hashes))

    def bm25_search(self, query: str) -> list[dict[str, Any]]:
        """Return BM25 candidates for one query."""

        return bm25_search(
            self.client,
            query,
            index_name=self.index_name,
        )

    def vector_search(self, query: str) -> list[dict[str, Any]]:
        """Return vector candidates for one query."""

        return vector_search(
            self.client,
            self.embedding_model,
            query,
            index_name=self.index_name,
        )

    def hybrid_search(
        self,
        query: str,
        *,
        limit: int = RESULT_COUNT,
    ) -> list[dict[str, Any]]:
        """Fuse BM25 and vector candidates with reciprocal rank fusion."""

        return reciprocal_rank_fusion(
            [
                self.bm25_search(query),
                self.vector_search(query),
            ],
            limit=limit,
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        """Return hybrid evidence, reranked when a model is configured."""

        fused_results = self.hybrid_search(
            query,
            limit=(
                CANDIDATE_COUNT
                if self.reranker_model is not None
                else RESULT_COUNT
            ),
        )

        if self.reranker_model is None:
            return fused_results

        return rerank_results(
            self.reranker_model,
            query,
            fused_results,
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

    retriever = EvidenceRetriever.from_defaults(
        use_reranker=False,
    )
    results = retriever.search(arguments.query)

    for rank, result in enumerate(results, start=1):
        document = result["_source"]

        print(f"\n{rank}. {document['title']}")
        print(f"   Source: {document['source']}")
        print(f"   URL: {document['source_url']}")
        print(f"   Score: {result['rrf_score']:.6f}")
        print(f"   {document['content'][:500]}")


if __name__ == "__main__":
    main()
