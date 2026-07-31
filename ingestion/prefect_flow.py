"""Orchestrate the ingestion pipeline with Prefect."""

from collections.abc import Callable
from typing import Any

from prefect import flow, task

from ingestion.sources import (
    ca_nhpid_monographs,
    eu_health_claims,
    us_dri_tables,
    us_nccih_herbs,
    us_nih_ods,
    us_ods_faq,
    us_ods_guidance,
)

from ingestion import chunk_documents, index_documents

SOURCE_RUNNERS: dict[str, Callable[[], dict[str, Any]]] = {
    "eu_health_claims": eu_health_claims.run,
    "ca_nhpid_monographs": ca_nhpid_monographs.run,
    "us_nih_ods": us_nih_ods.run,
    "us_ods_guidance": us_ods_guidance.run,
    "us_dri_tables": us_dri_tables.run,
    "us_nccih_herbs": us_nccih_herbs.run,
    "us_ods_faq": us_ods_faq.run,
}


@task(task_run_name="ingest-{source_name}")
def ingest_source(source_name: str) -> dict[str, Any]:
    """Run one source adapter."""

    return SOURCE_RUNNERS[source_name]()

@task
def build_chunks() -> dict[str, Any]:
    """Build searchable document chunks."""

    return chunk_documents.run()


@task
def build_index() -> dict[str, Any]:
    """Embed chunks and rebuild the Elasticsearch index."""

    return index_documents.run()

@flow(name="supplement-evidence-ingestion", log_prints=True)
def run_pipeline() -> dict[str, Any]:
    """Run the complete ingestion pipeline."""

    # These task calls are intentionally synchronous: every source must finish
    # writing its processed file before chunks are built, and chunking must
    # finish before indexing starts. If source ingestion is changed to use
    # .submit() in the future, preserve this ordering explicitly with wait_for.
    source_results = {
        source_name: ingest_source(source_name)
        for source_name in SOURCE_RUNNERS
    }
    chunk_result = build_chunks()
    index_result = build_index()

    return {
        "sources": source_results,
        "chunks": chunk_result,
        "index": index_result,
    }


if __name__ == "__main__":
    run_pipeline()
