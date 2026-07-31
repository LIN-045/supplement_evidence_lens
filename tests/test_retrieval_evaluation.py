import json
from pathlib import Path

import pytest

import evaluation.retrieval.calculate_retrieval_eval_metrics as metrics


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def build_pool_record() -> dict:
    record = {
        "question_id": "question-1",
        "question": "Example question?",
        "source": "example_source",
        "seed_document_id": "parent-document-1",
        "index_document_sha256": "index-hash",
        "candidates": [
            {
                "document_id": "parent-document-1::chunk-1",
                "source_document_id": "parent-document-1",
                "title": "Example",
                "source": "example_source",
                "content": "Example content",
                "retrieved_by": {"hybrid": 1},
            }
        ],
    }
    return {
        **record,
        "pool_hash": metrics.record_sha256(record),
    }


def test_retrieval_evaluation_uses_relevant_chunk_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_path = tmp_path / "pool.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    pooled_record = build_pool_record()

    write_jsonl(pool_path, [pooled_record])
    write_jsonl(
        judgments_path,
        [
            {
                "question_id": "question-1",
                "pool_hash": pooled_record["pool_hash"],
                "judge_version": "relevance_judge_v1",
                "judge_model": "test-model",
                "judgments": [
                    {
                        "document_id": (
                            "parent-document-1::chunk-1"
                        ),
                        "source_document_id": "parent-document-1",
                        "relevance": "relevant",
                        "reason": "Contains the answer.",
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(metrics, "POOL_PATH", pool_path)
    monkeypatch.setattr(
        metrics,
        "JUDGMENTS_PATH",
        judgments_path,
    )

    records = metrics.build_evaluation_records()

    assert records[0]["relevant_chunk_ids"] == {
        "parent-document-1::chunk-1",
    }
    assert metrics.ranked_chunk_ids(
        records[0],
        "hybrid",
    ) == ["parent-document-1::chunk-1"]


def test_zero_relevant_pool_chunks_score_zero() -> None:
    record = {
        "candidates": [],
        "relevant_chunk_ids": set(),
    }

    assert metrics.calculate_metrics(
        [record],
        "bm25",
    ) == {
        "hit_rate@5": 0.0,
        "mrr@5": 0.0,
        "pooled_recall@5": 0.0,
    }


def test_retrieval_evaluation_rejects_changed_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_path = tmp_path / "pool.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    pooled_record = build_pool_record()

    write_jsonl(pool_path, [pooled_record])
    write_jsonl(
        judgments_path,
        [
            {
                "question_id": "question-1",
                "pool_hash": "old-pool-hash",
                "judge_version": "relevance_judge_v1",
                "judge_model": "test-model",
                "judgments": [],
            }
        ],
    )
    monkeypatch.setattr(metrics, "POOL_PATH", pool_path)
    monkeypatch.setattr(
        metrics,
        "JUDGMENTS_PATH",
        judgments_path,
    )

    with pytest.raises(
        ValueError,
        match="Judgments use a different pool",
    ):
        metrics.build_evaluation_records()
