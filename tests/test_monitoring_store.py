import json
import sqlite3
from pathlib import Path

import pytest

import monitoring.store as store


def test_feedback_comment_migration_preserves_existing_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "interactions.db"
    monkeypatch.setattr(store, "DATABASE_PATH", database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE interactions (
                interaction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                search_queries TEXT NOT NULL,
                contexts TEXT NOT NULL,
                search_count INTEGER NOT NULL,
                context_count INTEGER NOT NULL,
                response_time_seconds REAL NOT NULL,
                model_name TEXT NOT NULL,
                rag_version TEXT NOT NULL,
                feedback INTEGER,
                feedback_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO interactions VALUES (
                'interaction-1', '2026-07-31T00:00:00+00:00',
                'Question?', 'Answer.', ?, ?, 1, 1, 1.0,
                'test-model', 'agentic_rag_v4', -1,
                '2026-07-31T00:01:00+00:00'
            )
            """,
            (json.dumps(["query"]), json.dumps([{"reference": 1}])),
        )

    store.initialize_database()
    records = store.load_interactions()

    assert len(records) == 1
    assert records[0]["feedback"] == -1
    assert records[0]["feedback_comment"] is None

    store.record_feedback(
        "interaction-1",
        -1,
        "  The citation does not support the protein estimate.  ",
    )

    updated_record = store.load_interactions()[0]
    assert updated_record["feedback"] == -1
    assert updated_record["feedback_comment"] == (
        "The citation does not support the protein estimate."
    )

    store.record_feedback("interaction-1", 1)

    changed_rating = store.load_interactions()[0]
    assert changed_rating["feedback"] == 1
    assert changed_rating["feedback_comment"] == (
        "The citation does not support the protein estimate."
    )

    store.record_feedback(
        "interaction-1",
        1,
        "The revised answer was useful.",
    )

    replaced_comment = store.load_interactions()[0]
    assert replaced_comment["feedback_comment"] == (
        "The revised answer was useful."
    )
