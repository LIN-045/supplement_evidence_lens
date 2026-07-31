"""Persist application interactions and user feedback in SQLite."""

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(
    os.environ.get(
        "MONITORING_DATABASE_PATH",
        PROJECT_ROOT / "data/monitoring/interactions.db",
    )
)


def connect() -> sqlite3.Connection:
    """Open a connection with rows accessible by column name."""

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    """Create or migrate the interactions table without losing records."""

    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
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
                feedback_at TEXT,
                feedback_comment TEXT
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(interactions)"
            ).fetchall()
        }

        if "rag_version" not in columns:
            connection.execute(
                """
                ALTER TABLE interactions
                ADD COLUMN rag_version TEXT NOT NULL
                DEFAULT 'agentic_rag_v1'
                """
            )

        if "feedback_comment" not in columns:
            connection.execute(
                """
                ALTER TABLE interactions
                ADD COLUMN feedback_comment TEXT
                """
            )


def record_interaction(
    result: dict[str, Any],
    response_time_seconds: float,
    model_name: str,
    rag_version: str,
) -> str:
    """Save one completed RAG interaction and return its identifier."""

    initialize_database()
    interaction_id = str(uuid4())

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO interactions (
                interaction_id,
                created_at,
                question,
                answer,
                search_queries,
                contexts,
                search_count,
                context_count,
                response_time_seconds,
                model_name,
                rag_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                datetime.now(UTC).isoformat(),
                result["question"],
                result["answer"],
                json.dumps(
                    result["search_queries"],
                    ensure_ascii=False,
                ),
                json.dumps(
                    result["contexts"],
                    ensure_ascii=False,
                ),
                len(result["search_queries"]),
                len(result["contexts"]),
                response_time_seconds,
                model_name,
                rag_version,
            ),
        )

    return interaction_id


def record_feedback(
    interaction_id: str,
    feedback: int,
    feedback_comment: str | None = None,
) -> None:
    """Attach a rating and optional explanation to an interaction."""

    if feedback not in {-1, 1}:
        raise ValueError("Feedback must be 1 or -1")

    if feedback_comment is not None:
        feedback_comment = feedback_comment.strip() or None
        if feedback_comment is not None and len(feedback_comment) > 1000:
            raise ValueError(
                "Feedback comment must be 1000 characters or fewer"
            )

    initialize_database()

    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE interactions
            SET feedback = ?,
                feedback_at = ?,
                feedback_comment = COALESCE(?, feedback_comment)
            WHERE interaction_id = ?
            """,
            (
                feedback,
                datetime.now(UTC).isoformat(),
                feedback_comment,
                interaction_id,
            ),
        )

        if cursor.rowcount != 1:
            raise ValueError(
                f"Unknown interaction: {interaction_id}"
            )


def load_interactions() -> list[dict[str, Any]]:
    """Return all interactions in chronological order."""

    initialize_database()

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM interactions
            ORDER BY created_at
            """
        ).fetchall()

    records = []

    for row in rows:
        record = dict(row)
        record["search_queries"] = json.loads(
            record["search_queries"]
        )
        record["contexts"] = json.loads(record["contexts"])
        records.append(record)

    return records
