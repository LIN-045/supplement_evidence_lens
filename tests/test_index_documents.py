import json
from pathlib import Path
from typing import Any

import pytest

import ingestion.index_documents as index_documents
from ingestion.index_documents import switch_alias


class FakeIndices:
    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] = []
        self.created: list[str] = []
        self.deleted: list[str] = []

    def create(
        self,
        *,
        index: str,
        mappings: dict[str, Any],
    ) -> None:
        self.created.append(index)

    def delete(
        self,
        *,
        index: str,
        ignore_unavailable: bool = False,
    ) -> None:
        self.deleted.append(index)

    def refresh(self, *, index: str) -> None:
        pass

    def exists_alias(self, *, name: str) -> bool:
        return True

    def get_alias(self, *, name: str) -> dict[str, dict]:
        return {
            "supplement_evidence-old": {},
        }

    def exists(self, *, index: str) -> bool:
        return False

    def update_aliases(
        self,
        *,
        actions: list[dict[str, Any]],
    ) -> None:
        self.actions = actions


class FakeClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()

    def ping(self) -> bool:
        return True

    def options(self, **kwargs: Any) -> "FakeClient":
        return self

    def count(self, *, index: str) -> dict[str, int]:
        return {"count": 1}


def test_switch_alias_replaces_existing_physical_index() -> None:
    client = FakeClient()

    old_indices = switch_alias(
        client,
        alias_name="supplement_evidence",
        new_index_name="supplement_evidence-new",
    )

    assert old_indices == ["supplement_evidence-old"]
    assert client.indices.actions == [
        {
            "remove": {
                "index": "supplement_evidence-old",
                "alias": "supplement_evidence",
            }
        },
        {
            "add": {
                "index": "supplement_evidence-new",
                "alias": "supplement_evidence",
                "is_write_index": True,
            }
        },
    ]


def test_failed_index_build_keeps_existing_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "document_chunks.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "document-1::chunk-1",
                "content": "Example content",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeEmbeddings(list):
        def tolist(self) -> list[list[float]]:
            return list(self)

    class FakeModel:
        def encode(
            self,
            texts: list[str],
            **kwargs: Any,
        ) -> FakeEmbeddings:
            return FakeEmbeddings([[0.0] * 384])

    client = FakeClient()

    monkeypatch.setattr(
        index_documents,
        "SentenceTransformer",
        lambda model_name: FakeModel(),
    )
    monkeypatch.setattr(
        index_documents,
        "Elasticsearch",
        lambda url: client,
    )

    def fail_bulk(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated indexing failure")

    monkeypatch.setattr(
        index_documents.helpers,
        "bulk",
        fail_bulk,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated indexing failure",
    ):
        index_documents.run(input_path=input_path)

    assert len(client.indices.created) == 1
    assert client.indices.actions == []
    assert client.indices.deleted == client.indices.created


def test_successful_index_build_switches_alias_and_deletes_old_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "document_chunks.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "document_id": "document-1::chunk-1",
                "content": "Example content",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeEmbeddings(list):
        def tolist(self) -> list[list[float]]:
            return list(self)

    class FakeModel:
        def encode(
            self,
            texts: list[str],
            **kwargs: Any,
        ) -> FakeEmbeddings:
            return FakeEmbeddings([[0.0] * 384])

    client = FakeClient()
    monkeypatch.setattr(
        index_documents,
        "SentenceTransformer",
        lambda model_name: FakeModel(),
    )
    monkeypatch.setattr(
        index_documents,
        "Elasticsearch",
        lambda url: client,
    )
    monkeypatch.setattr(
        index_documents.helpers,
        "bulk",
        lambda *args, **kwargs: (1, []),
    )

    result = index_documents.run(input_path=input_path)

    assert result["stored_count"] == 1
    assert client.indices.actions[-1]["add"]["alias"] == (
        "supplement_evidence"
    )
    assert client.indices.deleted == ["supplement_evidence-old"]
