import json
from types import SimpleNamespace

from app.agentic_rag import AgenticRAG
from app.base_rag import ANSWER_RULES, BaseRAG


EVIDENCE_ROLE = "clinical_evidence_summary"


def test_overlapping_chunk_content_is_merged_without_repetition() -> None:
    first_part = "A" * 100
    overlap = "shared evidence " * 40
    second_part = "B" * 100

    first_chunk = first_part + overlap
    second_chunk = overlap + second_part

    merged = BaseRAG._merge_overlapping_content(
        first_chunk,
        second_chunk,
    )

    assert merged == first_part + overlap + second_part
    assert merged.count(overlap) == 1



def test_citations_from_overlapping_chunks_are_compacted() -> None:
    first_part = "A" * 100
    overlap = "shared evidence " * 40
    second_part = "B" * 100

    references = [
        {
            "_id": "document-1::chunk-1",
            "_source": {
                "source_document_id": "document-1",
                "title": "Example document",
                "source": "example_source",
                "jurisdiction": "US",
                "evidence_role": EVIDENCE_ROLE,
                "source_url": "https://example.com/document#first",
                "content": first_part + overlap,
            },
        },
        {
            "_id": "document-1::chunk-2",
            "_source": {
                "source_document_id": "document-1",
                "title": "Example document",
                "source": "example_source",
                "jurisdiction": "US",
                "evidence_role": EVIDENCE_ROLE,
                "source_url": "https://example.com/document#second",
                "content": overlap + second_part,
            },
        },
    ]

    answer, cited_references = BaseRAG._compact_citations(
        "The evidence supports this statement [1, 2].",
        references,
    )

    assert answer == "The evidence supports this statement [1]."
    assert len(cited_references) == 1
    assert (
        cited_references[0]["_source"]["content"]
        == first_part + overlap + second_part
    )


def test_citation_ranges_are_expanded_and_preserved() -> None:
    references = [
        {
            "_id": f"document-{number}::chunk-1",
            "_source": {
                "source_document_id": f"document-{number}",
                "title": f"Document {number}",
                "source": "example_source",
                "jurisdiction": "US",
                "evidence_role": EVIDENCE_ROLE,
                "source_url": f"https://example.com/{number}",
                "content": f"Evidence {number}.",
            },
        }
        for number in range(1, 6)
    ]

    for separator in ("-", "–", "—"):
        answer, cited_references = BaseRAG._compact_citations(
            f"All five references support this [1{separator}5].",
            references,
        )

        assert answer == (
            "All five references support this [1, 2, 3, 4, 5]."
        )
        assert len(cited_references) == 5


def test_mixed_citation_numbers_and_ranges_are_expanded() -> None:
    assert BaseRAG._citation_numbers("1, 3-5") == [
        1,
        3,
        4,
        5,
    ]


def test_invalid_citation_is_removed_without_empty_brackets() -> None:
    answer, cited_references = BaseRAG._compact_citations(
        "The available excerpts do not support this [99].",
        [],
    )

    assert answer == "The available excerpts do not support this."
    assert cited_references == []


def test_base_rag_always_returns_structured_result() -> None:
    class FakeRetriever:
        def search(self, query: str) -> list[dict]:
            return [
                {
                    "_id": "document-1::chunk-1",
                    "_source": {
                        "source_document_id": "document-1",
                        "title": "Example document",
                        "source": "example_source",
                        "jurisdiction": "US",
                        "evidence_role": EVIDENCE_ROLE,
                        "source_url": "https://example.com/document",
                        "content": "Example evidence.",
                    },
                }
            ]

    class FakeResponses:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                output_text="Example answer [1]."
            )

    fake_openai_client = SimpleNamespace(
        responses=FakeResponses()
    )
    rag = BaseRAG(
        FakeRetriever(),
        fake_openai_client,
        model_name="test-model",
    )

    result = rag.answer("Example question?")

    assert result == {
        "question": "Example question?",
        "answer": "Example answer [1].",
        "search_queries": ["Example question?"],
        "contexts": [
            {
                "reference": 1,
                "document_id": "document-1::chunk-1",
                "source_document_id": "document-1",
                "title": "Example document",
                "source": "example_source",
                "jurisdiction": "US",
                "evidence_role": EVIDENCE_ROLE,
                "source_url": "https://example.com/document",
                "content": "Example evidence.",
            }
        ],
    }


def test_evidence_role_is_sent_to_the_llm_without_quality_weighting() -> None:
    class FakeRetriever:
        def search(self, query: str) -> list[dict]:
            return [
                {
                    "_id": "document-1::chunk-1",
                    "_source": {
                        "source_document_id": "document-1",
                        "title": "Example document",
                        "source": "example_source",
                        "jurisdiction": "EU",
                        "evidence_role": "regulatory_claim",
                        "source_url": "https://example.com/document",
                        "content": "Example regulatory evidence.",
                    },
                }
            ]

    class FakeResponses:
        def __init__(self) -> None:
            self.request: dict | None = None

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.request = kwargs
            return SimpleNamespace(output_text="Example answer [1].")

    responses = FakeResponses()
    rag = BaseRAG(
        FakeRetriever(),
        SimpleNamespace(responses=responses),
        model_name="test-model",
    )

    rag.answer("Example question?")

    assert responses.request is not None
    prompt = json.loads(str(responses.request["input"]))
    assert (
        prompt["contexts"][0]["evidence_role"]
        == "regulatory_claim"
    )
    assert "not its quality, authority" in ANSWER_RULES


def test_agentic_search_results_include_evidence_role() -> None:
    results = [
        {
            "_id": "document-1::chunk-1",
            "_source": {
                "source_document_id": "document-1",
                "title": "Example document",
                "source": "example_source",
                "jurisdiction": "CA",
                "evidence_role": "regulatory_monograph",
                "source_url": "https://example.com/document",
                "content": "Example monograph evidence.",
            },
        }
    ]
    references: list[dict] = []
    reference_numbers: dict[str, int] = {}

    tool_output = AgenticRAG._format_search_results(
        results,
        references,
        reference_numbers,
    )

    parsed_output = json.loads(tool_output)
    assert (
        parsed_output["results"][0]["evidence_role"]
        == "regulatory_monograph"
    )


def test_agentic_rag_runs_search_tool_then_returns_answer() -> None:
    class FakeRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str) -> list[dict]:
            self.queries.append(query)
            return [
                {
                    "_id": "document-1::chunk-1",
                    "_source": {
                        "source_document_id": "document-1",
                        "title": "Example document",
                        "source": "example_source",
                        "jurisdiction": "US",
                        "evidence_role": EVIDENCE_ROLE,
                        "source_url": "https://example.com/document",
                        "content": "Example evidence.",
                    },
                }
            ]

    class FakeResponses:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 1:
                tool_call = SimpleNamespace(
                    type="function_call",
                    arguments=json.dumps({"query": "focused query"}),
                    call_id="call-1",
                )
                return SimpleNamespace(
                    output=[tool_call],
                    output_text="",
                )

            return SimpleNamespace(
                output=[],
                output_text="Grounded answer [1].",
            )

    retriever = FakeRetriever()
    responses = FakeResponses()
    rag = AgenticRAG(
        retriever,
        SimpleNamespace(responses=responses),
        model_name="test-model",
    )

    result = rag.answer("Example question?")

    assert responses.calls == 2
    assert retriever.queries == ["focused query"]
    assert result["answer"] == "Grounded answer [1]."
    assert result["search_queries"] == ["focused query"]
    assert result["contexts"][0]["source_document_id"] == (
        "document-1"
    )


def test_same_url_does_not_merge_different_source_documents() -> None:
    references = [
        {
            "_id": "document-1::chunk-1",
            "_source": {
                "source_document_id": "document-1",
                "title": "First document",
                "source": "example_source",
                "jurisdiction": "US",
                "evidence_role": EVIDENCE_ROLE,
                "source_url": "https://example.com/shared",
                "content": "First evidence.",
            },
        },
        {
            "_id": "document-2::chunk-1",
            "_source": {
                "source_document_id": "document-2",
                "title": "Second document",
                "source": "example_source",
                "jurisdiction": "US",
                "evidence_role": EVIDENCE_ROLE,
                "source_url": "https://example.com/shared",
                "content": "Second evidence.",
            },
        },
    ]

    answer, cited_references = BaseRAG._compact_citations(
        "Both documents matter [1, 2].",
        references,
    )

    assert answer == "Both documents matter [1, 2]."
    assert len(cited_references) == 2
