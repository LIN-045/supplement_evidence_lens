from app.base_rag import BaseRAG
from types import SimpleNamespace

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
                "title": "Example document",
                "source": "example_source",
                "jurisdiction": "US",
                "source_url": "https://example.com/document#first",
                "content": first_part + overlap,
            },
        },
        {
            "_id": "document-1::chunk-2",
            "_source": {
                "title": "Example document",
                "source": "example_source",
                "jurisdiction": "US",
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

def test_base_rag_always_returns_structured_result() -> None:
    class FakeRetriever:
        def search(self, query: str) -> list[dict]:
            return [
                {
                    "_id": "document-1::chunk-1",
                    "_source": {
                        "title": "Example document",
                        "source": "example_source",
                        "jurisdiction": "US",
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
                "title": "Example document",
                "source": "example_source",
                "jurisdiction": "US",
                "source_url": "https://example.com/document",
                "content": "Example evidence.",
            }
        ],
    }