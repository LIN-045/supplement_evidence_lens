from typing import Never

from app.retrieval import rerank_results


class FailingReranker:
    """Fail if an empty result set reaches the model."""

    def predict(self, pairs: object) -> Never:
        raise AssertionError("The reranker should not be called")


def test_reranker_returns_empty_results_without_calling_model() -> None:
    assert rerank_results(FailingReranker(), "question", []) == []
