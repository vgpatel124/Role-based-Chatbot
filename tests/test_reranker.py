import numpy as np
import pytest
import torch
from langchain_core.documents import Document

from app.config import settings
from app.services.reranker import CrossEncoderReranker

SCORES = {"a": 0.90, "b": 0.05, "c": 0.40, "d": 0.70, "e": 0.20}


@pytest.fixture
def reranker():
    r = CrossEncoderReranker("fake-model")
    r._model.predict = lambda pairs: np.array([SCORES[passage] for _, passage in pairs], dtype=np.float32)
    return r


def _candidates() -> list[Document]:
    return [Document(page_content=key, metadata={"source": f"{key}.md"}) for key in SCORES]


def test_model_is_loaded_once_with_sigmoid_activation():
    r = CrossEncoderReranker()
    assert r._model.model_name == settings.RERANKER_MODEL
    # Scores must be 0-1 for RERANKER_SCORE_THRESHOLD to mean the same thing for every model.
    assert isinstance(r._model.kwargs["activation_fn"], torch.nn.Sigmoid)


def test_results_sorted_by_score_descending(reranker):
    docs, scores = reranker.rerank("q", _candidates(), top_k=10, threshold=0.0)
    assert [d.page_content for d in docs] == ["a", "d", "c", "e", "b"]
    assert scores == sorted(scores, reverse=True)


def test_documents_below_threshold_are_excluded(reranker):
    docs, scores = reranker.rerank("q", _candidates(), top_k=10, threshold=0.3)
    assert [d.page_content for d in docs] == ["a", "d", "c"]
    assert all(s >= 0.3 for s in scores)


def test_all_candidates_below_threshold_returns_empty(reranker):
    assert reranker.rerank("q", _candidates(), top_k=10, threshold=0.95) == ([], [])


def test_top_k_limit_is_respected(reranker):
    docs, scores = reranker.rerank("q", _candidates(), top_k=2, threshold=0.0)
    assert [d.page_content for d in docs] == ["a", "d"]
    assert len(scores) == 2


def test_reranker_score_is_added_to_metadata(reranker):
    docs, scores = reranker.rerank("q", _candidates(), top_k=3, threshold=0.0)
    for doc, score in zip(docs, scores):
        assert doc.metadata["reranker_score"] == pytest.approx(score)
    assert docs[0].metadata["reranker_score"] == pytest.approx(0.90)


def test_empty_candidates_returns_empty(reranker):
    assert reranker.rerank("q", [], top_k=5, threshold=0.1) == ([], [])


def test_prediction_failure_fails_closed(reranker):
    def boom(pairs):
        raise RuntimeError("model crashed")

    reranker._model.predict = boom
    assert reranker.rerank("q", _candidates(), top_k=5, threshold=0.0) == ([], [])
