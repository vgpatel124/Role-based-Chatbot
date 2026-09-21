import json
from unittest.mock import MagicMock

import pytest
import redis

from app import rag
from app.config import settings
from app.services.cache import RBACSemanticCache

ALL_COLLECTIONS = ["finance", "marketing", "hr_docs", "engineering", "general"]
CACHED = {"answer": "Cached: Q3 gross margin was 62%.", "sources": ["quarterly_financial_report.md"]}


@pytest.fixture
def fake_cache(monkeypatch):
    cache = MagicMock(name="semantic_cache")
    cache.get.return_value = None
    monkeypatch.setattr(rag, "semantic_cache", cache)
    return cache


# --- query_rag integration ------------------------------------------------


def test_cache_hit_returns_cached_response_without_calling_llm(fake_cache, mock_llm, mock_chromadb):
    fake_cache.get.return_value = CACHED

    result = rag.query_rag("What was the gross margin in Q3 2024?", "finance", session_id="alice:s1")

    assert result == CACHED
    mock_llm.invoke.assert_not_called()
    assert mock_chromadb.dense_calls == []  # no retrieval either
    fake_cache.set.assert_not_called()
    fake_cache.get.assert_called_once_with("What was the gross margin in Q3 2024?", "finance", ["finance"])


def test_cache_miss_runs_full_pipeline_and_stores_result(fake_cache, mock_llm, mock_chromadb):
    result = rag.query_rag("What was the gross margin in Q3 2024?", "finance", session_id="alice:s1")

    mock_llm.invoke.assert_called_once()
    assert mock_chromadb.touched() == {"finance"}
    fake_cache.set.assert_called_once()
    question, role, collections, stored = fake_cache.set.call_args[0]
    assert (question, role, collections) == ("What was the gross margin in Q3 2024?", "finance", ["finance"])
    assert stored == result
    assert result["answer"] == "Mocked answer [source.md]"
    assert "quarterly_financial_report.md" in result["sources"]


def test_safe_no_context_answer_is_not_cached(fake_cache, mock_llm, mock_chromadb):
    result = rag.query_rag("What is the leave balance of FINEMP1010?", "finance")
    assert result["sources"] == []
    fake_cache.set.assert_not_called()


# --- namespacing ----------------------------------------------------------


def test_finance_and_hr_namespaces_differ_for_identical_question():
    cache = RBACSemanticCache()  # CACHE_ENABLED=false in tests → no connection
    assert cache._make_namespace("finance", ["finance"]) != cache._make_namespace("hr", ["hr_docs"])


def test_c_level_namespace_differs_from_finance():
    cache = RBACSemanticCache()
    assert cache._make_namespace("c_level", ALL_COLLECTIONS) != cache._make_namespace("finance", ["finance"])


def test_namespace_is_order_independent_and_versioned(monkeypatch):
    cache = RBACSemanticCache()
    ns = cache._make_namespace("c_level", ALL_COLLECTIONS)
    assert ns == cache._make_namespace("c_level", list(reversed(ALL_COLLECTIONS)))
    assert ns.endswith(":v1")
    monkeypatch.setattr(settings, "RETRIEVAL_VERSION", "v2")
    assert cache._make_namespace("c_level", ALL_COLLECTIONS) != ns


def test_lookup_filters_on_exact_escaped_namespace(mock_redis):
    cache = RBACSemanticCache()
    cache.get("What was Q3 revenue?", "finance", ["finance"])
    query = mock_redis.index.query.call_args[0][0]
    assert "@namespace:{finance\\:finance\\:v1}" in str(query)


# --- hit / miss thresholds against a mocked Redis -------------------------


def _hit(namespace: str, distance: float) -> list[dict]:
    return [{"namespace": namespace, "response_json": json.dumps(CACHED), "vector_distance": str(distance)}]


def test_get_returns_entry_above_similarity_threshold(mock_redis):
    cache = RBACSemanticCache()
    mock_redis.index.query.return_value = _hit("finance:finance:v1", 0.05)  # similarity 0.95
    assert cache.get("q3 revenue", "finance", ["finance"]) == CACHED


def test_get_ignores_entry_below_similarity_threshold(mock_redis):
    cache = RBACSemanticCache()
    mock_redis.index.query.return_value = _hit("finance:finance:v1", 0.20)  # similarity 0.80
    assert cache.get("q3 revenue", "finance", ["finance"]) is None


def test_get_rejects_entry_from_another_namespace(mock_redis):
    cache = RBACSemanticCache()
    mock_redis.index.query.return_value = _hit("hr:hr_docs:v1", 0.0)
    assert cache.get("q3 revenue", "finance", ["finance"]) is None


def test_set_writes_namespace_and_applies_ttl(mock_redis):
    cache = RBACSemanticCache()
    cache.set("What was Q3 revenue?", "finance", ["finance"], CACHED)

    record = mock_redis.index.load.call_args[0][0][0]
    assert record["namespace"] == "finance:finance:v1"
    assert record["question"] == "what was q3 revenue?"
    assert json.loads(record["response_json"]) == CACHED
    assert isinstance(record["embedding"], bytes) and len(record["embedding"]) == 384 * 4
    mock_redis.client.expire.assert_called_once_with("rbac_cache_v1:01TESTKEY", settings.CACHE_TTL)


# --- failure modes --------------------------------------------------------


def test_redis_down_at_startup_disables_cache_without_raising(mock_redis):
    mock_redis.client.ping.side_effect = redis.exceptions.ConnectionError("Connection refused")
    cache = RBACSemanticCache()
    assert cache.available is False
    assert cache.get("q", "finance", ["finance"]) is None
    cache.set("q", "finance", ["finance"], CACHED)  # must not raise
    mock_redis.index.load.assert_not_called()


def test_redis_errors_at_runtime_are_swallowed(mock_redis):
    cache = RBACSemanticCache()
    assert cache.available is True
    mock_redis.index.query.side_effect = redis.exceptions.ConnectionError("lost connection")
    mock_redis.index.load.side_effect = redis.exceptions.ConnectionError("lost connection")
    assert cache.get("q", "finance", ["finance"]) is None
    cache.set("q", "finance", ["finance"], CACHED)  # must not raise


def test_chatbot_answers_normally_when_redis_is_down(
    monkeypatch, mock_redis, mock_llm, mock_chromadb, client, auth_header
):
    mock_redis.client.ping.side_effect = redis.exceptions.ConnectionError("Connection refused")
    monkeypatch.setattr(rag, "semantic_cache", RBACSemanticCache())

    response = client.post(
        "/chat", json={"question": "What was the gross margin in Q3 2024?"}, headers=auth_header("finance")
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Mocked answer [source.md]"
    assert response.json()["cache_hit"] is False


def test_cache_disabled_get_always_returns_none(mock_redis, monkeypatch):
    cache = RBACSemanticCache()
    mock_redis.index.query.return_value = _hit("finance:finance:v1", 0.0)
    monkeypatch.setattr(settings, "CACHE_ENABLED", False)

    assert cache.get("q3 revenue", "finance", ["finance"]) is None
    cache.set("q3 revenue", "finance", ["finance"], CACHED)
    mock_redis.index.query.assert_not_called()
    mock_redis.index.load.assert_not_called()
