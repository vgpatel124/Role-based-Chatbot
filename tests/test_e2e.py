"""End-to-end through the FastAPI app with the Groq LLM, ChromaDB and Redis
mocked. Authentication (bcrypt + JWT) and the RBAC mapping are real."""

import pytest

from app import rag
from app.services.cache import RBACSemanticCache

QUESTION = "What was the gross margin in Q3 2024?"


class InMemorySemanticCache:
    """Exact-match stand-in for Redis that keeps RBACSemanticCache's namespacing."""

    def __init__(self):
        self._namespacer = RBACSemanticCache()  # CACHE_ENABLED=false → never connects
        self._entries: dict[tuple[str, str], dict] = {}
        self.set_calls = 0

    def _key(self, question, role, collections):
        return (self._namespacer._make_namespace(role, collections), question.lower().strip())

    def get(self, question, role, collections):
        return self._entries.get(self._key(question, role, collections))

    def set(self, question, role, collections, response):
        self.set_calls += 1
        self._entries[self._key(question, role, collections)] = response


@pytest.fixture
def memory_cache(monkeypatch):
    cache = InMemorySemanticCache()
    monkeypatch.setattr(rag, "semantic_cache", cache)
    return cache


def _login(client, username: str) -> dict:
    response = client.post("/auth/login-json", json={"username": username, "password": "password123"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    return {"Authorization": f"Bearer {token}"}


def test_full_flow_login_to_answer(client, mock_llm, mock_chromadb, memory_cache):
    headers = _login(client, "alice")

    response = client.post("/chat", json={"question": QUESTION}, headers=headers)

    assert response.status_code == 200
    body = response.json()
    # JWT → role → authorized collections
    assert body["role"] == "finance"
    assert mock_chromadb.touched() == {"finance"}
    # retrieval → reranking → LLM
    mock_llm.invoke.assert_called_once()
    prompt = mock_llm.invoke.call_args[0][0][1].content
    assert "62%" in prompt and "FINEMP" not in prompt
    # response with sources, session_id and request id
    assert body["answer"] == "Mocked answer [source.md]"
    assert "quarterly_financial_report.md" in body["sources"]
    assert set(body["sources"]) <= {"quarterly_financial_report.md", "financial_summary.md"}
    assert len(body["session_id"]) == 36
    assert body["cache_hit"] is False
    assert len(response.headers["X-Request-ID"]) == 36


def test_session_id_is_echoed_and_history_recorded(client, mock_llm, mock_chromadb, memory_cache):
    headers = _login(client, "alice")
    body = client.post("/chat", json={"question": QUESTION, "session_id": "my-session"}, headers=headers).json()

    assert body["session_id"] == "my-session"
    from app.services.session_store import session_store

    assert len(session_store.get_history("alice:my-session")) == 2


def test_second_identical_request_is_served_from_cache(client, mock_llm, mock_chromadb, memory_cache):
    headers = _login(client, "alice")

    first = client.post("/chat", json={"question": QUESTION}, headers=headers).json()
    llm_calls_after_first = mock_llm.invoke.call_count
    retrievals_after_first = len(mock_chromadb.dense_calls)

    second = client.post("/chat", json={"question": QUESTION}, headers=headers).json()

    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert second["answer"] == first["answer"]
    assert second["sources"] == first["sources"]
    assert mock_llm.invoke.call_count == llm_calls_after_first  # LLM not called again
    assert len(mock_chromadb.dense_calls) == retrievals_after_first  # no retrieval either
    assert memory_cache.set_calls == 1


def test_cached_answer_is_not_served_to_another_role(client, mock_llm, mock_chromadb, memory_cache):
    client.post("/chat", json={"question": QUESTION}, headers=_login(client, "alice"))

    hr_body = client.post("/chat", json={"question": QUESTION}, headers=_login(client, "carol")).json()

    assert hr_body["cache_hit"] is False
    assert hr_body["role"] == "hr"
    assert "quarterly_financial_report.md" not in hr_body["sources"]


def test_all_six_users_can_log_in_and_chat(client, mock_llm, mock_chromadb, memory_cache):
    expected_roles = {
        "alice": "finance",
        "bob": "marketing",
        "carol": "hr",
        "dave": "engineering",
        "eve": "c_level",
        "frank": "employee",
    }
    for username, role in expected_roles.items():
        response = client.post("/chat", json={"question": "Kubernetes microservices"}, headers=_login(client, username))
        assert response.status_code == 200
        assert response.json()["role"] == role


def test_wrong_password_is_rejected(client):
    response = client.post("/auth/login-json", json={"username": "alice", "password": "wrong"})
    assert response.status_code == 401


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    "llm_output",
    ["I don't have access to that information.", "I don’t have access to that information", '"I don\'t have access to that information."'],
)
def test_declined_answer_returns_no_sources_and_is_not_cached(client, mock_llm, mock_chromadb, memory_cache, llm_output):
    mock_llm.invoke.return_value.content = llm_output

    body = client.post("/chat", json={"question": QUESTION}, headers=_login(client, "alice")).json()

    mock_llm.invoke.assert_called_once()  # context passed the threshold, the LLM declined
    assert body["answer"] == llm_output
    assert body["sources"] == []
    assert memory_cache.set_calls == 0


def test_real_answer_mentioning_access_keeps_sources(client, mock_llm, mock_chromadb, memory_cache):
    mock_llm.invoke.return_value.content = "Q3 gross margin was 62%; access to Q4 data is not covered here."

    body = client.post("/chat", json={"question": QUESTION}, headers=_login(client, "alice")).json()

    assert "quarterly_financial_report.md" in body["sources"]
    assert memory_cache.set_calls == 1
