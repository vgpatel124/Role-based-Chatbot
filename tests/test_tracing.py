"""Langfuse tracing, checked against the OpenTelemetry spans the SDK would export.

A real Langfuse client is used with an in-memory span exporter, so what is asserted
here is exactly what would leave the process: span names, nesting, types, attributes
and (the important part for an HR/finance assistant) which text is withheld.
"""

import json
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import rag
from app.config import settings
from app.utils import tracing
from app.utils.logger import RequestContext

QUESTION = "What was the gross margin in Q3 2024?"
CHUNK_TEXT = "Q3 2024 gross margin was 62% and revenue was $2.4 billion."
ANSWER = "The gross margin was 62%."

EXPECTED_STEPS = {
    "answer-question",
    "lookup-cache",
    "retrieve-context",
    "search-collection",
    "rerank-candidates",
    "generate-answer",
}


class Traces:
    def __init__(self, client: Langfuse, exporter: InMemorySpanExporter, scores: list[dict]):
        self.client = client
        self._exporter = exporter
        self.scores = scores

    def spans(self):
        self.client.flush()
        return list(self._exporter.get_finished_spans())

    def by_name(self, name: str):
        return [s for s in self.spans() if s.name == name]

    @staticmethod
    def meta(span, key: str):
        return json.loads(span.attributes[f"langfuse.observation.metadata.{key}"])


@pytest.fixture
def traces(monkeypatch):
    """Tracing on, exporting to memory. Scores are recorded instead of sent."""
    exporter = InMemorySpanExporter()
    public_key = f"pk-lf-test-{uuid.uuid4().hex}"  # the SDK reuses a client per public key
    scores: list[dict] = []

    monkeypatch.setattr(settings, "LANGFUSE_TRACING_ENABLED", True)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(
        tracing,
        "_build_client",
        lambda: Langfuse(
            public_key=public_key,
            secret_key="sk-lf-test",
            base_url="http://127.0.0.1:9",
            span_exporter=exporter,
            mask_otel_spans=tracing._mask_otel_spans,
        ),
    )
    monkeypatch.setattr(
        Langfuse,
        "score_current_trace",
        lambda self, **kwargs: scores.append(kwargs),
    )
    tracing.reset_for_tests()
    client = tracing.get_client()
    assert client is not None
    yield Traces(client, exporter, scores)
    client.shutdown()
    tracing.reset_for_tests()


@pytest.fixture
def fake_llm(monkeypatch):
    """A real LangChain chat model, so the Langfuse callback records a generation."""
    llm = FakeListChatModel(responses=[ANSWER] * 10)
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    return llm


# --- Off by default, never in the way ------------------------------------------------


def test_without_keys_tracing_is_off_and_the_pipeline_is_unchanged(mock_llm, mock_chromadb):
    assert tracing.get_client() is None

    result = rag.query_rag(QUESTION, "finance")

    assert result["answer"] == "Mocked answer [source.md]"
    assert tracing.current_trace_id() is None


def test_client_startup_failure_does_not_break_requests(monkeypatch, mock_llm, mock_chromadb):
    monkeypatch.setattr(settings, "LANGFUSE_TRACING_ENABLED", True)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "sk-lf-x")

    def boom():
        raise RuntimeError("langfuse unavailable")

    monkeypatch.setattr(tracing, "_build_client", boom)
    tracing.reset_for_tests()

    result = rag.query_rag(QUESTION, "finance")

    assert result["sources"]
    assert tracing.get_client() is None
    tracing.reset_for_tests()


def test_observation_errors_do_not_break_requests(traces, monkeypatch, mock_llm, mock_chromadb):
    monkeypatch.setattr(
        traces.client, "start_as_current_observation", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("sdk"))
    )

    result = rag.query_rag(QUESTION, "finance")

    assert result["sources"]


def test_steps_outside_a_trace_create_nothing(traces, mock_chromadb):
    """Scripts that call retrieval helpers directly must not produce stray traces."""
    rag.retrieve_for_collections(QUESTION, ["finance"], embeddings=None, ctx=RequestContext())

    assert traces.spans() == []


# --- Structure -----------------------------------------------------------------------


def test_one_chat_turn_is_one_trace_with_named_steps(traces, fake_llm, mock_chromadb):
    rag.query_rag(QUESTION, "finance")

    spans = traces.spans()
    assert {s.name for s in spans} == EXPECTED_STEPS
    assert len({s.context.trace_id for s in spans}) == 1
    root = next(s for s in spans if s.name == "answer-question")
    assert root.parent is None
    assert all(s.parent is not None for s in spans if s is not root)


@pytest.mark.parametrize(
    "role, can_resolve",
    [
        ("finance", False),
        ("marketing", False),
        ("engineering", False),
        ("employee", False),
        ("hr", True),
        ("c_level", True),  # can read the HR collection
    ],
)
def test_reference_resolution_is_traced_only_for_roles_that_can_reach_hr(
    role, can_resolve, traces, fake_llm, mock_chromadb
):
    rag.query_rag("What is the leave balance of employee FINEMP1005?", role)

    steps = traces.by_name("resolve-references")
    assert (len(steps) == 1) is can_resolve
    if can_resolve:
        assert steps[0].attributes["langfuse.observation.type"] == "retriever"
        assert traces.meta(steps[0], "rows_added") == 0  # nothing to follow in the fake corpus


def test_observation_types_are_specific(traces, fake_llm, mock_chromadb):
    rag.query_rag(QUESTION, "finance")

    types = {s.name: s.attributes["langfuse.observation.type"] for s in traces.spans()}
    assert types["answer-question"] == "chain"
    assert types["search-collection"] == "retriever"
    assert types["retrieve-context"] == "retriever"
    assert types["lookup-cache"] == "retriever"
    assert types["generate-answer"] == "generation"


def test_names_are_fixed_and_run_specific_values_live_in_metadata(traces, fake_llm, mock_chromadb):
    rag.query_rag(QUESTION, "c_level")

    search_spans = traces.by_name("search-collection")
    assert len(search_spans) == 5  # c_level searches every collection
    assert {traces.meta(s, "collection") for s in search_spans} == set(rag.ALL_COLLECTIONS)
    assert all(s.name == "search-collection" for s in search_spans)


def test_pipeline_facts_are_recorded_as_metadata(traces, fake_llm, mock_chromadb):
    rag.query_rag(QUESTION, "finance")

    root = traces.by_name("answer-question")[0]
    assert traces.meta(root, "role") == "finance"
    assert traces.meta(root, "authorized_collections") == ["finance"]
    assert traces.meta(root, "cache_hit") is False
    assert traces.meta(root, "sources") == ["financial_summary.md", "quarterly_financial_report.md"]
    assert set(traces.meta(root, "latency_ms")) >= {"cache_lookup", "retrieval", "rerank", "llm"}
    rerank = traces.by_name("rerank-candidates")[0]
    assert traces.meta(rerank, "kept") >= 1
    assert all(0 <= score <= 1 for score in traces.meta(rerank, "scores"))


def test_unauthorized_collections_never_appear_in_a_trace(traces, fake_llm, mock_chromadb):
    rag.query_rag(QUESTION, "hr")

    searched = {traces.meta(s, "collection") for s in traces.by_name("search-collection")}
    assert searched == {"hr_docs"}


def test_cache_hit_trace_has_no_retrieval_or_generation(traces, fake_llm, monkeypatch):
    cached = {"answer": ANSWER, "sources": ["financial_summary.md"]}
    monkeypatch.setattr(rag, "semantic_cache", SimpleNamespace(get=lambda *a: cached, set=lambda *a: None))

    rag.query_rag(QUESTION, "finance")

    assert {s.name for s in traces.spans()} == {"answer-question", "lookup-cache"}
    assert traces.meta(traces.by_name("lookup-cache")[0], "hit") is True
    assert {"name": "cache_hit", "value": 1.0, "data_type": "BOOLEAN", "comment": None} in traces.scores


def test_rewrite_is_traced_as_its_own_generation(traces, fake_llm, mock_chromadb):
    rag.session_store.add_message("alice:s1", "user", "What was revenue in Q3 2024?")
    rag.session_store.add_message("alice:s1", "assistant", "$2.4 billion")

    rag.query_rag("And what about that for Q2?", "finance", session_id="alice:s1")

    names = [s.name for s in traces.spans()]
    assert "rewrite-query" in names
    assert traces.by_name("rewrite-query")[0].attributes["langfuse.observation.type"] == "generation"


def test_pipeline_errors_mark_the_trace_and_still_raise(traces, fake_llm, mock_chromadb, monkeypatch):
    def explode(**kwargs):
        raise RuntimeError("reranker down")

    monkeypatch.setattr(rag.reranker, "rerank", explode)

    with pytest.raises(RuntimeError, match="reranker down"):
        rag.query_rag(QUESTION, "finance")

    root = traces.by_name("answer-question")[0]
    assert root.attributes["langfuse.observation.level"] == "ERROR"
    assert root.attributes["langfuse.observation.status_message"] == "RuntimeError"


def test_llm_error_messages_are_reduced_to_the_error_class(traces, mock_chromadb, monkeypatch):
    """The callback handler exports an error's message as the span status; a provider
    error can echo the prompt, so only the class (and HTTP status) is kept."""

    class ProviderError(Exception):
        status_code = 429

    class FailingLLM(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise ProviderError(f"rejected: {CHUNK_TEXT}")

    monkeypatch.setattr(rag, "get_llm", lambda: FailingLLM(responses=["x"]))

    with pytest.raises(ProviderError):
        rag.query_rag(QUESTION, "finance")

    generation = traces.by_name("generate-answer")[0]
    assert generation.attributes["langfuse.observation.level"] == "ERROR"
    assert generation.attributes["langfuse.observation.status_message"] == "ProviderError (429)"
    assert generation.status.description == "ProviderError (429)"
    assert CHUNK_TEXT not in _all_exported_text(traces.spans())


def test_exception_text_and_stack_traces_are_never_exported(traces, fake_llm, mock_chromadb, monkeypatch):
    """The SDK would export an exception's message and traceback as span events, which
    the masking hook cannot reach, and messages can echo question or document text."""

    def explode(**kwargs):
        raise ValueError(f"cannot rank: {CHUNK_TEXT}")

    monkeypatch.setattr(rag.reranker, "rerank", explode)

    with pytest.raises(ValueError):
        rag.query_rag(QUESTION, "finance")

    for span in traces.spans():
        assert span.events == ()
        assert CHUNK_TEXT not in (span.status.description or "")
    assert CHUNK_TEXT not in _all_exported_text(traces.spans())


# --- Correlation and attributes ------------------------------------------------------


def test_trace_id_is_derived_from_the_request_id_and_stored_in_the_context(traces, fake_llm, mock_chromadb):
    ctx = RequestContext(request_id="req-1234")

    rag.query_rag(QUESTION, "finance", ctx=ctx)

    expected = traces.client.create_trace_id(seed="req-1234")
    assert ctx.trace_id == expected
    assert format(traces.by_name("answer-question")[0].context.trace_id, "032x") == expected


def test_user_session_role_and_name_are_set_on_every_observation(traces, fake_llm, mock_chromadb):
    with tracing.request_attributes(user_id="alice", session_id="alice:s1", role="finance"):
        rag.query_rag(QUESTION, "finance")

    spans = traces.spans()
    assert spans
    for span in spans:
        assert span.attributes["user.id"] == "alice"
        assert span.attributes["session.id"] == "alice:s1"
        assert "role:finance" in span.attributes["langfuse.trace.tags"]
        assert span.attributes["langfuse.trace.name"] == "chat-request"
        assert json.loads(span.attributes["langfuse.trace.metadata.content_captured"]) == "false"


def test_chat_endpoint_attributes_the_trace_to_the_user_and_request(
    traces, fake_llm, mock_chromadb, client, auth_header
):
    response = client.post(
        "/chat", json={"question": QUESTION, "session_id": "s-42"}, headers=auth_header("finance")
    )

    request_id = response.headers["X-Request-ID"]
    root = traces.by_name("answer-question")[0]
    assert root.attributes["user.id"] == "alice"
    assert root.attributes["session.id"] == "alice:s-42"  # per-user, like the conversation history
    assert format(root.context.trace_id, "032x") == traces.client.create_trace_id(seed=request_id)


def test_quality_signals_are_scored(traces, fake_llm, mock_chromadb):
    rag.query_rag(QUESTION, "finance")

    scored = {s["name"]: s for s in traces.scores}
    assert scored["cache_hit"]["value"] == 0.0 and scored["cache_hit"]["data_type"] == "BOOLEAN"
    assert scored["answer_declined"]["value"] == 0.0
    assert scored["no_relevant_context"]["value"] == 0.0
    assert scored["reranker_top_score"]["data_type"] == "NUMERIC"
    assert 0 < scored["reranker_top_score"]["value"] <= 1


def test_a_declined_answer_is_scored_as_declined(traces, monkeypatch, mock_chromadb):
    monkeypatch.setattr(rag, "get_llm", lambda: FakeListChatModel(responses=[rag.DECLINED_ANSWER]))

    rag.query_rag(QUESTION, "finance")

    assert {s["name"]: s["value"] for s in traces.scores}["answer_declined"] == 1.0


# --- Privacy -------------------------------------------------------------------------


def _all_exported_text(spans) -> str:
    return " ".join(str(value) for span in spans for value in span.attributes.values())


def test_no_question_answer_prompt_or_document_text_is_exported_by_default(traces, fake_llm, mock_chromadb):
    rag.query_rag(QUESTION, "finance")

    spans = traces.spans()
    assert {s.name for s in spans} == EXPECTED_STEPS
    for span in spans:
        assert "langfuse.observation.input" not in span.attributes
        assert "langfuse.observation.output" not in span.attributes
    exported = _all_exported_text(spans)
    for secret in (QUESTION, CHUNK_TEXT, ANSWER, "$2.4 billion", "Mocked"):
        assert secret not in exported
    # The generation still carries what is needed without text.
    assert "langfuse.observation.type" in traces.by_name("generate-answer")[0].attributes


def test_content_is_exported_when_capture_is_switched_on(traces, fake_llm, mock_chromadb, monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_CAPTURE_CONTENT", True)

    rag.query_rag(QUESTION, "finance")

    root = traces.by_name("answer-question")[0]
    assert root.attributes["langfuse.observation.input"] == QUESTION
    assert root.attributes["langfuse.observation.output"] == ANSWER
    assert "gross margin" in traces.by_name("generate-answer")[0].attributes["langfuse.observation.input"]


def test_secrets_and_emails_are_scrubbed_even_when_content_is_captured(traces, fake_llm, mock_chromadb, monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_CAPTURE_CONTENT", True)
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbGljZSJ9.c2lnbmF0dXJlLXZhbHVl"
    question = f"Send it to jane.doe@fintechco.com using Bearer {token} and key gsk_abcdefghijklmnopqrstuvwx"

    rag.query_rag(question, "finance")

    exported = _all_exported_text(traces.spans())
    assert "jane.doe@fintechco.com" not in exported
    assert token not in exported
    assert "gsk_abcdefghijklmnopqrstuvwx" not in exported
    assert "[email]" in exported


def test_user_and_session_ids_are_not_scrubbed(traces):
    with tracing.request_attributes(user_id="alice@corp.com", session_id="alice@corp.com:s1", role="finance"):
        with tracing.observation("answer-question", root=True):
            pass

    span = traces.by_name("answer-question")[0]
    assert span.attributes["user.id"] == "alice@corp.com"
    assert span.attributes["session.id"] == "alice@corp.com:s1"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail a.b+c@corp.example.com now", "mail [email] now"),
        ("Authorization: Bearer abcdefgh12345678", "Authorization: Bearer [redacted]"),
        ("key sk-lf-0123456789abcdef0123", "key [api-key]"),
        ("organization `org_01k44xe14ae8na59` limit", "organization `org_<redacted>` limit"),
        ("nothing sensitive here", "nothing sensitive here"),
    ],
)
def test_scrub(text, expected):
    assert tracing.scrub(text) == expected
