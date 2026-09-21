from types import SimpleNamespace

import pytest

from app import rag
from app.config import settings
from app.services.query_rewriter import needs_rewriting, rewrite_query
from app.services.session_store import InMemorySessionStore, SessionStore, session_store
from app.utils.logger import RequestContext

HISTORY = [
    {"role": "user", "content": "What is the leave policy for employees?"},
    {"role": "assistant", "content": "Employees get 12 days of sick leave per year."},
]


# --- needs_rewriting --------------------------------------------------------


def test_self_contained_question_without_history_is_not_rewritten():
    assert needs_rewriting("What is the leave policy?", []) is False


def test_what_about_follow_up_is_rewritten():
    assert needs_rewriting("What about contractors?", HISTORY) is True


def test_pronoun_follow_up_with_punctuation_is_rewritten():
    assert needs_rewriting("And that?", HISTORY) is True


def test_follow_up_without_history_is_not_rewritten():
    assert needs_rewriting("What about contractors?", []) is False


@pytest.mark.parametrize("question", ["Compared to Q2?", "How does it compare?", "Is this the same as before?"])
def test_context_dependent_follow_ups_are_rewritten(question):
    assert needs_rewriting(question, HISTORY) is True


@pytest.mark.parametrize("question", ["Show itemized vendor costs", "What are the SDLC stages?"])
def test_self_contained_questions_with_history_are_not_rewritten(question):
    # "it" inside "itemized" must not count: signals match whole words only.
    assert needs_rewriting(question, HISTORY) is False


def test_rewrite_query_falls_back_to_original_on_llm_error():
    llm = SimpleNamespace(invoke=lambda messages: (_ for _ in ()).throw(RuntimeError("groq down")))
    assert rewrite_query("And that?", HISTORY, llm) == "And that?"


def test_rewrite_query_falls_back_to_original_on_empty_output():
    llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="   "))
    assert rewrite_query("And that?", HISTORY, llm) == "And that?"


# --- session store ----------------------------------------------------------


def test_in_memory_store_satisfies_protocol():
    assert isinstance(InMemorySessionStore(), SessionStore)


def test_history_is_capped_at_six_messages():
    store = InMemorySessionStore()
    for i in range(10):
        store.add_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
    history = store.get_history("s1")
    assert len(history) == 6
    assert [m["content"] for m in history] == ["m4", "m5", "m6", "m7", "m8", "m9"]


def test_sessions_are_isolated():
    store = InMemorySessionStore()
    store.add_message("alice:s1", "user", "finance question")
    store.add_message("carol:s1", "user", "hr question")
    assert [m["content"] for m in store.get_history("alice:s1")] == ["finance question"]
    assert [m["content"] for m in store.get_history("carol:s1")] == ["hr question"]
    assert store.get_history("unknown") == []


def test_returned_history_cannot_mutate_the_store():
    store = InMemorySessionStore()
    store.add_message("s1", "user", "original")
    store.get_history("s1")[0]["content"] = "tampered"
    assert store.get_history("s1")[0]["content"] == "original"


# --- query_rag with history ---------------------------------------------------


def test_query_adds_user_and_assistant_messages_to_history(mock_llm, mock_chromadb):
    rag.query_rag("What was the gross margin in Q3 2024?", "finance", session_id="alice:s1")

    history = session_store.get_history("alice:s1")
    assert history == [
        {"role": "user", "content": "What was the gross margin in Q3 2024?"},
        {"role": "assistant", "content": "Mocked answer [source.md]"},
    ]


def test_follow_up_is_rewritten_and_rewrite_drives_retrieval(mock_llm, mock_chromadb):
    session_store.add_message("alice:s1", "user", "What was the gross margin in Q3 2024?")
    session_store.add_message("alice:s1", "assistant", "It was 62%.")
    mock_llm.invoke.side_effect = [
        SimpleNamespace(content="What was the gross margin in Q4 2024?"),  # rewrite call
        SimpleNamespace(content="Q4 gross margin was 64%."),  # answer call
    ]

    ctx = RequestContext()
    result = rag.query_rag("What about Q4?", "finance", session_id="alice:s1", ctx=ctx)

    assert ctx.query_rewritten is True
    assert ctx.rewritten_query == "What was the gross margin in Q4 2024?"
    assert mock_chromadb.dense_calls == [("finance", "What was the gross margin in Q4 2024?")]
    answer_prompt = mock_llm.invoke.call_args_list[1][0][0][1].content
    assert answer_prompt.startswith("Question: What about Q4?")  # original question is answered
    assert result["answer"] == "Q4 gross margin was 64%."


def test_self_contained_question_skips_rewrite_llm_call(mock_llm, mock_chromadb):
    session_store.add_message("alice:s1", "user", "Earlier question")
    ctx = RequestContext()
    rag.query_rag("What was the gross margin in Q3 2024?", "finance", session_id="alice:s1", ctx=ctx)
    assert ctx.query_rewritten is False
    assert mock_llm.invoke.call_count == 1  # answer only, no rewrite call


def test_rewriting_disabled_by_setting(monkeypatch, mock_llm, mock_chromadb):
    monkeypatch.setattr(settings, "QUERY_REWRITING_ENABLED", False)
    session_store.add_message("alice:s1", "user", "What was the gross margin in Q3 2024?")
    ctx = RequestContext()
    rag.query_rag("What about the gross margin in Q4?", "finance", session_id="alice:s1", ctx=ctx)
    assert ctx.query_rewritten is False


def test_other_users_history_is_never_used(client, auth_header, mock_llm, mock_chromadb):
    # alice builds history under session "shared"; carol sends the same session_id.
    session_store.add_message("alice:shared", "user", "What was the gross margin in Q3 2024?")
    session_store.add_message("alice:shared", "assistant", "62%.")

    client.post("/chat", json={"question": "And that?", "session_id": "shared"}, headers=auth_header("hr"))

    # carol had no history of her own, so no rewrite happened and alice's session is untouched.
    assert len(session_store.get_history("alice:shared")) == 2
    rewrite_calls = [c for c in mock_llm.invoke.call_args_list if "Conversation history" in c[0][0][1].content]
    assert rewrite_calls == []
