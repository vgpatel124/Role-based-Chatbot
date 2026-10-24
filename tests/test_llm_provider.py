"""The LLM provider setting: Groq by default, Anthropic when asked.

Covers the model factory, reading Claude's block-list replies, the preflight, and the
evaluation scripts' credential and error handling. Every provider call is mocked.
"""

import sys
from types import SimpleNamespace

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import ValidationError

from app import rag
from app.config import Settings, settings
from app.services.query_rewriter import rewrite_query
from app.utils import llm as llm_module
from evaluation import evaluate_answers as ev
from evaluation import redteam as rt

FAKE_KEY = "sk-ant-test-SECRET-DO-NOT-LEAK-0123456789"
REPLY_BLOCKS = [
    {"type": "thinking", "thinking": "", "signature": "sig"},
    {"type": "text", "text": "The gross margin was 62%."},
]


@pytest.fixture
def anthropic_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setattr(settings, "ANTHROPIC_MODEL", "claude-haiku-5-5")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)


# --- Settings -----------------------------------------------------------------------------


def test_provider_defaults_to_groq():
    assert settings.LLM_PROVIDER == "groq"
    assert Settings().LLM_PROVIDER == "groq"


def test_unknown_provider_is_rejected_at_startup():
    with pytest.raises(ValidationError):
        Settings(LLM_PROVIDER="openai")


def test_anthropic_model_defaults_to_claude_haiku_5_5():
    assert Settings().ANTHROPIC_MODEL == "claude-haiku-5-5"


# --- The model factory ------------------------------------------------------------------------


def test_groq_remains_the_default_model():
    model = llm_module.build_chat_model()

    assert isinstance(model, ChatGroq)
    assert model.model_name == settings.GROQ_MODEL
    assert model.temperature == 0.2
    assert llm_module.llm_model_name() == settings.GROQ_MODEL


def test_anthropic_provider_builds_claude_without_sampling_parameters(anthropic_provider):
    model = llm_module.build_chat_model()

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-haiku-5-5"
    assert model.max_tokens == llm_module.ANTHROPIC_MAX_TOKENS
    assert model.temperature is None and model.top_p is None and model.top_k is None  # never configured for Claude
    payload = model._get_request_payload([HumanMessage(content="hi")])
    assert not {"temperature", "top_p", "top_k"} & set(payload)
    assert llm_module.llm_model_name() == "claude-haiku-5-5"


@pytest.mark.parametrize(
    "provider, key_field, expected",
    [("groq", "GROQ_API_KEY", "GROQ_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")],
)
def test_missing_credential_is_named_and_building_the_model_fails(monkeypatch, provider, key_field, expected):
    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(settings, key_field, "")

    assert llm_module.missing_llm_credential() == expected
    with pytest.raises(RuntimeError, match=expected):
        llm_module.build_chat_model()


def test_credentials_present_means_nothing_is_missing(anthropic_provider):
    assert llm_module.missing_llm_credential() is None


# --- Reading replies ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, expected",
    [
        ("plain string reply", "plain string reply"),
        (REPLY_BLOCKS, "The gross margin was 62%."),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        (["bare ", {"type": "text", "text": "mix"}], "bare mix"),
        ([{"type": "thinking", "thinking": "hidden"}], ""),
        ([], ""),
        (None, ""),
    ],
)
def test_message_text_reads_strings_and_block_lists(content, expected):
    assert llm_module.message_text(SimpleNamespace(content=content)) == expected
    assert llm_module.message_text(AIMessage(content=content or "")) == expected


def test_a_claude_block_list_reply_becomes_a_clean_answer(monkeypatch, mock_chromadb):
    fake = SimpleNamespace(invoke=lambda messages, **kwargs: AIMessage(content=REPLY_BLOCKS))
    monkeypatch.setattr(rag, "get_llm", lambda: fake)

    result = rag.query_rag("What was the gross margin in Q3 2024?", "finance")

    assert result["answer"] == "The gross margin was 62%."  # not the repr of a list of dicts
    assert result["sources"]


def test_the_whole_chatbot_runs_on_a_real_claude_model_object(anthropic_provider, monkeypatch, mock_chromadb):
    """Real ChatAnthropic with only the network call stubbed: a thinking block plus a text block,
    as Claude Haiku 5.5 returns, goes through query_rag and comes out as clean text."""
    from anthropic.types import Message, TextBlock, ThinkingBlock, Usage

    sent = []

    class Raw:
        headers: dict = {}

        def __init__(self, message):
            self._message = message

        def parse(self):
            return self._message

    def fake_create(self, payload):
        sent.append(payload)
        return Raw(
            Message(
                id="msg_test", type="message", role="assistant", model="claude-haiku-5-5",
                content=[
                    ThinkingBlock(type="thinking", thinking="", signature="sig"),
                    TextBlock(type="text", text="The gross margin was 62%."),
                ],
                stop_reason="end_turn", stop_sequence=None, usage=Usage(input_tokens=5, output_tokens=7),
            )
        )

    monkeypatch.setattr(ChatAnthropic, "_create", fake_create)
    rag.get_llm.cache_clear()
    try:
        result = rag.query_rag("What was the gross margin in Q3 2024?", "finance")
    finally:
        rag.get_llm.cache_clear()

    assert result["answer"] == "The gross margin was 62%."
    assert result["sources"]
    assert len(sent) == 1
    assert sent[0]["model"] == "claude-haiku-5-5"
    assert sent[0]["max_tokens"] == llm_module.ANTHROPIC_MAX_TOKENS
    assert not {"temperature", "top_p", "top_k"} & set(sent[0])
    assert rag.SYSTEM_PROMPT in str(sent[0]["system"])  # the same system prompt as for Groq


def test_a_reply_with_no_text_is_declined_not_shown_as_a_list(monkeypatch, mock_chromadb):
    """A safety refusal comes back with no text blocks."""
    fake = SimpleNamespace(invoke=lambda messages, **kwargs: AIMessage(content=[]))
    monkeypatch.setattr(rag, "get_llm", lambda: fake)

    result = rag.query_rag("What was the gross margin in Q3 2024?", "finance")

    assert result["answer"] == rag.DECLINED_ANSWER
    assert result["sources"] == []


def test_query_rewriting_reads_block_list_replies():
    fake = SimpleNamespace(invoke=lambda messages, **kwargs: AIMessage(content=[{"type": "text", "text": " What was Q4 revenue? "}]))
    history = [{"role": "user", "content": "What was Q3 revenue?"}, {"role": "assistant", "content": "$2.4 billion"}]

    assert rewrite_query("And Q4?", history, fake) == "What was Q4 revenue?"


# --- Preflight --------------------------------------------------------------------------------------


class _StatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _FakeModels:
    def __init__(self, error=None):
        self._error = error

    def retrieve(self, model):
        if self._error:
            raise self._error
        return object()


class _FakeAnthropic:
    error = None

    def __init__(self, *args, **kwargs):
        self.models = _FakeModels(type(self).error)


class _FakeChat:
    def __init__(self, reply=None, error=None):
        self._reply, self._error = reply, error

    def invoke(self, prompt):
        if self._error:
            raise self._error
        return self._reply


def _reply(text="OK", stop_reason="end_turn"):
    return AIMessage(content=text, response_metadata={"stop_reason": stop_reason})


@pytest.fixture
def fake_provider(monkeypatch, anthropic_provider):
    import anthropic

    _FakeAnthropic.error = None
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.setattr(llm_module, "build_chat_model", lambda: _FakeChat(reply=_reply()))
    return _FakeAnthropic


def test_preflight_does_nothing_for_groq():
    assert llm_module.preflight_chat_model() is None


def test_preflight_passes_when_the_model_answers(fake_provider):
    assert llm_module.preflight_chat_model() is None


@pytest.mark.parametrize("status", [401, 403, 404, 529])
def test_preflight_reports_availability_errors_without_the_message_or_the_key(fake_provider, status):
    fake_provider.error = _StatusError(status, f"key {FAKE_KEY} refused for org org_abc123")

    problem = llm_module.preflight_chat_model()

    assert problem and f"HTTP {status}" in problem and "claude-haiku-5-5" in problem
    assert FAKE_KEY not in problem and "org_abc123" not in problem


@pytest.mark.parametrize(
    "chat, fragment",
    [
        (_FakeChat(error=_StatusError(400, "temperature is not supported")), "HTTP 400"),
        (_FakeChat(reply=_reply(stop_reason="max_tokens")), "did not finish normally"),
        (_FakeChat(reply=_reply(text="")), "returned no text"),
    ],
)
def test_preflight_rejects_a_model_that_fails_the_test_call(fake_provider, monkeypatch, chat, fragment):
    monkeypatch.setattr(llm_module, "build_chat_model", lambda: chat)

    problem = llm_module.preflight_chat_model()

    assert problem and fragment in problem
    assert "temperature is not supported" not in problem


# --- evaluate_answers.py ---------------------------------------------------------------------------------


@pytest.fixture
def eval_env(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "RESULTS_PATH", tmp_path / "answer_results.json")
    calls = []

    def must_not_generate(*args, **kwargs):
        calls.append("generated")
        return []

    monkeypatch.setattr(ev, "score_deterministic", must_not_generate)
    monkeypatch.setattr(ev, "preflight_judge", lambda: None)
    return calls


def test_evaluation_stops_before_generating_when_the_chatbot_key_is_missing(eval_env, monkeypatch, capsys):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--check", "--ragas-limit", "12"])

    assert ev.main() == 2
    assert eval_env == []
    assert "ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic" in capsys.readouterr().err


def test_evaluation_stops_before_generating_when_the_chatbot_model_fails_preflight(eval_env, anthropic_provider, monkeypatch, capsys):
    monkeypatch.setattr(ev, "preflight_chat_model", lambda: "chatbot model claude-haiku-5-5 could not be confirmed: AuthenticationError (HTTP 401)")
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--check", "--ragas-limit", "12"])

    assert ev.main() == 2
    assert eval_env == []
    assert "Generator unavailable" in capsys.readouterr().err


def test_evaluation_with_anthropic_does_not_need_a_groq_key(eval_env, anthropic_provider, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(ev, "preflight_chat_model", lambda: None)
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--no-ragas", "--limit", "1"])

    assert ev.main() == 0
    assert eval_env == ["generated"]


def test_reuse_does_not_contact_the_generator(eval_env, anthropic_provider, monkeypatch):
    monkeypatch.setattr(ev, "preflight_chat_model", lambda: pytest.fail("--reuse generates nothing"))
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--reuse", "--no-ragas", "--limit", "1"])

    assert ev.main() == 0


def test_results_name_the_provider_and_model(eval_env, anthropic_provider, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "preflight_chat_model", lambda: None)
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--no-ragas", "--limit", "1"])

    ev.main()

    import json

    assert json.loads((tmp_path / "answer_results.json").read_text())["generator"] == "anthropic/claude-haiku-5-5"


# --- redteam.py ------------------------------------------------------------------------------------------


@pytest.fixture
def red_env(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "ROOT", tmp_path)
    monkeypatch.setattr(rt, "RESULTS_PATH", tmp_path / "redteam_results.json")
    monkeypatch.setattr(rt, "build_cases", _three_cases)
    monkeypatch.setattr(rt, "LLM_PAUSE_SECONDS", 0)


def _three_cases():
    from evaluation.redteam_cases import build_cases

    return build_cases()[:3]


def _outcome(case, *, error=None, leaks=None):
    row = {"id": case.id, "role": case.role, "target": case.target, "template": case.template,
           "leaks": leaks or [], "candidates": 1, "passed_threshold": 0, "refused": True}
    if error:
        row["error"] = error
        row.pop("refused")
    return row


def test_red_team_llm_mode_needs_the_providers_key(red_env, monkeypatch, capsys):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(sys, "argv", ["redteam.py", "--llm"])

    assert rt.main() == 2
    assert "--llm needs ANTHROPIC_API_KEY for LLM_PROVIDER=anthropic" in capsys.readouterr().err


def test_red_team_fails_fast_when_the_chatbot_model_is_unavailable(red_env, anthropic_provider, monkeypatch, capsys):
    monkeypatch.setattr(rt, "preflight_chat_model", lambda: "chatbot model claude-haiku-5-5 could not be confirmed: AuthenticationError (HTTP 401)")
    monkeypatch.setattr(rt, "run_case", lambda case, use_llm: pytest.fail("no attack should run"))
    monkeypatch.setattr(sys, "argv", ["redteam.py", "--llm"])

    assert rt.main() == 2
    assert "chatbot LLM unavailable" in capsys.readouterr().err


def test_provider_errors_are_inconclusive_never_a_pass(red_env, anthropic_provider, monkeypatch, capsys):
    monkeypatch.setattr(rt, "preflight_chat_model", lambda: None)
    monkeypatch.setattr(rt, "run_case", lambda case, use_llm: _outcome(case, error="AuthenticationError: rejected"))
    monkeypatch.setattr(sys, "argv", ["redteam.py", "--llm"])

    status = rt.main()

    out = capsys.readouterr().out
    assert status == 1
    assert "INCONCLUSIVE" in out and "PASS" not in out


def test_clean_run_passes(red_env, anthropic_provider, monkeypatch, capsys):
    monkeypatch.setattr(rt, "preflight_chat_model", lambda: None)
    monkeypatch.setattr(rt, "run_case", lambda case, use_llm: _outcome(case))
    monkeypatch.setattr(sys, "argv", ["redteam.py", "--llm"])

    assert rt.main() == 0
    assert "RESULT: PASS - no leaks" in capsys.readouterr().out


def test_a_leak_fails_the_run(red_env, anthropic_provider, monkeypatch, capsys):
    monkeypatch.setattr(rt, "preflight_chat_model", lambda: None)
    monkeypatch.setattr(rt, "run_case", lambda case, use_llm: _outcome(case, leaks=["answer: contains ['$9.4 billion']"]))
    monkeypatch.setattr(sys, "argv", ["redteam.py", "--llm"])

    assert rt.main() == 1
    assert "FAIL - unauthorized data reached a role" in capsys.readouterr().out


def test_retrieval_only_red_team_does_not_need_any_llm_key(red_env, monkeypatch, capsys):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(rt, "run_case", lambda case, use_llm: _outcome(case))
    monkeypatch.setattr(sys, "argv", ["redteam.py"])

    assert rt.main() == 0
