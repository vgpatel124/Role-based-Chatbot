"""The RAGAS judge (Claude Haiku): request shape, finish detection, preflight, wiring.

Every provider call is mocked: no Anthropic or Groq request is made, and no credential is read.
"""

import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from anthropic.types import Message, TextBlock, ThinkingBlock, Usage
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompt_values import StringPromptValue

from evaluation import evaluate_answers as ev

ROOT = Path(__file__).resolve().parent.parent
FAKE_KEY = "sk-ant-test-SECRET-DO-NOT-LEAK-0123456789"


@pytest.fixture
def anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)  # keep the tests independent of a developer's gateway setting


class _RawResponse:
    """What `messages.with_raw_response.create` returns: the parsed Message plus headers."""

    headers: dict = {}

    def __init__(self, message):
        self._message = message

    def parse(self):
        return self._message


def _message(stop_reason="end_turn", text='{"ok": 1}', thinking=True):
    blocks = [ThinkingBlock(type="thinking", thinking="", signature="sig")] if thinking else []
    blocks.append(TextBlock(type="text", text=text))
    return Message(
        id="msg_test", type="message", role="assistant", model=ev.JUDGE_MODEL, content=blocks,
        stop_reason=stop_reason, stop_sequence=None, usage=Usage(input_tokens=3, output_tokens=2),
    )


# --- Model choice ------------------------------------------------------------------------


def test_judge_is_claude_haiku_with_a_pinned_dateless_id():
    assert ev.JUDGE_MODEL == "claude-haiku-5-5"
    assert not re.search(r"-\d{8}$", ev.JUDGE_MODEL)  # Claude IDs from the 4.6 generation on carry no date suffix
    assert ev.JUDGE_MODEL.startswith("claude-haiku-")


# --- Request shape ---------------------------------------------------------------------------


def test_judge_strips_sampling_parameters_even_if_the_library_would_send_them(anthropic_key, monkeypatch):
    """Claude Haiku 5.5 answers 400 to sampling parameters. langchain-anthropic 1.7.6 already
    withholds them, so this makes the library misbehave to prove the judge's own guard."""

    def sends_sampling(self, *args, **kwargs):
        return {
            "model": self.model, "messages": [], "max_tokens": self.max_tokens,
            "temperature": 0.01, "top_p": 0.9, "top_k": 5,
        }

    monkeypatch.setattr(ChatAnthropic, "_get_request_payload", sends_sampling)

    payload = ev.build_judge_model()._get_request_payload([HumanMessage(content="hi")])

    assert payload["model"] == "claude-haiku-5-5"
    assert payload["max_tokens"] == ev.JUDGE_MAX_TOKENS
    for key in ("temperature", "top_p", "top_k"):
        assert key not in payload


def test_default_request_carries_no_sampling_parameters(anthropic_key):
    model = ev.build_judge_model()
    model.temperature = 0.01  # what RAGAS does on every call

    payload = model._get_request_payload([HumanMessage(content="hi")])

    assert not {"temperature", "top_p", "top_k"} & set(payload)


def test_ragas_wrapper_cannot_inject_a_temperature_into_the_request(anthropic_key, monkeypatch):
    """Drive the real RAGAS wrapper (sync path, which sets a temperature unconditionally) against a
    stubbed transport and inspect what would have been sent."""
    from ragas.llms import LangchainLLMWrapper

    sent = []

    def fake_create(self, payload):
        sent.append(payload)
        return _RawResponse(_message())

    monkeypatch.setattr(ChatAnthropic, "_create", fake_create)
    wrapper = LangchainLLMWrapper(
        ev.build_judge_model(), bypass_temperature=True, bypass_n=True, is_finished_parser=ev.judge_finished
    )

    result = wrapper.generate_text(StringPromptValue(text="Extract the statements."))

    assert len(sent) == 1
    assert not {"temperature", "top_p", "top_k"} & set(sent[0])
    assert result.generations[0][0].text == '{"ok": 1}'  # the empty thinking block is not part of the text
    assert wrapper.is_finished(result) is True


def test_truncated_judge_output_is_not_finished(anthropic_key, monkeypatch):
    from ragas.llms import LangchainLLMWrapper

    monkeypatch.setattr(
        ChatAnthropic, "_create", lambda self, payload: _RawResponse(_message(stop_reason="max_tokens", text='{"ok": '))
    )
    wrapper = LangchainLLMWrapper(
        ev.build_judge_model(), bypass_temperature=True, bypass_n=True, is_finished_parser=ev.judge_finished
    )

    result = wrapper.generate_text(StringPromptValue(text="x"))

    assert wrapper.is_finished(result) is False


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"stop_reason": "end_turn"}, True),
        ({"stop_reason": "stop_sequence"}, True),
        ({"stop_reason": "max_tokens"}, False),
        ({"stop_reason": "refusal"}, False),
        ({"stop_reason": "tool_use"}, False),
        ({}, False),
    ],
)
def test_judge_finished_only_accepts_a_natural_end(metadata, expected):
    from langchain_core.messages import AIMessage

    result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x", response_metadata=metadata))]])

    assert ev.judge_finished(result) is expected


# --- Preflight: credentials, availability, provider errors are never a pass -------------------------


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


class _StatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _FakeJudge:
    def __init__(self, reply=None, error=None):
        self._reply, self._error = reply, error

    def invoke(self, prompt):
        if self._error:
            raise self._error
        return self._reply


def _ok_reply(stop_reason="end_turn"):
    from langchain_core.messages import AIMessage

    return AIMessage(content="OK", response_metadata={"stop_reason": stop_reason})


@pytest.fixture
def fake_provider(monkeypatch, anthropic_key):
    import anthropic

    _FakeAnthropic.error = None
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.setattr(ev, "build_judge_model", lambda: _FakeJudge(reply=_ok_reply()))
    return _FakeAnthropic


def test_preflight_passes_when_the_model_is_available_and_answers(fake_provider):
    assert ev.preflight_judge() is None


def test_missing_anthropic_key_is_a_blocker_not_a_pass(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    problem = ev.preflight_judge()

    assert problem and "ANTHROPIC_API_KEY is not set" in problem


@pytest.mark.parametrize("status, label", [(401, "HTTP 401"), (404, "HTTP 404"), (403, "HTTP 403"), (529, "HTTP 529")])
def test_availability_errors_are_reported_without_the_provider_message_or_the_key(fake_provider, status, label):
    fake_provider.error = _StatusError(status, f"request with key {FAKE_KEY} was refused for org org_abc123")

    problem = ev.preflight_judge()

    assert problem and label in problem and "claude-haiku-5-5" in problem
    assert FAKE_KEY not in problem and "org_abc123" not in problem


def test_failing_test_call_is_a_blocker(fake_provider, monkeypatch):
    monkeypatch.setattr(ev, "build_judge_model", lambda: _FakeJudge(error=_StatusError(400, "temperature is not supported")))

    problem = ev.preflight_judge()

    assert problem and "test call" in problem and "HTTP 400" in problem
    assert "temperature is not supported" not in problem


def test_unfinished_test_call_is_a_blocker(fake_provider, monkeypatch):
    monkeypatch.setattr(ev, "build_judge_model", lambda: _FakeJudge(reply=_ok_reply("max_tokens")))

    assert "did not finish normally" in ev.preflight_judge()


# --- main(): nothing is generated unless the judge is usable ------------------------------------------


@pytest.fixture
def main_env(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "RESULTS_PATH", tmp_path / "answer_results.json")
    monkeypatch.setattr(ev.settings, "GROQ_API_KEY", "gsk_test_not_a_real_key")
    calls = []

    def must_not_generate(*args, **kwargs):
        calls.append("generated")
        return []

    monkeypatch.setattr(ev, "score_deterministic", must_not_generate)
    return calls


def test_main_stops_before_generating_when_the_judge_key_is_missing(main_env, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--check", "--ragas-limit", "12"])

    status = ev.main()

    assert status == 2
    assert main_env == []  # no Groq tokens were spent
    assert "RAGAS judge unavailable" in capsys.readouterr().err


def test_main_stops_before_generating_when_the_preflight_fails(main_env, monkeypatch):
    monkeypatch.setattr(ev, "preflight_judge", lambda: "model claude-haiku-5-5 could not be confirmed: NotFoundError (HTTP 404)")
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--check", "--ragas-limit", "12"])

    assert ev.main() == 2
    assert main_env == []


def test_main_without_ragas_does_not_need_the_judge(main_env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ev, "preflight_judge", lambda: pytest.fail("the judge must not be contacted with --no-ragas"))
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py", "--no-ragas", "--limit", "1"])

    assert ev.main() == 0
    assert main_env == ["generated"]


# --- CI wiring ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load((ROOT / ".github" / "workflows" / "evals.yml").read_text(encoding="utf-8"))


def _steps(workflow):
    """The named steps (`uses:` setup steps have no name)."""
    return [s for s in workflow["jobs"]["evals"]["steps"] if "name" in s]


LLM_STEPS = ["RBAC red-team (full pipeline with LLM)", "Answer quality (RAGAS, Claude Haiku judge)"]


def test_anthropic_secret_is_passed_to_the_two_llm_backed_steps_only(workflow):
    holders = [s["name"] for s in _steps(workflow) if "ANTHROPIC_API_KEY" in (s.get("env") or {})]

    assert holders == LLM_STEPS
    for step in _steps(workflow):
        if step["name"] in LLM_STEPS:
            assert step["env"]["ANTHROPIC_API_KEY"] == "${{ secrets.ANTHROPIC_API_KEY }}"
            assert step["env"]["LLM_PROVIDER"] == "anthropic"
    assert "ANTHROPIC_API_KEY" not in (workflow["jobs"]["evals"].get("env") or {})
    assert "ANTHROPIC_API_KEY" not in (workflow.get(True) or workflow.get("on") or {})


def test_the_evals_workflow_does_not_use_groq(workflow):
    """Groq's free-tier daily limit kept stopping the evaluation; the chatbot now runs on Anthropic here."""
    lines = (ROOT / ".github" / "workflows" / "evals.yml").read_text(encoding="utf-8").splitlines()
    code = [line for line in lines if not line.lstrip().startswith("#")]

    assert not [line for line in code if "GROQ" in line.upper()]


def test_llm_steps_run_unless_the_pr_is_from_a_fork_so_a_missing_secret_fails_loudly(workflow):
    steps = {s["name"]: s for s in _steps(workflow)}

    for name in LLM_STEPS:
        assert steps[name]["if"] == "env.IS_FORK_PR != 'true'"  # no 'secret present' gate that could skip silently
    assert "IS_FORK_PR" in workflow["jobs"]["evals"]["env"]


def test_llm_step_commands_are_unchanged(workflow):
    steps = {s["name"]: s for s in _steps(workflow)}

    assert steps["Answer quality (RAGAS, Claude Haiku judge)"]["run"] == (
        "python evaluation/evaluate_answers.py --check --ragas-limit 12"
    )
    assert steps["RBAC red-team (full pipeline with LLM)"]["run"] == "python evaluation/redteam.py --llm"
    assert "env" not in steps["RBAC red-team (retrieval and reranker)"]  # needs no key at all


def test_no_credential_value_is_written_in_the_workflow_or_the_eval_sources():
    pattern = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}|gsk_[A-Za-z0-9]{20,}")
    for path in (
        ROOT / ".github" / "workflows" / "evals.yml",
        ROOT / "evaluation" / "evaluate_answers.py",
        ROOT / "requirements-eval.txt",
        ROOT / ".env.example",
    ):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


def test_thresholds_are_unchanged():
    expected = {
        "refusal_accuracy": 0.93,
        "answer_rate": 0.93,
        "decline_rate_out_of_scope": 1.0,
        "keyword_coverage": 0.95,
        "source_hit_rate": 0.95,
        "faithfulness": 0.8,
        "answer_relevancy": 0.8,
        "context_precision": 0.75,
        "context_recall": 0.85,
    }

    assert json.loads((ROOT / "evaluation" / "thresholds.json").read_text(encoding="utf-8")) == expected


def test_anthropic_client_is_pinned_in_the_runtime_requirements():
    """The chatbot can run on Anthropic (LLM_PROVIDER), so the client is a runtime dependency."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert re.search(r"^langchain-anthropic==\d+\.\d+\.\d+$", requirements, re.MULTILINE)
