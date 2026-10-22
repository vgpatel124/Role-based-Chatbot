"""The answer-evaluation harness must tell a provider outage from a quality problem:
rate-limit errors are waited out or reported, never scored as an empty answer."""

import pytest

from evaluation import evaluate_answers as ev

DAILY_LIMIT = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in "
    "organization `org_01abc123xyz` service tier `on_demand` on tokens per day (TPD): Limit 200000, "
    "Used 199000, Requested 3000. Please try again in 2h10m5.5s.'}}"
)
MINUTE_LIMIT = "Error code: 429 - Rate limit reached on tokens per minute. Please try again in 7.74s."


@pytest.mark.parametrize(
    "message, expected",
    [
        (MINUTE_LIMIT, 7.74),
        (DAILY_LIMIT, 2 * 3600 + 10 * 60 + 5.5),
        ("Error code: 429 - slow down", 15.0),
        ("Rate limit exceeded", 15.0),
    ],
)
def test_rate_limit_wait_follows_the_providers_retry_after(message, expected):
    assert ev._rate_limit_wait(Exception(message)) == pytest.approx(expected)


def test_other_errors_are_not_treated_as_rate_limits():
    assert ev._rate_limit_wait(ValueError("boom")) is None


def test_error_text_hides_the_organisation_id():
    text = ev._error_text(Exception(DAILY_LIMIT))

    assert "org_01abc123xyz" not in text
    assert "org_<redacted>" in text
    assert "tokens per day" in text


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(ev.time, "sleep", recorded.append)
    return recorded


def test_short_rate_limit_is_waited_out_then_succeeds(monkeypatch, sleeps):
    calls = iter([Exception(MINUTE_LIMIT), {"answer": "ok", "sources": [], "contexts": []}])

    def fake_query_rag(*args, **kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(ev, "query_rag", fake_query_rag)

    response = ev.ask({"question": "q", "role": "finance"})

    assert response["answer"] == "ok"
    assert sleeps == [pytest.approx(7.74 + 1)]


def test_daily_limit_is_reported_without_pointless_retries(monkeypatch, sleeps):
    attempts = []

    def fake_query_rag(*args, **kwargs):
        attempts.append(1)
        raise Exception(DAILY_LIMIT)

    monkeypatch.setattr(ev, "query_rag", fake_query_rag)

    response = ev.ask({"question": "q", "role": "finance"})

    assert len(attempts) == 1
    assert sleeps == []
    assert "tokens per day" in response["error"]
    assert "org_01abc123xyz" not in response["error"]


def test_errored_answer_is_not_scored_as_a_wrong_answer(monkeypatch, sleeps):
    sample = {
        "id": "s1",
        "role": "finance",
        "type": "answerable",
        "question": "q",
        "keywords": ["$1"],
        "expected_sources": ["a.md"],
    }
    monkeypatch.setattr(ev, "ask", lambda s: {"answer": "", "sources": [], "contexts": [], "error": "RateLimitError: x"})

    (row,) = ev.score_deterministic([sample])

    assert row["refusal_correct"] is None
    assert row["keyword_coverage"] is None
    assert row["source_hit"] is None
    summary = ev.summarize([row])
    assert summary["keyword_coverage"] is None
    assert summary["refusal_accuracy"] is None
    assert any("provider errors" in reason for reason in ev.inconclusive([row], judged=0))


def test_clean_run_is_conclusive():
    rows = [{"id": "s1", "error": None, "faithfulness": 1.0, "answer_relevancy": 1.0,
             "context_precision": 1.0, "context_recall": 1.0}]

    assert ev.inconclusive(rows, judged=1) == []


def test_missing_ragas_scores_make_the_run_inconclusive():
    rows = [{"id": f"s{i}", "error": None} for i in range(4)]

    reasons = ev.inconclusive(rows, judged=4)

    assert len(reasons) == 4
    assert all("scored for only 0 of 4" in r for r in reasons)
