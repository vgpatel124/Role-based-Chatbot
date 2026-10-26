"""The pure parts of evaluation/benchmark_latency.py: statistics, workload, log parsing and
report assembly. The live run (real server, Redis, Groq, Langfuse) is exercised by hand."""

import json
from datetime import datetime

import pytest

from evaluation import benchmark_latency as bench
from evaluation.redteam_cases import EXPECTED_ACCESS

SAMPLES = bench.load_benchmark_samples()


# --- Statistics ---------------------------------------------------------------------


def test_percentile_interpolates_linearly():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert bench.percentile(values, 0) == 10
    assert bench.percentile(values, 50) == 30
    assert bench.percentile(values, 100) == 50
    assert bench.percentile(values, 25) == 20
    assert bench.percentile(values, 90) == pytest.approx(46)


def test_percentile_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        bench.percentile([], 50)


def test_summary_withholds_percentiles_the_sample_cannot_support():
    small = bench.summarize([float(i) for i in range(1, 11)])
    medium = bench.summarize([float(i) for i in range(1, 51)])
    large = bench.summarize([float(i) for i in range(1, 101)])

    assert small["n"] == 10 and small["p50"] == 5.5
    assert small["p95"] is None and small["p99"] is None
    assert medium["p95"] == pytest.approx(47.55) and medium["p99"] is None
    assert large["p95"] == pytest.approx(95.05) and large["p99"] == pytest.approx(99.01)


def test_summary_is_independent_of_input_order():
    assert bench.summarize([3.0, 1.0, 2.0]) == bench.summarize([1.0, 2.0, 3.0])
    assert bench.summarize([]) == {"n": 0}


# --- Workload ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def workload():
    return bench.build_workload(SAMPLES, run_id="t")


def test_workload_has_the_documented_composition(workload):
    counts: dict[str, int] = {}
    for step in workload:
        counts[step.category] = counts.get(step.category, 0) + 1

    assert counts == {"answerable": 41, "repeat": 25, "out_of_scope": 6, "unauthorized": 20, "follow_up": 8}
    assert len(workload) == 100
    assert [s.seq for s in workload] == list(range(1, 101))


def test_workload_is_deterministic_for_a_seed():
    a = [s.record() for s in bench.build_workload(SAMPLES, seed=1, run_id="x")]
    b = [s.record() for s in bench.build_workload(SAMPLES, seed=1, run_id="x")]
    c = [s.record() for s in bench.build_workload(SAMPLES, seed=2, run_id="x")]

    assert a == b
    assert a != c


def test_follow_ups_come_straight_after_their_anchor_in_the_same_session(workload):
    follow_ups = [s for s in workload if s.category == "follow_up"]
    assert len(follow_ups) == 8
    for follow_up in follow_ups:
        anchor = workload[follow_up.seq - 2]  # seq is 1-based, so the previous step is index seq-2
        assert follow_up.key == f"followup:{anchor.key}"
        assert anchor.category == "answerable"
        assert anchor.session == follow_up.session
        assert anchor.role == follow_up.role
        assert follow_up.question == bench.FOLLOW_UPS[follow_up.role]


def test_repeats_follow_their_original_in_a_new_session(workload):
    position = {s.key: s.seq for s in workload}
    repeats = [s for s in workload if s.category == "repeat"]
    assert len(repeats) == 25
    assert len({s.key for s in repeats}) == 25  # each question repeated once
    for repeat in repeats:
        original_key = repeat.key.removeprefix("repeat:")
        assert position[original_key] < repeat.seq
        original = next(s for s in workload if s.key == original_key)
        assert original.question == repeat.question
        assert original.session != repeat.session
        assert original.role == repeat.role


def test_unauthorized_requests_all_ask_for_another_departments_data(workload):
    unauthorized = [s for s in workload if s.category == "unauthorized"]
    assert len(unauthorized) == 20
    assert {s.role for s in unauthorized} == {"finance", "marketing", "hr", "engineering", "employee"}
    assert all(s.role != "c_level" for s in unauthorized)  # c_level can read everything
    for step in unauthorized:
        target = step.key.split("->")[1].split("/")[0]  # key is "unauth:<role>-><collection>/direct"
        assert target not in EXPECTED_ACCESS[step.role], step.key


def test_result_records_never_contain_question_text(workload):
    dumped = json.dumps([s.record() for s in workload])

    for step in workload:
        assert step.question not in dumped


def test_every_session_is_scoped_to_the_run(workload):
    assert all(s.session.startswith("bench-t-") for s in workload)


# --- Log parsing -----------------------------------------------------------------------

SUMMARY = (
    '2026-10-23T10:00:00+0530 | INFO     | app.request | request_summary '
    '{"request_id": "r1", "cache_hit": false, "latency_ms": {"cache_lookup": 1.0, "retrieval": 50.0, "rerank": 100.0, "llm": 900.0, "total": 1060.0}}'
)


def test_request_summaries_are_found_by_request_id():
    found = bench.parse_request_summaries(["noise", SUMMARY, "more noise"])

    assert set(found) == {"r1"}
    assert found["r1"]["latency_ms"]["llm"] == 900.0


def test_retry_lines_become_epoch_times():
    lines = [
        "[2026-10-23 10:00:05 - groq._base_client:1068 - INFO] Retrying request to /openai/v1/chat/completions in 1.5 seconds",
        "[2026-10-23 10:00:06 - groq._base_client:1068 - DEBUG] something else",
        "plain line",
    ]

    (when,) = bench.parse_retry_times(lines)

    assert when == datetime(2026, 10, 23, 10, 0, 5).timestamp()


@pytest.mark.parametrize(
    "log, expected",
    [
        (["Chat request failed", "RateLimitError: ... on tokens per day (TPD): Limit 200000"], "provider_daily_token_limit"),
        (["Chat request failed", "groq.RateLimitError: tokens per minute (TPM)"], "provider_rate_limit"),
        (["Chat request failed | request_id=x", "Traceback", "KeyError"], "server_error"),
        (["nothing relevant"], "unknown"),
    ],
)
def test_failures_are_classified_from_the_requests_log_lines(log, expected):
    assert bench.classify_failure(log) == expected


def test_retries_are_attributed_to_the_request_they_happened_in():
    rows = [
        {"start_wall": 100.0, "end_wall": 102.0},
        {"start_wall": 110.0, "end_wall": 112.0},
    ]

    bench.mark_retried(rows, [101.0])

    assert [r["provider_retry"] for r in rows] == [True, False]


@pytest.mark.parametrize(
    "summary, path",
    [
        (None, "unknown"),
        ({"cache_hit": True, "latency_ms": {"cache_lookup": 1}}, "cache_hit"),
        ({"cache_hit": True, "latency_ms": {"rewrite": 900, "cache_lookup": 1}}, "rewrite_cache_hit"),
        ({"cache_hit": False, "latency_ms": {"rewrite": 5, "llm": 9}}, "rewrite_path"),
        ({"cache_hit": False, "latency_ms": {"llm": 9}}, "llm_answer"),
        ({"cache_hit": False, "latency_ms": {"retrieval": 9, "rerank": 4}}, "no_llm"),
    ],
)
def test_request_path(summary, path):
    assert bench.request_path(summary) == path


def test_trace_id_matches_the_langfuse_sdk_derivation():
    # Observed from Langfuse().create_trace_id(seed="abc") during the tracing work.
    assert bench.trace_id_for("abc") == "ba7816bf8f01cfea414140de5dae2223"


# --- Report assembly ----------------------------------------------------------------------


def _row(i, e2e, *, ok=True, category="answerable", role="finance", retry=False, kind=None):
    return {
        "seq": i, "key": f"k{i}", "category": category, "role": role, "phase": "measured",
        "ok": ok, "request_id": f"r{i}", "status": 200 if ok else 500, "error_kind": kind,
        "e2e_ms": e2e, "start_wall": float(i), "end_wall": float(i) + 1, "provider_retry": retry,
    }


def _summary(i, *, hit=False, llm=800.0, total=1000.0):
    latency = {"cache_lookup": 1.0, "total": total}
    if not hit:
        latency |= {"retrieval": 60.0, "rerank": 100.0, "llm": llm}
    return {"request_id": f"r{i}", "cache_hit": hit, "latency_ms": latency}


def _generation(i, **usage):
    return {"traceId": bench.trace_id_for(f"r{i}"), "type": "GENERATION", "model": "openai/gpt-oss-120b", "usageDetails": usage}


def test_report_separates_warmups_failures_and_measured_requests():
    rows = [
        {**_row(-1, 5000.0), "phase": "warmup"},
        _row(1, 1200.0), _row(2, 1300.0), _row(3, 50.0, category="repeat"),
        _row(4, 30000.0, ok=False, kind="provider_rate_limit"),
    ]
    summaries = {"r1": _summary(1), "r2": _summary(2), "r3": _summary(3, hit=True, total=40.0)}

    report = bench.build_report(rows, summaries, {})

    assert report["counts"] == {"warmup": 1, "measured_attempted": 4, "measured_ok": 3, "measured_failed": 1, "provider_retry_flagged": 0}
    assert report["error_rate"] == 0.25
    assert report["failure_kinds"] == {"provider_rate_limit": 1}
    assert report["e2e_ms"]["all_ok"]["n"] == 3  # neither the warm-up nor the failure is in the latency statistics
    assert report["e2e_ms"]["all_ok"]["max"] == 1300.0
    assert report["cache"] == {"hits": 1, "misses": 2}
    assert report["e2e_ms"]["by_path"]["cache_hit"]["n"] == 1


def test_provider_retries_are_reported_both_ways():
    rows = [_row(1, 1000.0), _row(2, 9000.0, retry=True)]
    summaries = {"r1": _summary(1), "r2": _summary(2)}

    report = bench.build_report(rows, summaries, {})

    assert report["counts"]["provider_retry_flagged"] == 1
    assert report["e2e_ms"]["all_ok"]["n"] == 2
    assert report["e2e_ms"]["all_ok_without_provider_retries"]["n"] == 1
    assert report["e2e_ms"]["all_ok_without_provider_retries"]["max"] == 1000.0


def test_components_cover_uncached_requests_only_and_overhead_is_the_remainder():
    rows = [_row(1, 1100.0), _row(2, 60.0, category="repeat")]
    summaries = {"r1": _summary(1, llm=800.0, total=1000.0), "r2": _summary(2, hit=True, total=40.0)}

    report = bench.build_report(rows, summaries, {})

    assert report["components_ms_cold_requests"]["llm"]["n"] == 1
    assert report["components_ms_cold_requests"]["llm"]["mean"] == 800.0
    assert report["components_ms_cold_requests"]["total"]["n"] == 1
    overhead = report["http_and_auth_overhead_ms"]
    assert overhead["n"] == 2 and overhead["min"] == 20.0 and overhead["max"] == 100.0


def test_tokens_come_from_generation_observations_and_sum_the_rewrite_call():
    observations = [
        _generation(1, input=1400, output=60, total=1500, output_reasoning=100),
        _generation(2, input=200, output=20, total=230, output_reasoning=10),  # rewrite call of request 2
        _generation(2, input=1300, output=50, total=1400, output_reasoning=90),  # answer call of request 2
        {"traceId": bench.trace_id_for("r1"), "type": "SPAN"},
    ]
    rows = [_row(1, 1000.0), _row(2, 1500.0)]
    summaries = {"r1": _summary(1), "r2": _summary(2)}

    report = bench.build_report(rows, summaries, bench.tokens_by_request(observations))

    assert report["llm_requests_with_usage"] == 2
    assert report["tokens_total"] == {"input": 2900, "output": 130, "reasoning": 200, "total": 3130}
    assert report["tokens_per_llm_request"]["total"]["max"] == 1630


def test_report_and_rows_contain_no_question_text():
    secret_question = "What is the salary of employee FINEMP9999, a very specific question?"
    step = bench.Step(seq=1, key="k1", category="answerable", role="hr", question=secret_question, session="s")
    row = {**_row(1, 1000.0), **step.record()}

    report = bench.build_report([row], {"r1": _summary(1)}, {})

    text = json.dumps({"report": report, "rows": [row]}, default=str) + bench.format_report({"run_id": "t"}, report)
    assert secret_question not in text
    assert "FINEMP9999" not in text


def test_formatted_report_says_na_where_the_sample_is_too_small():
    rows = [_row(i, 1000.0 + i) for i in range(1, 11)]
    summaries = {f"r{i}": _summary(i) for i in range(1, 11)}

    text = bench.format_report({"run_id": "t", "model": "m"}, bench.build_report(rows, summaries, {}))

    assert "n/a" in text
    assert "all successful" in text
    assert "error rate 0.0%" in text


def test_exact_repeat_hits_are_separated_from_unexpected_hits_and_rewrite_hits():
    rows = [
        _row(1, 40.0, category="repeat"),
        _row(2, 60.0, category="repeat"),
        _row(3, 16.0, category="answerable"),  # a question's first appearance answered from the cache
        _row(4, 1200.0, category="follow_up"),
    ]
    summaries = {
        "r1": _summary(1, hit=True, total=30.0),
        "r2": _summary(2, hit=True, total=50.0),
        "r3": _summary(3, hit=True, total=10.0),
        "r4": {"request_id": "r4", "cache_hit": True, "latency_ms": {"rewrite": 1000.0, "cache_lookup": 5.0, "total": 1100.0}},
    }

    report = bench.build_report(rows, summaries, {})

    assert report["e2e_ms"]["by_path"]["cache_hit_exact_repeats"]["n"] == 2
    assert report["e2e_ms"]["by_path"]["cache_hit"]["n"] == 3
    assert report["e2e_ms"]["by_path"]["rewrite_cache_hit"]["n"] == 1
    assert report["unexpected_cache_hits"] == ["k3"]
    assert report["cache"] == {"hits": 4, "misses": 0}
    assert report["components_ms_cold_requests"] == {}  # cache hits are not cold requests


def test_llm_path_breakdown_shares_add_up_to_the_server_total():
    rows = [_row(1, 1100.0), _row(2, 2100.0)]
    summaries = {
        "r1": {"request_id": "r1", "cache_hit": False, "latency_ms": {"cache_lookup": 100.0, "retrieval": 100.0, "rerank": 100.0, "llm": 600.0, "total": 1000.0}},
        "r2": {"request_id": "r2", "cache_hit": False, "latency_ms": {"cache_lookup": 100.0, "retrieval": 100.0, "rerank": 100.0, "llm": 1600.0, "total": 2000.0}},
    }

    breakdown = bench.build_report(rows, summaries, {})["llm_answer_path_breakdown"]

    assert breakdown["llm"]["mean_ms"] == 1100.0
    assert breakdown["llm"]["share_of_server_time"] == pytest.approx(2200 / 3000)
    # components account for 900 of 1,000 and 1,900 of 2,000 ms; the remaining 100 ms each is "other"
    assert breakdown["other_in_server"]["mean_ms"] == pytest.approx(100.0)
    assert breakdown["other_in_server"]["share_of_server_time"] == pytest.approx(200 / 3000)
    assert sum(v["share_of_server_time"] for v in breakdown.values()) == pytest.approx(1.0)


def test_benchmark_questions_do_not_change_when_the_dataset_grows():
    samples = bench.load_benchmark_samples()

    assert len(samples) == bench.BENCHMARK_QUESTIONS == 47
    assert sum(s["type"] == "answerable" for s in samples) == 41
    assert all("evidence" not in s for s in samples)  # the original questions, not later additions
