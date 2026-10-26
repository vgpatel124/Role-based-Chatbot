"""End-to-end latency benchmark of the running chatbot.

Usage (from the project root, after `python app/ingest.py`):
    python evaluation/benchmark_latency.py --dry-run          # show the workload, no network
    python evaluation/benchmark_latency.py --limit 6          # small smoke run
    python evaluation/benchmark_latency.py                    # the full 100-request run

What it does
    1. Starts Redis Stack on loopback if nothing listens on the port (and stops it afterwards).
    2. Starts the API with `uvicorn` as a separate process, with Langfuse tracing in the
       "evaluation" environment so benchmark traces stay apart from real traffic.
    3. Warm-up: one request per role (loads BM25 indexes and vector collections), then the
       semantic cache is emptied. Warm-ups are reported, never measured.
    4. Measured: 100 sequential requests over real HTTP, timed by the client.
    5. Reads the server's per-request log line (component latencies), then the Langfuse
       traces (token usage, model), matched through the request id.

The workload comes from the repo's synthetic, sanitized datasets. Result files contain
dataset ids and numbers only: no question, answer or document text.

Groq's free tier allows 8,000 tokens/minute and 200,000/day for the generator, so requests
are paced, provider retries (which would inflate latency) are detected and flagged, and the
run stops cleanly if the daily limit is reached.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.redteam_cases import build_cases  # noqa: E402

DATASET_PATH = ROOT / "evaluation" / "answer_dataset.json"
# The benchmark workload was designed on the first 47 questions (41 answerable + 6 out-of-scope).
# Questions added to the dataset later are appended after them, so taking only these keeps the
# workload, and every latency figure measured with it, comparable over time.
BENCHMARK_QUESTIONS = 47
RETRIEVAL_DATASET_PATH = ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = ROOT / "evaluation" / "benchmark_results.json"
REPORT_PATH = ROOT / "evaluation" / "benchmark_report.md"

SEED = 20261023
N_REPEATS = 25  # answerable questions asked a second time, in a new session
# Percentiles are only reported when the sample is large enough to mean something.
MIN_N_P95 = 20
MIN_N_P99 = 100

# Generic follow-up questions (they trigger the rewrite step); anchors per role.
FOLLOW_UPS = {
    "finance": "What about Q4?",
    "marketing": "And what about Q3?",
    "hr": "What about their manager?",
    "engineering": "How about the scaling part?",
    "employee": "What about for maternity leave?",
    "c_level": "And how does that compare?",
}
FOLLOW_UP_PLAN = {"finance": 2, "marketing": 2, "hr": 1, "engineering": 1, "employee": 1, "c_level": 1}

# One warm-up question per role, from the retrieval set (wording differs from the workload).
WARMUP_IDS = {
    "finance": ("fin_vendor_expenses", "finance"),
    "marketing": ("mkt_campaign_performance", "marketing"),
    "hr": ("hr_by_name", "hr"),
    "engineering": ("eng_microservices", "engineering"),
    "employee": ("gen_work_hours", "employee"),
    "c_level": ("fin_gross_margin", "c_level"),
}

USERNAME_FOR_ROLE = {
    "finance": "alice",
    "marketing": "bob",
    "hr": "carol",
    "engineering": "dave",
    "c_level": "eve",
    "employee": "frank",
}
PASSWORD = "password123"  # the documented demo users (app/auth.py)


# --- Statistics ---------------------------------------------------------------------


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in 0..100) of an ascending list."""
    if not sorted_values:
        raise ValueError("no values")
    position = (len(sorted_values) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def summarize(values: list[float]) -> dict:
    """n, mean, min, p50, max, plus p95 / p99 only when the sample permits."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "mean": sum(ordered) / n,
        "min": ordered[0],
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95) if n >= MIN_N_P95 else None,
        "p99": percentile(ordered, 99) if n >= MIN_N_P99 else None,
        "max": ordered[-1],
    }


# --- Workload -----------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    seq: int
    key: str  # dataset id; never the question text
    category: str  # answerable | repeat | out_of_scope | unauthorized | follow_up
    role: str
    question: str
    session: str

    def record(self) -> dict:
        """The step without its question text, for result files."""
        return {"seq": self.seq, "key": self.key, "category": self.category, "role": self.role}


def load_benchmark_samples() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))[:BENCHMARK_QUESTIONS]


def build_workload(samples: list[dict], seed: int = SEED, run_id: str = "run") -> list[Step]:
    """100 requests: 41 answerable + 25 repeats + 6 out-of-scope + 20 cross-department
    refusals + 8 follow-ups. Deterministic for a given seed. Follow-ups stay directly
    behind their anchor; repeats come after their original so the cache can serve them."""
    rng = random.Random(seed)
    answerable = [s for s in samples if s["type"] == "answerable"]
    out_of_scope = [s for s in samples if s["type"] != "answerable"]
    unauthorized = [c for c in build_cases() if c.template == "direct"]

    units: list[list[tuple]] = []  # (key, category, role, question, session-group)
    anchors: dict[str, set[str]] = {}
    for role, count in FOLLOW_UP_PLAN.items():
        pool = [s for s in answerable if s["role"] == role]
        anchors[role] = {s["id"] for s in rng.sample(pool, count)}

    for sample in answerable:
        unit = [(sample["id"], "answerable", sample["role"], sample["question"], sample["id"])]
        if sample["id"] in anchors[sample["role"]]:
            unit.append((f"followup:{sample['id']}", "follow_up", sample["role"], FOLLOW_UPS[sample["role"]], sample["id"]))
        units.append(unit)
    for sample in out_of_scope:
        units.append([(sample["id"], "out_of_scope", sample["role"], sample["question"], sample["id"])])
    for case in unauthorized:
        units.append([(f"unauth:{case.id}", "unauthorized", case.role, case.question, case.id)])
    rng.shuffle(units)

    flat = [step for unit in units for step in unit]
    for sample in rng.sample(answerable, N_REPEATS):
        origin = next(i for i, step in enumerate(flat) if step[0] == sample["id"])
        position = rng.randint(origin + 1, len(flat))
        while position < len(flat) and flat[position][1] == "follow_up":
            position += 1  # never split a question from its follow-up
        flat.insert(position, (f"repeat:{sample['id']}", "repeat", sample["role"], sample["question"], f"repeat:{sample['id']}"))

    return [
        Step(seq=i, key=key, category=category, role=role, question=question, session=f"bench-{run_id}-{group}")
        for i, (key, category, role, question, group) in enumerate(flat, start=1)
    ]


# --- Server log parsing -------------------------------------------------------------

_SUMMARY = re.compile(r"request_summary (\{.*\})\s*$")
_RETRY = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - groq\._base_client:\d+ - INFO\] Retrying request to")


def parse_request_summaries(lines: list[str]) -> dict[str, dict]:
    """The structured per-request log lines of app/utils/logger.py, by request id."""
    found: dict[str, dict] = {}
    for line in lines:
        match = _SUMMARY.search(line)
        if match:
            payload = json.loads(match.group(1))
            found[payload["request_id"]] = payload
    return found


def parse_retry_times(lines: list[str]) -> list[float]:
    """Epoch seconds of each provider retry (the Groq SDK logs them at INFO with GROQ_LOG=info)."""
    times = []
    for line in lines:
        match = _RETRY.match(line)
        if match:
            times.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp())
    return times


def classify_failure(new_lines: list[str]) -> str:
    """Why a request failed, from the server log lines written while it ran (the API
    itself only answers 500, without a request id header)."""
    text = "\n".join(new_lines)
    if "tokens per day" in text:
        return "provider_daily_token_limit"
    if "tokens per minute" in text or "RateLimitError" in text:
        return "provider_rate_limit"
    if "Chat request failed" in text:
        return "server_error"
    return "unknown"


def mark_retried(rows: list[dict], retry_times: list[float], slack_s: float = 1.5) -> None:
    """Flag requests during which the provider SDK retried a call (latency then includes a back-off)."""
    for row in rows:
        row["provider_retry"] = any(row["start_wall"] - slack_s <= t <= row["end_wall"] + slack_s for t in retry_times)


def request_path(summary: dict | None) -> str:
    """cache_hit | rewrite_cache_hit | rewrite_path | llm_answer | no_llm, from the server's own
    account of the request. A follow-up is rewritten (an LLM call) before the cache lookup, so
    a hit on its rewritten form still costs one LLM round trip and is kept apart from plain hits."""
    if summary is None:
        return "unknown"
    latency = summary.get("latency_ms", {})
    if summary.get("cache_hit"):
        return "rewrite_cache_hit" if "rewrite" in latency else "cache_hit"
    if "rewrite" in latency:
        return "rewrite_path"
    if "llm" in latency:
        return "llm_answer"
    return "no_llm"


def trace_id_for(request_id: str) -> str:
    """Langfuse trace id for a request id (the SDK hashes the seed with SHA-256)."""
    return hashlib.sha256(request_id.encode()).hexdigest()[:32]


# --- Report assembly ----------------------------------------------------------------

COMPONENTS = ["cache_lookup", "rewrite", "retrieval", "rerank", "llm"]


def tokens_by_request(observations: list[dict]) -> dict[str, dict]:
    """Token usage and model per trace id from Langfuse generation observations."""
    usage: dict[str, dict] = {}
    for obs in observations:
        if obs.get("type") != "GENERATION":
            continue
        entry = usage.setdefault(obs["traceId"], {"input": 0, "output": 0, "reasoning": 0, "total": 0, "models": set(), "calls": 0})
        details = obs.get("usageDetails") or {}
        entry["input"] += details.get("input", 0)
        entry["output"] += details.get("output", 0)
        entry["reasoning"] += details.get("output_reasoning", 0)
        entry["total"] += details.get("total", 0)
        entry["calls"] += 1
        if obs.get("model"):
            entry["models"].add(obs["model"])
    return usage


def build_report(rows: list[dict], summaries: dict[str, dict], usage: dict[str, dict]) -> dict:
    """Aggregate measured rows. `rows` hold e2e timings; component data comes from the server log."""
    measured = [r for r in rows if r["phase"] == "measured"]
    ok = [r for r in measured if r["ok"]]
    failed = [r for r in measured if not r["ok"]]
    for row in ok:
        summary = summaries.get(row["request_id"])
        row["path"] = request_path(summary)
        row["components_ms"] = (summary or {}).get("latency_ms", {})
        tokens = usage.get(trace_id_for(row["request_id"]))
        row["tokens"] = (
            {k: tokens[k] for k in ("input", "output", "reasoning", "total", "calls")} if tokens else None
        )

    def stats(group: list[dict]) -> dict:
        return summarize([r["e2e_ms"] for r in group])

    clean = [r for r in ok if not r.get("provider_retry")]
    by_path = {p: stats([r for r in ok if r["path"] == p]) for p in sorted({r["path"] for r in ok})}
    by_category = {c: stats([r for r in ok if r["category"] == c]) for c in sorted({r["category"] for r in ok})}
    by_role = {c: stats([r for r in ok if r["role"] == c]) for c in sorted({r["role"] for r in ok})}

    by_path["cache_hit_exact_repeats"] = stats([r for r in ok if r["path"] == "cache_hit" and r["category"] == "repeat"])
    unexpected_hits = [r["key"] for r in ok if r["path"] == "cache_hit" and r["category"] != "repeat"]
    llm_path = [r for r in ok if r["path"] == "llm_answer" and r["components_ms"].get("total")]
    breakdown = {}
    if llm_path:
        total = sum(r["components_ms"]["total"] for r in llm_path)
        for name in COMPONENTS:
            part = sum(r["components_ms"].get(name, 0.0) for r in llm_path)
            breakdown[name] = {"mean_ms": part / len(llm_path), "share_of_server_time": part / total}
        accounted = sum(v["share_of_server_time"] for v in breakdown.values())
        breakdown["other_in_server"] = {"mean_ms": (1 - accounted) * total / len(llm_path), "share_of_server_time": 1 - accounted}

    cold = [r for r in ok if r["path"] not in ("cache_hit", "rewrite_cache_hit") and r["components_ms"]]
    components = {}
    for name in [*COMPONENTS, "total"]:
        values = [r["components_ms"][name] for r in cold if name in r["components_ms"]]
        if values:
            components[name] = summarize(values)
    overhead = [r["e2e_ms"] - r["components_ms"]["total"] for r in ok if "total" in r["components_ms"]]

    llm_rows = [r for r in ok if r["tokens"]]
    token_stats = {
        key: summarize([r["tokens"][key] for r in llm_rows]) for key in ("input", "output", "reasoning", "total")
    }
    failure_kinds: dict[str, int] = {}
    for row in failed:
        failure_kinds[row["error_kind"]] = failure_kinds.get(row["error_kind"], 0) + 1

    return {
        "counts": {
            "warmup": len([r for r in rows if r["phase"] == "warmup"]),
            "measured_attempted": len(measured),
            "measured_ok": len(ok),
            "measured_failed": len(failed),
            "provider_retry_flagged": len([r for r in ok if r.get("provider_retry")]),
        },
        "error_rate": (len(failed) / len(measured)) if measured else None,
        "failure_kinds": failure_kinds,
        "e2e_ms": {
            "all_ok": stats(ok),
            "all_ok_without_provider_retries": stats(clean),
            "by_path": by_path,
            "by_category": by_category,
            "by_role": by_role,
        },
        "components_ms_cold_requests": components,
        "llm_answer_path_breakdown": breakdown,
        "unexpected_cache_hits": unexpected_hits,
        "http_and_auth_overhead_ms": summarize(overhead),
        "tokens_per_llm_request": token_stats,
        "tokens_total": {key: sum(r["tokens"][key] for r in llm_rows) for key in ("input", "output", "reasoning", "total")},
        "llm_requests_with_usage": len(llm_rows),
        "cache": {
            "hits": len([r for r in ok if r["path"] in ("cache_hit", "rewrite_cache_hit")]),
            "misses": len([r for r in ok if r["path"] not in ("cache_hit", "rewrite_cache_hit")]),
        },
    }


def format_report(meta: dict, report: dict) -> str:
    def ms(value) -> str:
        return "n/a" if value is None else f"{value:,.0f}"

    def row(name: str, s: dict) -> str:
        if not s.get("n"):
            return f"| {name} | 0 | | | | | | |"
        return f"| {name} | {s['n']} | {ms(s['mean'])} | {ms(s['p50'])} | {ms(s['p95'])} | {ms(s['p99'])} | {ms(s['min'])} | {ms(s['max'])} |"

    header = "| Group | n | mean | P50 | P95 | P99 | min | max |\n|---|---|---|---|---|---|---|---|"
    lines = [f"# Latency benchmark ({meta['run_id']})", "", "## Test conditions", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items() if k != "run_id"]
    c = report["counts"]
    lines += [
        "",
        "## Requests",
        "",
        f"- warm-up (excluded): {c['warmup']}",
        f"- measured attempted: {c['measured_attempted']}, succeeded: {c['measured_ok']}, failed: {c['measured_failed']}"
        f" (error rate {report['error_rate']:.1%})" if report["error_rate"] is not None else "- no measured requests",
        f"- failure kinds: {report['failure_kinds'] or 'none'}",
        f"- successful requests during which the provider SDK retried (latency includes a back-off): {c['provider_retry_flagged']}",
        "",
        "## End-to-end latency (ms, client-timed over HTTP; P95 needs n>=20, P99 needs n>=100)",
        "",
        header,
        row("all successful", report["e2e_ms"]["all_ok"]),
        row("all successful, no provider retry", report["e2e_ms"]["all_ok_without_provider_retries"]),
    ]
    lines += [row(f"path: {k}", v) for k, v in report["e2e_ms"]["by_path"].items()]
    lines += [row(f"category: {k}", v) for k, v in report["e2e_ms"]["by_category"].items()]
    lines += [row(f"role: {k}", v) for k, v in report["e2e_ms"]["by_role"].items()]
    lines += ["", "## Server-side components, uncached requests (ms)", "", header]
    lines += [row(k, v) for k, v in report["components_ms_cold_requests"].items()]
    lines += [row("HTTP + auth overhead (e2e - server total)", report["http_and_auth_overhead_ms"])]
    if report["llm_answer_path_breakdown"]:
        lines += ["", "## Where server time goes on the LLM-answer path", "", "| component | mean ms | share |", "|---|---|---|"]
        lines += [
            f"| {name} | {v['mean_ms']:,.0f} | {v['share_of_server_time']:.0%} |" for name, v in report["llm_answer_path_breakdown"].items()
        ]
    lines += ["", f"Requests answered from the cache on their first appearance in the run: {report['unexpected_cache_hits'] or 'none'}"]
    lines += ["", "## Tokens per LLM-using request (from Langfuse)", "", "| | n | mean | P50 | P95 | max |", "|---|---|---|---|---|---|"]
    for key, s in report["tokens_per_llm_request"].items():
        if s.get("n"):
            lines.append(f"| {key} | {s['n']} | {s['mean']:,.0f} | {s['p50']:,.0f} | {ms(s['p95'])} | {s['max']:,.0f} |")
    lines += ["", f"Total tokens: {report['tokens_total']}", f"Cache: {report['cache']}"]
    return "\n".join(lines) + "\n"


# --- Live run -------------------------------------------------------------------------


def _port_open(port: int) -> bool:
    import socket

    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_redis(port: int, workdir: Path):
    if _port_open(port):
        return None, f"existing Redis on port {port} (left running)"
    proc = subprocess.Popen(
        ["redis-stack-server", "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(workdir)],
        stdout=open(workdir / "redis.log", "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        if _port_open(port):
            return proc, "redis-stack-server started by the benchmark (no persistence)"
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("Redis Stack did not start")


def stop_redis(port: int, wrapper) -> None:
    """redis-stack-server is a wrapper script: ending it does not end the server it started,
    so ask the server itself to shut down (no persistence was configured)."""
    import redis

    try:
        redis.Redis(port=port).shutdown(nosave=True)
    except Exception:
        pass  # a successful SHUTDOWN drops the connection, which surfaces as an error
    wrapper.terminate()
    for _ in range(20):
        if not _port_open(port):
            return
        time.sleep(0.5)
    print(f"WARNING: Redis is still listening on port {port}; stop it manually")


def start_server(port: int, log_path: Path, redis_port: int):
    env = {
        **os.environ,
        "LANGFUSE_TRACING_ENVIRONMENT": "evaluation",
        "GROQ_LOG": "info",
        "PYTHONUNBUFFERED": "1",
        "CACHE_ENABLED": "true",
        "REDIS_URL": f"redis://localhost:{redis_port}",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    import requests

    for _ in range(360):
        if proc.poll() is not None:
            raise RuntimeError("API exited during startup; see the server log")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("API did not become healthy")


class Client:
    def __init__(self, base: str, refresh_s: float):
        import requests

        self.base = base
        self.http = requests.Session()
        self.refresh_s = refresh_s
        self._tokens: dict[str, tuple[str, float]] = {}

    def token(self, role: str, force: bool = False) -> str:
        cached = self._tokens.get(role)
        if cached and not force and time.time() - cached[1] < self.refresh_s:
            return cached[0]
        response = self.http.post(
            f"{self.base}/auth/login-json", json={"username": USERNAME_FOR_ROLE[role], "password": PASSWORD}, timeout=30
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        self._tokens[role] = (token, time.time())
        return token

    def chat(self, step: Step, phase: str, timeout_s: float) -> dict:
        import requests

        headers = {"Authorization": f"Bearer {self.token(step.role)}"}
        row = {**step.record(), "phase": phase, "ok": False, "request_id": None, "status": None, "error_kind": None}
        start_wall = time.time()
        started = time.perf_counter()
        try:
            response = self.http.post(
                f"{self.base}/chat", json={"question": step.question, "session_id": step.session}, headers=headers, timeout=timeout_s
            )
            elapsed = (time.perf_counter() - started) * 1000
            row.update(status=response.status_code, request_id=response.headers.get("X-Request-ID"))
            if response.status_code == 200:
                row["ok"] = True
                row["cache_hit_response"] = bool(response.json().get("cache_hit"))
            else:
                row["error_kind"] = f"http_{response.status_code}"
        except requests.Timeout:
            elapsed = (time.perf_counter() - started) * 1000
            row["error_kind"] = "client_timeout"
        except requests.RequestException as exc:
            elapsed = (time.perf_counter() - started) * 1000
            row["error_kind"] = type(exc).__name__
        row.update(e2e_ms=elapsed, start_wall=start_wall, end_wall=time.time())
        return row


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def fetch_langfuse(run_id: str, start: datetime, end: datetime) -> list[dict]:
    """All observations of this run's sessions from Langfuse (v2 API), in any environment,
    so a trace that missed the 'evaluation' environment would show up."""
    import requests
    from dotenv import dotenv_values

    env = {**dotenv_values(ROOT / ".env"), **{k: v for k, v in os.environ.items() if k.startswith("LANGFUSE_")}}
    base, auth = env["LANGFUSE_BASE_URL"].rstrip("/"), (env["LANGFUSE_PUBLIC_KEY"], env["LANGFUSE_SECRET_KEY"])
    flt = json.dumps(
        [
            {"type": "string", "column": "sessionId", "operator": "contains", "value": run_id},
        ]
    )
    params = {
        "fields": "core,basic,usage,model,metrics,io",
        "filter": flt,
        "limit": 1000,
        "fromStartTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toStartTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out: list[dict] = []
    cursor = None
    while True:
        response = requests.get(f"{base}/api/public/v2/observations", params={**params, **({"cursor": cursor} if cursor else {})}, auth=auth, timeout=60)
        response.raise_for_status()
        body = response.json()
        out += body.get("data", [])
        cursor = (body.get("meta") or {}).get("cursor")
        if not cursor:
            return out


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def run_live(args: argparse.Namespace) -> Path:
    """Warm up, then run the measured workload. Rows are appended to disk as they complete, so
    a failure later (or an early stop on quota) never loses the measurements."""
    from app.config import settings

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    workload = build_workload(load_benchmark_samples(), run_id=run_id)
    if args.categories:
        workload = [s for s in workload if s.category in args.categories.split(",")]
    if args.limit:
        workload = workload[: args.limit]
    composition: dict[str, int] = {}
    for step in workload:
        composition[step.category] = composition.get(step.category, 0) + 1
    print(f"Workload: {len(workload)} measured requests {composition}, seed {SEED}")

    workdir = Path(tempfile.mkdtemp(prefix="bench-"))  # private; holds the raw server log, which is never copied into the repo
    log_path, rows_path = workdir / "server.log", workdir / "rows.jsonl"
    print(f"Run directory: {workdir}")
    redis_proc, redis_note = start_redis(args.redis_port, workdir)
    meta = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "redis_note": redis_note,
        "pace": args.pace,
        "composition": composition,
        "abort_reason": None,
    }
    server = None
    try:
        server = start_server(args.port, log_path, args.redis_port)
        client = Client(f"http://127.0.0.1:{args.port}", refresh_s=min(480.0, settings.JWT_EXPIRE_MINUTES * 30.0))

        if not args.skip_warmup:
            print("Warm-up (excluded from statistics):")
            retrieval = {s["id"]: s for s in json.loads(RETRIEVAL_DATASET_PATH.read_text(encoding="utf-8"))}
            for i, (role, (sample_id, _)) in enumerate(WARMUP_IDS.items(), start=1):
                step = Step(seq=-i, key=f"warmup:{sample_id}", category="warmup", role=role, question=retrieval[sample_id]["question"], session=f"bench-{run_id}-warmup-{i}")
                row = client.chat(step, "warmup", args.timeout)
                append_row(rows_path, row)
                print(f"  warm-up {i}/6 {role:<11} ok={row['ok']} {row['e2e_ms']:,.0f} ms")
                time.sleep(args.pace if row["ok"] else 2)

        import redis

        cache = redis.Redis(port=args.redis_port)
        removed = sum(cache.delete(k) for k in cache.scan_iter("rbac_cache_*"))
        print(f"Cache emptied after warm-up ({removed} entries); measured phase starts cold.")

        pace, last_llm_start, consecutive_failures, seen_log = args.pace, 0.0, 0, 0
        for index, step in enumerate(workload, start=1):
            wait = last_llm_start + pace - time.time()
            if wait > 0:
                time.sleep(wait)
            row = client.chat(step, "measured", args.timeout)
            lines = read_lines(log_path)
            new_lines = lines[seen_log:]
            summaries = parse_request_summaries(new_lines)
            retried = parse_retry_times(new_lines)
            seen_log = len(lines)
            summary = summaries.get(row["request_id"])
            if not row["ok"]:
                if row["status"] == 500:
                    row["error_kind"] = classify_failure(new_lines)
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if summary and ("llm" in summary.get("latency_ms", {}) or "rewrite" in summary.get("latency_ms", {})):
                last_llm_start = row["start_wall"]
            if retried:
                pace = min(pace * 1.5, 60.0)  # the provider pushed back: slow down
            append_row(rows_path, row)
            if index % 5 == 0 or not row["ok"]:
                print(f"  {index:>3}/{len(workload)} {step.category:<12} ok={row['ok']} {row['e2e_ms']:>7,.0f} ms pace={pace:.0f}s")
            if row["error_kind"] == "provider_daily_token_limit" or consecutive_failures >= 3:
                meta["abort_reason"] = row["error_kind"] if row["error_kind"] == "provider_daily_token_limit" else "3 consecutive failures"
                print(f"Stopping early: {meta['abort_reason']}")
                break
    finally:
        if server is not None:
            server.send_signal(signal.SIGTERM)  # graceful: flushes queued traces
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
        if redis_proc is not None:
            stop_redis(args.redis_port, redis_proc)
        meta["ended_at"] = datetime.now(timezone.utc).isoformat()
        (workdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return workdir


def finalize(workdir: Path) -> int:
    """Parse the server log, fetch the traces from Langfuse, aggregate, and write the report."""
    from app.config import settings

    meta = json.loads((workdir / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (workdir / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
    lines = read_lines(workdir / "server.log")
    summaries = parse_request_summaries(lines)
    mark_retried(rows, parse_retry_times(lines))

    started = datetime.fromisoformat(meta["started_at"]).replace(microsecond=0)
    ended = datetime.fromisoformat(meta["ended_at"]).replace(microsecond=0) + timedelta(seconds=120)
    expected = len({trace_id_for(r["request_id"]) for r in rows if r["ok"] and r["request_id"]})
    observations: list[dict] = []
    for _attempt in range(12):  # Langfuse ingestion is eventually consistent
        observations = fetch_langfuse(meta["run_id"], started, ended)
        if len({o["traceId"] for o in observations}) >= expected:
            break
        time.sleep(10)

    report = build_report(rows, summaries, tokens_by_request(observations))
    environments = sorted({str(o.get("environment")) for o in observations})
    report_meta = {
        "run_id": meta["run_id"],
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}, Python {platform.python_version()}",
        "model": settings.GROQ_MODEL,
        "reranker": settings.RERANKER_MODEL,
        "embeddings": "all-MiniLM-L6-v2",
        "client": "sequential, 1 request in flight, real HTTP over loopback, requests.Session keep-alive",
        "server": "uvicorn, 1 worker, tracing on (Langfuse environment 'evaluation')",
        "cache": f"semantic cache on; {meta['redis_note']}",
        "pacing": f"min {meta['pace']:.0f}s between LLM-using requests (Groq 8,000 tokens/min limit)",
        "workload": f"{meta['composition']}, seed {SEED}",
        "stopped_early": meta["abort_reason"] or "no",
        "langfuse": (
            f"{len({o['traceId'] for o in observations})} traces for {expected} successful requests; "
            f"environments seen: {environments}; observations carrying input/output text: "
            f"{sum(1 for o in observations if o.get('input') or o.get('output'))}"
        ),
    }
    RESULTS_PATH.write_text(json.dumps({"meta": report_meta, "report": report, "rows": rows}, indent=2, default=str), encoding="utf-8")
    text = format_report(report_meta, report)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")
    return 0 if meta["abort_reason"] is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the workload composition and exit")
    parser.add_argument("--analyze", type=Path, help="re-run the analysis for an earlier run directory")
    parser.add_argument("--limit", type=int, help="only the first N measured requests (smoke run)")
    parser.add_argument("--categories", help="comma-separated categories to run, e.g. unauthorized (smoke run)")
    parser.add_argument("--skip-warmup", action="store_true", help="skip the warm-up requests (smoke run)")
    parser.add_argument("--pace", type=float, default=16.0, help="min seconds between starts of LLM-using requests")
    parser.add_argument("--timeout", type=float, default=120.0, help="client timeout per request, seconds")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    if args.analyze:
        return finalize(args.analyze)
    if args.dry_run:
        workload = build_workload(load_benchmark_samples())
        counts: dict[str, int] = {}
        for step in workload:
            counts[step.category] = counts.get(step.category, 0) + 1
        print(f"Workload: {len(workload)} measured requests {counts}, seed {SEED}")
        return 0
    return finalize(run_live(args))


if __name__ == "__main__":
    sys.exit(main())
