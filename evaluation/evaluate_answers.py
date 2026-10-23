"""Answer-level evaluation of the full RAG pipeline (retrieval + generation).

Usage (from the project root, after `python app/ingest.py`):
    pip install -r requirements-eval.txt
    python evaluation/evaluate_answers.py                  # everything
    python evaluation/evaluate_answers.py --no-ragas       # deterministic checks only (cheap)
    python evaluation/evaluate_answers.py --check          # exit 1 if below evaluation/thresholds.json
    python evaluation/evaluate_answers.py --reuse          # reuse answers from the last run, only re-score
    python evaluation/evaluate_answers.py --ragas-limit 12 # judge an evenly spaced subset of 12 answers
    python evaluation/evaluate_answers.py --limit 5 --role finance

Runs every question in evaluation/answer_dataset.json through query_rag with
the semantic cache off, as the question's role, and scores the result.

Deterministic metrics (no judge, free, reproducible apart from LLM sampling):
    refusal_accuracy    answerable questions are answered; out-of-scope ones are declined
    keyword_coverage    share of the expected facts present in the answer
    source_hit_rate     share of answers that cite at least one expected document

RAGAS metrics, scored by a judge LLM (answerable, non-declined samples only):
    faithfulness        claims in the answer that are supported by the retrieved context
    answer_relevancy    how directly the answer addresses the question
    context_precision   retrieved chunks that are relevant, with relevant ones ranked first
    context_recall      share of the reference answer that the retrieved context supports

The judge is a different model family from the generator on purpose
(EVAL_JUDGE_MODEL, default Alibaba's qwen3 vs. the app's OpenAI gpt-oss), so the
generator does not grade its own work.

Both models are rate limited (Groq free tier: 8,000 tokens/minute), so a full
run takes tens of minutes; --reuse and --ragas-limit keep iteration cheap.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["CACHE_ENABLED"] = "false"  # measure the pipeline, not yesterday's answers
# If Langfuse tracing is configured, keep these runs apart from real traffic.
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "evaluation")

from app.config import settings  # noqa: E402
from app.rag import NO_RELEVANT_CONTEXT_ANSWER, is_declined_answer, query_rag  # noqa: E402
from app.utils.embeddings import get_embeddings  # noqa: E402
from app.utils.logger import RequestContext  # noqa: E402

DATASET_PATH = ROOT / "evaluation" / "answer_dataset.json"
THRESHOLDS_PATH = ROOT / "evaluation" / "thresholds.json"
RESULTS_PATH = ROOT / "evaluation" / "answer_results.json"

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "qwen/qwen3.8-27b")
PAUSE_SECONDS = 2.0  # keep under the provider's requests-per-minute limit
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def is_refusal(answer: str) -> bool:
    return is_declined_answer(answer) or answer.strip() == NO_RELEVANT_CONTEXT_ANSWER or "don't have access" in answer


_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")


def _normalize(text: str) -> str:
    """Compare facts, not typography: the model writes narrow no-break spaces,
    non-breaking hyphens, markdown bold and thousands separators."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    text = text.lower().replace("*", "").replace(",", "").replace("\u00d7", "x")
    text = " ".join(text.split()).replace(" %", "%")
    return re.sub(r"(\d) ([mbk])\b", r"\1\2", text)  # "$30 M" -> "$30m".replace(" x", "x")  # "3.75 ×" -> "3.75x"


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text = _normalize(answer)
    # "a|b" means either spelling counts ("$25M|$25 million").
    return sum(any(_normalize(alt) in text for alt in k.split("|")) for k in keywords) / len(keywords)


_RETRY_AFTER = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE)
MAX_WAIT_SECONDS = 90  # longer waits mean a per-day budget, which retrying cannot fix


def _error_text(exc: Exception) -> str:
    """Error for the report: organisation ids redacted, long enough to keep the limit details."""
    return re.sub(r"org_\w+", "org_<redacted>", f"{type(exc).__name__}: {exc}")[:600]


def _rate_limit_wait(exc: Exception) -> float | None:
    """Seconds the provider asks us to wait, or None when this is not a rate limit."""
    text = str(exc)
    if "429" not in text and "rate limit" not in text.lower():
        return None
    match = _RETRY_AFTER.search(text)
    if not match:
        return 15.0
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def ask(sample: dict, attempts: int = 6) -> dict:
    """Run the pipeline for one sample. Short rate limits are waited out as the
    provider asks; a per-day limit (or any other failure) is reported, not retried."""
    for attempt in range(1, attempts + 1):
        try:
            return query_rag(
                sample["question"], sample["role"], session_id=None, ctx=RequestContext(), return_contexts=True
            )
        except Exception as exc:
            wait = _rate_limit_wait(exc)
            if wait is None:
                wait = 10.0 * attempt
            if attempt == attempts or wait > MAX_WAIT_SECONDS:
                return {"answer": "", "sources": [], "contexts": [], "error": _error_text(exc)}
            time.sleep(wait + 1)
    raise AssertionError("unreachable")


def load_previous() -> dict[str, dict]:
    """Answers from the last run, by sample id (errors excluded)."""
    if not RESULTS_PATH.exists():
        return {}
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8")).get("rows", [])
    return {r["id"]: r for r in rows if not r.get("error")}


def score_deterministic(samples: list[dict], previous: dict[str, dict] | None = None) -> list[dict]:
    rows = []
    previous = previous or {}
    for i, sample in enumerate(samples, 1):
        reused = previous.get(sample["id"])
        print(f"  [{i}/{len(samples)}] {sample['role']:<11} {sample['id']}{' (reused)' if reused else ''}", flush=True)
        if reused and reused["question"] == sample["question"]:
            response = {"answer": reused["answer"], "sources": reused["sources"], "contexts": reused["contexts"]}
        else:
            response = ask(sample)
            time.sleep(PAUSE_SECONDS)
        answer = response["answer"]
        refused = is_refusal(answer) if "error" not in response else None
        answerable = sample["type"] == "answerable"
        row = {
            **sample,
            "answer": answer,
            "sources": response["sources"],
            "contexts": response.get("contexts", []),
            "error": response.get("error"),
            "refused": refused,
            "refusal_correct": None if refused is None else (refused != answerable),
            "keyword_coverage": (
                keyword_coverage(answer, sample["keywords"]) if answerable and refused is False else None
            ),
            "source_hit": (
                bool(set(sample["expected_sources"]) & set(response["sources"]))
                if answerable and refused is False
                else None
            ),
        }
        rows.append(row)
    return rows


def run_ragas(rows: list[dict], limit: int | None = None) -> int:
    """Add RAGAS scores to the rows that can be judged. Returns how many were judged."""
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    judged = [r for r in rows if r["type"] == "answerable" and r["refused"] is False and r["contexts"]]
    if limit and limit < len(judged):
        step = len(judged) / limit  # evenly spaced, so every department stays represented
        judged = [judged[int(i * step)] for i in range(limit)]
    if not judged:
        return 0

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["reference"],
            )
            for r in judged
        ]
    )
    judge = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0, groq_api_key=settings.GROQ_API_KEY))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            # Groq rejects n>1 completions, which strictness>1 needs.
            ResponseRelevancy(strictness=1),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, timeout=180, max_retries=8),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()
    columns = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "llm_context_precision_with_reference",
        "context_recall": "context_recall",
    }
    for row, (_, scores) in zip(judged, frame.iterrows()):
        for metric, column in columns.items():
            value = scores.get(column)
            row[metric] = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
    return len(judged)


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict]) -> dict[str, float | None]:
    answerable = [r for r in rows if r["type"] == "answerable"]
    out_of_scope = [r for r in rows if r["type"] != "answerable"]
    summary = {
        "refusal_accuracy": mean([float(r["refusal_correct"]) for r in rows if r["refusal_correct"] is not None]),
        "answer_rate": mean([float(not r["refused"]) for r in answerable if r["refused"] is not None]),
        "decline_rate_out_of_scope": mean([float(r["refused"]) for r in out_of_scope if r["refused"] is not None]),
        "keyword_coverage": mean([r["keyword_coverage"] for r in answerable]),
        "source_hit_rate": mean([float(r["source_hit"]) for r in answerable if r["source_hit"] is not None]),
    }
    for metric in RAGAS_METRICS:
        summary[metric] = mean([r.get(metric) for r in answerable])
    return summary


def inconclusive(rows: list[dict], judged: int) -> list[str]:
    """Reasons the run cannot support a verdict. These are not quality failures:
    the affected samples are excluded from the metrics, and --check still fails."""
    reasons = []
    errored = [r["id"] for r in rows if r["error"]]
    if errored:
        reasons.append(f"{len(errored)} of {len(rows)} answers failed with provider errors ({', '.join(errored)})")
    for metric in RAGAS_METRICS:
        scored = sum(1 for r in rows if r.get(metric) is not None)
        if judged and scored < judged / 2:
            reasons.append(f"{metric} was scored for only {scored} of {judged} judged answers")
    return reasons


def fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-ragas", action="store_true", help="skip the judge-LLM metrics")
    parser.add_argument("--check", action="store_true", help="fail if a metric is below evaluation/thresholds.json")
    parser.add_argument("--reuse", action="store_true", help="reuse answers from the last run instead of regenerating")
    parser.add_argument("--ragas-limit", type=int, help="judge only N evenly spaced answers (saves tokens)")
    parser.add_argument("--ids", help="only these sample ids (comma-separated)")
    parser.add_argument("--role", help="only samples for this role")
    parser.add_argument("--limit", type=int, help="only the first N samples")
    args = parser.parse_args()

    if not settings.GROQ_API_KEY:
        print("GROQ_API_KEY is required (environment or .env).", file=sys.stderr)
        return 2

    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        samples = [s for s in samples if s["id"] in wanted]
    if args.role:
        samples = [s for s in samples if s["role"] == args.role]
    if args.limit:
        samples = samples[: args.limit]

    print(f"Generating {len(samples)} answers with {settings.GROQ_MODEL} (cache off)")
    rows = score_deterministic(samples, load_previous() if args.reuse else None)

    if not args.no_ragas:
        print(f"\nScoring with RAGAS, judge = {JUDGE_MODEL}")
        judged = run_ragas(rows, args.ragas_limit)
        print(f"  judged {judged} answerable samples")

    summary = summarize(rows)

    print("\nPer-sample problems")
    problems = 0
    for r in rows:
        notes = []
        if r["error"]:
            notes.append(f"error: {r['error']}")
        elif r["refusal_correct"] is False:
            notes.append("answered an out-of-scope question" if r["type"] != "answerable" else "declined an answerable question")
        elif r["keyword_coverage"] is not None and r["keyword_coverage"] < 1:
            notes.append(f"keyword coverage {r['keyword_coverage']:.2f}")
        elif r["source_hit"] is False:
            notes.append(f"expected source missing, cited {r['sources']}")
        for metric in RAGAS_METRICS:
            if r.get(metric) is not None and r[metric] < 0.7:
                notes.append(f"{metric} {r[metric]:.2f}")
        if notes:
            problems += 1
            print(f"  {r['id']}: {'; '.join(notes)}")
    if not problems:
        print("  none")

    print("\nSummary")
    for name, value in summary.items():
        print(f"  {name:<28}{fmt(value)}")
    reasons = inconclusive(rows, judged if not args.no_ragas else 0)
    if reasons:
        print("\nINCONCLUSIVE (excluded from the metrics above, not counted as quality failures):")
        for reason in reasons:
            print(f"  - {reason}")
        first_error = next((r["error"] for r in rows if r["error"]), None)
        if first_error:
            print(f"  first error: {first_error}")

    RESULTS_PATH.write_text(
        json.dumps(
            {"generator": settings.GROQ_MODEL, "judge": None if args.no_ragas else JUDGE_MODEL, "summary": summary, "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS_PATH.relative_to(ROOT)}")

    if args.check:
        thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
        failed = [
            f"{name}: {summary[name]:.3f} < {minimum}"
            for name, minimum in thresholds.items()
            if summary.get(name) is not None and summary[name] < minimum
        ]
        if failed:
            print("\nBelow threshold:\n  " + "\n  ".join(failed))
            return 1
        if reasons:
            print("\nThresholds were met on the samples that could be scored, but the run is inconclusive.")
            return 1
        print("\nAll metrics meet evaluation/thresholds.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
