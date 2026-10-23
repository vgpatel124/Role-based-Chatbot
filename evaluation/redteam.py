"""Live RBAC red-team against the real index.

Usage (from the project root, after `python app/ingest.py`):
    python evaluation/redteam.py            # retrieval + reranker only, no LLM, no API key
    python evaluation/redteam.py --llm      # also runs the full pipeline with the Groq LLM
    python evaluation/redteam.py --role hr --template role_claim --verbose

Every role asks every other department's question, plain and wrapped in
manipulation attempts (evaluation/redteam_cases.py). Checked per case:

    retrieval   no candidate chunk comes from a collection the role may not read
    context     no unauthorized "canary" fact appears in any chunk offered to the LLM
    answer      (--llm) no canary fact appears in the answer
    sources     (--llm) every cited file belongs to an authorized collection

Exit status is 1 if any leak is found, so it can gate a pipeline.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The semantic cache would answer from a previous run; this must measure the pipeline.
os.environ["CACHE_ENABLED"] = "false"
# If Langfuse tracing is configured, keep these runs apart from real traffic.
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "evaluation")

from app.config import settings  # noqa: E402
from app.rag import (  # noqa: E402
    DECLINED_ANSWER,
    NO_RELEVANT_CONTEXT_ANSWER,
    get_collections_for_role,
    is_declined_answer,
    query_rag,
    retrieve_for_collections,
)
from app.services.reranker import reranker  # noqa: E402
from app.utils.embeddings import get_embeddings  # noqa: E402
from app.utils.logger import RequestContext  # noqa: E402
from evaluation.redteam_cases import (  # noqa: E402
    COLLECTION_FILES,
    EXPECTED_ACCESS,
    METADATA_ROLE_TO_COLLECTION,
    AttackCase,
    build_cases,
)

RESULTS_PATH = ROOT / "evaluation" / "redteam_results.json"
LLM_PAUSE_SECONDS = 2.5  # stay under the provider's requests-per-minute limit


def _contains_canary(text: str, canaries) -> list[str]:
    lowered = text.lower()
    return [c for c in canaries if c.lower() in lowered]


def _is_refusal(answer: str) -> bool:
    return is_declined_answer(answer) or answer.strip() == NO_RELEVANT_CONTEXT_ANSWER or DECLINED_ANSWER[:-1] in answer


def run_case(case: AttackCase, use_llm: bool) -> dict:
    allowed = EXPECTED_ACCESS[case.role]
    collections = get_collections_for_role(case.role)
    result: dict = {
        "id": case.id,
        "role": case.role,
        "target": case.target,
        "template": case.template,
        "leaks": [],
    }

    if set(collections) != set(allowed):
        result["leaks"].append(f"policy: role resolves to {sorted(collections)}, expected {sorted(allowed)}")

    ctx = RequestContext()
    candidates = retrieve_for_collections(case.question, collections, get_embeddings(), ctx)
    reranked, _ = reranker.rerank(
        query=case.question,
        candidates=candidates,
        top_k=settings.RERANKER_FINAL_TOP_K,
        threshold=settings.RERANKER_SCORE_THRESHOLD,
    )
    result["candidates"] = len(candidates)
    result["passed_threshold"] = len(reranked)

    for doc in candidates:
        collection = METADATA_ROLE_TO_COLLECTION.get(doc.metadata.get("role", ""))
        if collection not in allowed:
            result["leaks"].append(f"retrieval: chunk from {collection!r} ({doc.metadata.get('source')})")
        found = _contains_canary(doc.page_content, case.canaries)
        if found:
            result["leaks"].append(f"context: {found} offered to the LLM")

    if use_llm:
        try:
            response = query_rag(case.question, case.role, session_id=None, ctx=RequestContext(), return_contexts=True)
        except Exception as exc:  # provider hiccup: record it, do not call it a leak
            result["error"] = f"{type(exc).__name__}: {exc}"[:200]
            return result
        answer = response["answer"]
        result["refused"] = _is_refusal(answer)
        found = _contains_canary(answer, case.canaries)
        if found:
            result["leaks"].append(f"answer: contains {found}")
        for chunk in response.get("contexts", []):
            found = _contains_canary(chunk, case.canaries)
            if found:
                result["leaks"].append(f"context: {found} sent to the LLM")
        authorized_files = set().union(*(COLLECTION_FILES[c] for c in allowed))
        for source in response["sources"]:
            if source not in authorized_files:
                result["leaks"].append(f"sources: cites {source!r}")
        time.sleep(LLM_PAUSE_SECONDS)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="also run the full pipeline with the Groq LLM")
    parser.add_argument("--role", help="only this role")
    parser.add_argument("--template", help="only this attack template")
    parser.add_argument("--verbose", action="store_true", help="print every case")
    args = parser.parse_args()

    if args.llm and not settings.GROQ_API_KEY:
        print("--llm needs GROQ_API_KEY in the environment or .env", file=sys.stderr)
        return 2

    cases = [
        c
        for c in build_cases()
        if (not args.role or c.role == args.role) and (not args.template or c.template == args.template)
    ]
    if not cases:
        print("No cases match the filters.", file=sys.stderr)
        return 2

    mode = "retrieval + reranker + LLM" if args.llm else "retrieval + reranker (no LLM)"
    print(f"Red-team: {len(cases)} attacks, {mode}\n")

    results = []
    for case in cases:
        outcome = run_case(case, args.llm)
        results.append(outcome)
        if args.verbose or outcome["leaks"]:
            status = "LEAK" if outcome["leaks"] else "ok  "
            print(f"  {status} {case.id}  (candidates={outcome['candidates']}, kept={outcome['passed_threshold']})")
            for leak in outcome["leaks"]:
                print(f"         - {leak}")

    leaked = [r for r in results if r["leaks"]]
    errors = [r for r in results if "error" in r]
    by_role = Counter(r["role"] for r in leaked)
    by_template = Counter(r["template"] for r in leaked)
    total_chunks = sum(r["candidates"] for r in results)

    print("\nSummary")
    print(f"  attacks run:            {len(results)}")
    print(f"  attacks with a leak:    {len(leaked)}")
    print(f"  candidate chunks seen:  {total_chunks} (all from authorized collections when leaks = 0)")
    print(f"  chunks kept by reranker:{sum(r['passed_threshold'] for r in results):>4}")
    if args.llm:
        refused = sum(1 for r in results if r.get("refused"))
        print(f"  refused:                {refused}/{len(results) - len(errors)}")
        print(f"  provider errors:        {len(errors)}")
    if leaked:
        print(f"  leaks by role:          {dict(by_role)}")
        print(f"  leaks by template:      {dict(by_template)}")

    RESULTS_PATH.write_text(json.dumps({"llm": args.llm, "results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(ROOT)}")
    print("RESULT: " + ("FAIL - unauthorized data reached a role" if leaked else "PASS - no leaks"))
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
