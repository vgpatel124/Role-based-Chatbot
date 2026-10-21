"""Compare retrieval configurations on evaluation/eval_dataset.json.

Usage (from the project root):
    python evaluation/evaluate_retrieval.py [--k 5] [--verbose] [--check]

Every number printed is measured on this run; nothing is estimated. The LLM
is never called — this evaluates retrieval (and reranking) only.

Configurations:
    A  Dense only (original behaviour)
    B  Dense + Reranker
    C  Policy: hybrid for hr_docs/finance, semantic for others (no reranker)
    D  Hybrid + RRF for every collection (no reranker)
    E  Policy hybrid + RRF + Reranker (full production pipeline)

Relevance: a retrieved chunk is relevant when its source is in
expected_sources AND, if the sample sets expected_chunk_contains, the chunk
text contains that string. Without the second condition every chunk of a
single-file collection (hr_data.csv, engineering_master_doc.md, ...) would
count as relevant and the metrics would be meaningless.

Metrics over authorized samples (top K after each configuration):
    Recall@K     fraction of samples with >=1 relevant chunk in the top K
    Precision@K  relevant chunks in the top K / K, averaged over samples
    MRR@K        mean of 1/rank of the first relevant chunk (0 if none)
    KwCov@K      fraction of expected_answer_keywords present in the top-K text
    ms/query     mean wall-clock retrieval(+rerank) latency

Unauthorized samples: every configuration must return 0 chunks from a
collection the role is not authorized for ("RBAC leaks"). Configurations
with the reranker additionally report how many chunks passed the relevance
threshold (ideally 0 → the safe "unable to find" answer, no LLM call).
"""

import argparse
import json
import logging
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

from app.config import settings  # noqa: E402
from app.rag import _get_vectorstore, get_collections_for_role, retrieve_for_collections  # noqa: E402
from app.services.bm25_retriever import bm25_retriever  # noqa: E402
from app.services.reranker import reranker  # noqa: E402
from app.services.rrf import fuse_across_collections, reciprocal_rank_fusion  # noqa: E402
from app.utils.embeddings import get_embeddings  # noqa: E402
from app.utils.logger import RequestContext  # noqa: E402

DATASET_PATH = ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = ROOT / "evaluation" / "retrieval_results.json"
THRESHOLDS_PATH = ROOT / "evaluation" / "retrieval_thresholds.json"

# Chunk metadata "role" (set at ingest) → the collection it was stored in.
ROLE_METADATA_TO_COLLECTION = {
    "finance": "finance",
    "hr": "hr_docs",
    "marketing": "marketing",
    "engineering": "engineering",
    "employee": "general",
}


# --------------------------------------------------------------------------
# Retrieval configurations. Each takes (question, authorized collections) and
# returns a ranked list. All of them only ever search `collections`, which is
# produced by get_collections_for_role — the same RBAC gate as production.
# --------------------------------------------------------------------------


def _combine(per_collection: list[list[Document]]) -> list[Document]:
    if len(per_collection) > 1:
        return fuse_across_collections(
            per_collection, k=settings.RRF_K, top_n=settings.RERANKER_CANDIDATE_COUNT
        )
    return per_collection[0] if per_collection else []


def _dense(question: str, collection: str) -> list[Document]:
    store = _get_vectorstore(collection, get_embeddings())
    return store.similarity_search(question, k=settings.DENSE_TOP_K)


def _rerank(question: str, candidates: list[Document]) -> list[Document]:
    docs, _ = reranker.rerank(
        query=question,
        candidates=candidates,
        top_k=settings.RERANKER_FINAL_TOP_K,
        threshold=settings.RERANKER_SCORE_THRESHOLD,
    )
    return docs


def config_a_dense(question: str, collections: list[str]) -> list[Document]:
    return _combine([_dense(question, c) for c in collections])


def config_b_dense_rerank(question: str, collections: list[str]) -> list[Document]:
    return _rerank(question, config_a_dense(question, collections))


def config_c_policy(question: str, collections: list[str]) -> list[Document]:
    return retrieve_for_collections(question, collections, get_embeddings(), RequestContext())


def config_d_hybrid_all(question: str, collections: list[str]) -> list[Document]:
    per_collection = []
    for c in collections:
        dense = _dense(question, c)
        bm25 = bm25_retriever.retrieve(question, c, settings.BM25_TOP_K)
        per_collection.append(
            reciprocal_rank_fusion(dense, bm25, k=settings.RRF_K, top_n=settings.RRF_CANDIDATE_COUNT)
        )
    return _combine(per_collection)


def config_e_full(question: str, collections: list[str]) -> list[Document]:
    return _rerank(question, config_c_policy(question, collections))


CONFIGS: list[tuple[str, str, Callable[[str, list[str]], list[Document]], bool]] = [
    ("A", "Dense only", config_a_dense, False),
    ("B", "Dense + Reranker", config_b_dense_rerank, True),
    ("C", "Policy hybrid (HR/Fin), no reranker", config_c_policy, False),
    ("D", "Hybrid + RRF all, no reranker", config_d_hybrid_all, False),
    ("E", "Policy hybrid + RRF + Reranker", config_e_full, True),
]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def is_relevant(doc: Document, sample: dict) -> bool:
    if doc.metadata.get("source") not in sample["expected_sources"]:
        return False
    needle = sample.get("expected_chunk_contains")
    return needle is None or needle in doc.page_content


def chunk_collection(doc: Document) -> str | None:
    return ROLE_METADATA_TO_COLLECTION.get(doc.metadata.get("role"))


def score_sample(docs: list[Document], sample: dict, k: int) -> dict:
    top = docs[:k]
    flags = [is_relevant(d, sample) for d in top]
    first = next((i + 1 for i, f in enumerate(flags) if f), None)
    text = "\n".join(d.page_content for d in top).lower()
    keywords = sample.get("expected_answer_keywords", [])
    kw_found = [kw for kw in keywords if kw.lower() in text]
    return {
        "hit": first is not None,
        "precision": sum(flags) / k,
        "rr": 1.0 / first if first else 0.0,
        "first_relevant_rank": first,
        "kw_coverage": len(kw_found) / len(keywords) if keywords else None,
        "keywords_missing": [kw for kw in keywords if kw not in kw_found],
        "returned": len(top),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def check_rbac(samples: list[dict]) -> list[str]:
    """The dataset's expected_collections must match the live RBAC mapping."""
    problems = []
    for s in samples:
        actual = get_collections_for_role(s["role"])
        if sorted(actual) != sorted(s["expected_collections"]):
            problems.append(f"{s['id']}: role {s['role']} → {actual}, dataset says {s['expected_collections']}")
        target = s.get("unauthorized_target")
        if target and target in actual:
            problems.append(f"{s['id']}: role {s['role']} is actually authorized for {target}")
    return problems


def run_config(key: str, fn: Callable, samples: list[dict], k: int, verbose: bool) -> dict:
    per_sample = []
    latencies = []
    leaks = 0
    unauthorized_passing = []
    for s in samples:
        collections = get_collections_for_role(s["role"])
        t0 = time.perf_counter()
        docs = fn(s["question"], collections)
        latencies.append((time.perf_counter() - t0) * 1000)

        leaked = [d for d in docs if chunk_collection(d) not in collections]
        leaks += len(leaked)

        if s["category"] == "unauthorized":
            unauthorized_passing.append(len(docs))
            per_sample.append({"id": s["id"], "returned": len(docs), "rbac_leaks": len(leaked)})
            if verbose:
                print(f"    [{key}] {s['id']:28} returned={len(docs):2} leaks={len(leaked)}")
            continue

        m = score_sample(docs, s, k)
        m.update({"id": s["id"], "category": s["category"], "rbac_leaks": len(leaked)})
        per_sample.append(m)
        if verbose:
            print(
                f"    [{key}] {s['id']:28} hit={str(m['hit']):5} first_rank={m['first_relevant_rank']} "
                f"P={m['precision']:.2f} kw={m['kw_coverage']:.2f} returned={m['returned']} "
                f"missing={m['keywords_missing']}"
            )

    scored = [m for m in per_sample if "hit" in m]
    return {
        "recall": _mean([1.0 if m["hit"] else 0.0 for m in scored]),
        "precision": _mean([m["precision"] for m in scored]),
        "mrr": _mean([m["rr"] for m in scored]),
        "kw_coverage": _mean([m["kw_coverage"] for m in scored if m["kw_coverage"] is not None]),
        "latency_ms": _mean(latencies),
        "rbac_leaks": leaks,
        "unauthorized_returned": unauthorized_passing,
        "per_sample": per_sample,
    }


def check_gate(results: dict[str, dict], samples: list[dict]) -> list[str]:
    """Regression gate for the production configuration (see retrieval_thresholds.json).

    Besides the aggregate floors, a sample may set "max_first_rank" in the dataset:
    its first relevant chunk must appear at or above that rank. Used for queries
    where a generic summary chunk competes with the specific one at near-tied
    reranker scores, so a chunking or reranking change can silently push them down.
    """
    gate = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    config = results.get(gate["config"], {})
    if "error" in config:
        return [f"config {gate['config']} failed to run"]
    failures = []
    if config["recall"] < gate["min_recall"]:
        failures.append(f"Recall: {config['recall']:.3f} < {gate['min_recall']}")
    if config["mrr"] < gate["min_mrr"]:
        failures.append(f"MRR: {config['mrr']:.3f} < {gate['min_mrr']}")
    if config["rbac_leaks"] > gate["max_rbac_leaks"]:
        failures.append(f"RBAC leaks: {config['rbac_leaks']} > {gate['max_rbac_leaks']}")
    ranks = {r["id"]: r.get("first_relevant_rank") for r in config["per_sample"]}
    for sample in samples:
        limit = sample.get("max_first_rank")
        if limit is None:
            continue
        rank = ranks.get(sample["id"])
        if rank is None or rank > limit:
            failures.append(f"{sample['id']}: first relevant chunk at rank {rank}, required <= {limit}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the production config is below evaluation/retrieval_thresholds.json")
    parser.add_argument("--k", type=int, default=5, help="cut-off K for the metrics (default 5)")
    parser.add_argument("--verbose", action="store_true", help="print per-sample results")
    args = parser.parse_args()
    k = args.k

    # Keep the app's INFO logs (BM25 loads, cache) out of the report.
    logging.getLogger("app").setLevel(logging.WARNING)

    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    n_auth = sum(1 for s in samples if s["category"] != "unauthorized")
    n_unauth = len(samples) - n_auth
    print(f"Dataset: {len(samples)} samples ({n_auth} authorized, {n_unauth} unauthorized) | K={k}")
    print(
        f"Reranker: {settings.RERANKER_MODEL} | threshold={settings.RERANKER_SCORE_THRESHOLD} "
        f"| final_top_k={settings.RERANKER_FINAL_TOP_K}"
    )
    print("ragas: not used (retrieval-only evaluation)")

    rbac_problems = check_rbac(samples)
    if rbac_problems:
        print("\nRBAC MAPPING MISMATCH — dataset does not match get_collections_for_role:")
        for p in rbac_problems:
            print("  -", p)
        return 1
    print("RBAC mapping check: OK (dataset expected_collections == get_collections_for_role)\n")

    # Untimed warm-up: load the embedding model, open every vector store,
    # build every BM25 index and run the cross-encoder once, so one-off
    # startup cost is not charged to whichever configuration runs first.
    for collection in sorted({c for s in samples for c in get_collections_for_role(s["role"])}):
        config_e_full("warm-up query", [collection])
        bm25_retriever.retrieve("warm-up query", collection, 1)

    results: dict[str, dict] = {}
    for key, name, fn, _ in CONFIGS:
        if args.verbose:
            print(f"  Config {key}: {name}")
        try:
            results[key] = {"name": name, **run_config(key, fn, samples, k, args.verbose)}
        except Exception as e:
            results[key] = {"name": name, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()

    header = f"{'Config':<7}| {'Recall@' + str(k):<9}| {'Precision@' + str(k):<12}| {'MRR':<6}| {'KwCov@' + str(k):<9}| {'ms/query':<9}| {'RBAC leaks':<11}| Description"
    print(header)
    print("-" * len(header))
    for key, name, _, _uses_reranker in CONFIGS:
        r = results[key]
        if "error" in r:
            print(f"{key:<7}| FAILED — {r['error']}")
            continue
        print(
            f"{key:<7}| {r['recall']:<9.3f}| {r['precision']:<12.3f}| {r['mrr']:<6.3f}| "
            f"{r['kw_coverage']:<9.3f}| {r['latency_ms']:<9.1f}| {r['rbac_leaks']:<11}| {name}"
        )

    print("\nUnauthorized-access samples (chunks returned per sample; wrong-department chunks must be 0):")
    unauth_ids = [s["id"] for s in samples if s["category"] == "unauthorized"]
    for key, _name, _, uses_reranker in CONFIGS:
        r = results[key]
        if "error" in r:
            continue
        note = "after reranker threshold" if uses_reranker else "authorized-collection candidates (no threshold)"
        counts = ", ".join(f"{i}={n}" for i, n in zip(unauth_ids, r["unauthorized_returned"]))
        print(f"  {key}: leaks={r['rbac_leaks']} | {counts} ({note})")

    total_leaks = sum(r.get("rbac_leaks", 0) for r in results.values())
    print(f"\nRBAC verdict: {'PASS — 0 unauthorized chunks in any configuration' if total_leaks == 0 else f'FAIL — {total_leaks} unauthorized chunks'}")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "k": k,
                "reranker_model": settings.RERANKER_MODEL,
                "reranker_threshold": settings.RERANKER_SCORE_THRESHOLD,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Per-sample results written to {RESULTS_PATH.relative_to(ROOT)}")
    failed = any("error" in r for r in results.values())
    gate_failures = check_gate(results, samples) if args.check else []
    if args.check:
        if gate_failures:
            print("\nRetrieval gate FAILED:\n  " + "\n  ".join(gate_failures))
        else:
            print("\nRetrieval gate passed (evaluation/retrieval_thresholds.json)")
    return 1 if failed or total_leaks or gate_failures else 0


if __name__ == "__main__":
    sys.exit(main())
