import json

from langchain_core.documents import Document

from app import rag
from app.config import settings
from app.services.bm25_retriever import BM25RetrieverService
from app.services.retrieval_policy import get_policy
from app.services.rrf import reciprocal_rank_fusion
from app.utils.logger import RequestContext

ALL_COLLECTIONS = ["finance", "marketing", "hr_docs", "engineering", "general"]


def _d(text: str) -> Document:
    return Document(page_content=text, metadata={"source": f"{text}.md"})


# --- retrieval policy -------------------------------------------------------


def test_hr_and_finance_use_hybrid():
    assert get_policy("hr_docs").strategy == "hybrid"
    assert get_policy("finance").strategy == "hybrid"


def test_marketing_engineering_general_use_semantic_without_bm25():
    for name in ("marketing", "engineering", "general"):
        policy = get_policy(name)
        assert policy.strategy == "semantic"
        assert policy.bm25_top_k == 0


def test_unknown_collection_defaults_to_semantic():
    assert get_policy("legal").strategy == "semantic"


# --- BM25 -------------------------------------------------------------------


def test_bm25_returns_empty_list_when_json_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path))
    assert BM25RetrieverService().retrieve("FINEMP1005", "hr_docs", top_k=15) == []


def test_bm25_returns_empty_list_when_json_is_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path))
    (tmp_path / "bm25_docs").mkdir()
    (tmp_path / "bm25_docs" / "hr_docs.json").write_text("{not json", encoding="utf-8")
    assert BM25RetrieverService().retrieve("FINEMP1005", "hr_docs", top_k=15) == []


def test_bm25_ranks_exact_identifier_match_first(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path))
    (tmp_path / "bm25_docs").mkdir()
    records = [
        {"page_content": f"employee_id: FINEMP10{i:02d}\nleave_balance: {i}", "metadata": {"source": "hr_data.csv", "role": "hr"}}
        for i in range(10)
    ]
    (tmp_path / "bm25_docs" / "hr_docs.json").write_text(json.dumps(records), encoding="utf-8")

    service = BM25RetrieverService()
    results = service.retrieve("What is the leave balance of FINEMP1005?", "hr_docs", top_k=3)

    assert "FINEMP1005" in results[0].page_content
    assert set(results[0].metadata) == {"source", "role", "bm25_score"}
    # Results are copies: per-request metadata never leaks into the cached index.
    results[0].metadata["poison"] = True
    assert "poison" not in service.retrieve("FINEMP1005", "hr_docs", top_k=1)[0].metadata


# --- Reciprocal Rank Fusion ---------------------------------------------------


def test_rrf_doc_in_both_lists_scores_higher_than_doc_in_one():
    a, b, c = _d("a"), _d("b"), _d("c")
    fused = reciprocal_rank_fusion([a, b], [_d("b"), c], k=60, top_n=10)

    assert fused[0].page_content == "b"
    scores = {d.page_content: d.metadata["rrf_score"] for d in fused}
    assert scores["b"] > scores["a"] and scores["b"] > scores["c"]
    assert abs(scores["b"] - (1 / 62 + 1 / 61)) < 1e-12


def test_rrf_deduplicates_identical_page_content():
    fused = reciprocal_rank_fusion([_d("same"), _d("other")], [_d("same")], k=60, top_n=10)
    contents = [d.page_content for d in fused]
    assert contents.count("same") == 1
    assert len(fused) == 2


def test_rrf_sets_rank_and_source_metadata():
    fused = {d.page_content: d.metadata for d in reciprocal_rank_fusion([_d("a"), _d("b")], [_d("b"), _d("c")])}
    assert (fused["a"]["dense_rank"], fused["a"]["bm25_rank"], fused["a"]["retrieval_source"]) == (1, None, "dense")
    assert (fused["b"]["dense_rank"], fused["b"]["bm25_rank"], fused["b"]["retrieval_source"]) == (2, 1, "both")
    assert (fused["c"]["dense_rank"], fused["c"]["bm25_rank"], fused["c"]["retrieval_source"]) == (None, 2, "bm25")


def test_rrf_respects_top_n_and_does_not_mutate_inputs():
    dense = [_d(str(i)) for i in range(10)]
    fused = reciprocal_rank_fusion(dense, [], top_n=3)
    assert len(fused) == 3
    assert "rrf_score" not in dense[0].metadata


# --- retrieve_for_collections (RBAC scope) ------------------------------------


def test_finance_retrieval_never_touches_other_collections(mock_chromadb):
    ctx = RequestContext()
    docs = rag.retrieve_for_collections("What was Q3 revenue?", ["finance"], None, ctx)

    assert mock_chromadb.touched() == {"finance"}
    assert all(d.metadata["role"] == "finance" for d in docs)
    for forbidden in ("hr_docs", "marketing", "engineering", "general"):
        assert forbidden not in mock_chromadb.touched()
    assert ctx.retrieval_strategy == "hybrid"
    assert ctx.dense_count > 0 and ctx.bm25_count > 0


def test_semantic_collection_does_not_call_bm25(mock_chromadb):
    ctx = RequestContext()
    rag.retrieve_for_collections("Kubernetes", ["engineering"], None, ctx)
    assert mock_chromadb.bm25_calls == []
    assert ctx.retrieval_strategy == "semantic"
    assert ctx.bm25_count == 0


def test_c_level_retrieval_fuses_all_collections(mock_chromadb):
    ctx = RequestContext()
    docs = rag.retrieve_for_collections("anything", ALL_COLLECTIONS, None, ctx)
    assert mock_chromadb.touched() == set(ALL_COLLECTIONS)
    assert len(docs) <= settings.RERANKER_CANDIDATE_COUNT
    assert {d.metadata["role"] for d in docs} == {"finance", "hr", "marketing", "engineering", "employee"}


def test_retrieval_with_no_collections_returns_nothing(mock_chromadb):
    assert rag.retrieve_for_collections("q", [], None, RequestContext()) == []
    assert mock_chromadb.touched() == set()
