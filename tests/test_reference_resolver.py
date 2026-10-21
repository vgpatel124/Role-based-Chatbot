import json

import pytest
from langchain_core.documents import Document

from app.config import settings
from app.services import reference_resolver
from app.services.bm25_retriever import BM25RetrieverService
from app.services.reference_resolver import resolve_references


def _row(emp_id: str, manager_id: str, source: str = "hr_data.csv", role: str = "hr") -> Document:
    return Document(
        page_content=f"HR Data › employee_id: {emp_id}\n\nemployee_id: {emp_id}\nmanager_id: {manager_id}\n",
        metadata={"source": source, "role": role},
    )


@pytest.fixture
def hr_rows(monkeypatch):
    rows = {r.page_content.split("employee_id: ")[2].split("\n")[0]: r for r in (_row("FINEMP1007", "FINEMP1000"), _row("FINEMP1000", ""))}
    calls = []

    def fake_find_row(collection, field, value):
        calls.append((collection, field, value))
        return rows.get(value)

    monkeypatch.setattr(reference_resolver.bm25_retriever, "find_row", fake_find_row)
    return calls


def test_manager_row_is_resolved_from_the_authorized_hr_collection(hr_rows):
    resolved = resolve_references([_row("FINEMP1010", "FINEMP1007")], ["hr_docs"])

    assert [d.page_content.split("employee_id: ")[2].split("\n")[0] for d in resolved] == ["FINEMP1007"]
    assert hr_rows == [("hr_docs", "employee_id", "FINEMP1007")]


def test_nothing_is_looked_up_when_hr_is_not_authorized(hr_rows):
    for collections in (["finance"], ["general"], ["marketing", "engineering"]):
        assert resolve_references([_row("FINEMP1010", "FINEMP1007")], collections) == []
    assert hr_rows == []


def test_only_hr_rows_are_followed(hr_rows):
    lookalike = _row("FINEMP1010", "FINEMP1007", source="notes.md", role="employee")
    assert resolve_references([lookalike], ["hr_docs", "general"]) == []
    assert hr_rows == []


def test_rows_already_in_context_or_unknown_are_skipped(hr_rows):
    manager = _row("FINEMP1007", "FINEMP1000")
    assert resolve_references([_row("FINEMP1010", "FINEMP1007"), manager], ["hr_docs"])[0].page_content != manager.page_content
    assert resolve_references([_row("FINEMP1010", "FINEMP9999")], ["hr_docs"]) == []


def test_resolution_is_capped(hr_rows):
    docs = [_row(f"FINEMP20{i}", "FINEMP1007") for i in range(5)] + [_row("FINEMP1007", "FINEMP1000")]
    assert len(resolve_references(docs, ["hr_docs"])) <= reference_resolver.MAX_RESOLVED_ROWS


def test_bm25_find_row_is_an_exact_line_match(tmp_path, monkeypatch):
    bm25_dir = tmp_path / "bm25_docs"
    bm25_dir.mkdir()
    records = [
        {"page_content": f"employee_id: FINEMP{n}\nfull_name: Person {n}\n", "metadata": {"source": "hr_data.csv", "role": "hr"}}
        for n in (1007, 10071)
    ]
    (bm25_dir / "hr_docs.json").write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path))

    service = BM25RetrieverService()
    assert "Person 1007" in service.find_row("hr_docs", "employee_id", "FINEMP1007").page_content
    assert service.find_row("hr_docs", "employee_id", "FINEMP100") is None
    assert service.find_row("missing_collection", "employee_id", "FINEMP1007") is None
