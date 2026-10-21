"""Follow employee-id references between retrieved HR rows.

An HR row holds "manager_id: FINEMP1007", not the manager's name or role, so
"who is the manager of FINEMP1010?" needs a second row (FINEMP1007's) that
shares no words with the question and ranks far below the retrieval cutoff.
After reranking, the rows that were retrieved are scanned for such ids and the
rows they point to are added to the context.

RBAC: this adds no access. A row is looked up only in the collection the
referencing chunk was retrieved from, and only if that collection is in the
caller's already-filtered `collections` list.
"""

import re

from langchain_core.documents import Document

from app.services.bm25_retriever import bm25_retriever

# (collection, id field in the referencing row, field that identifies a row)
REFERENCE_FIELDS: dict[str, tuple[str, str]] = {
    "hr_docs": ("manager_id", "employee_id"),
}
MAX_RESOLVED_ROWS = 2

_ID = r"[A-Za-z0-9_-]+"


def resolve_references(docs: list[Document], collections: list[str]) -> list[Document]:
    """Rows referenced by `docs` (e.g. their managers) that are not already in `docs`."""
    resolved: list[Document] = []
    present = {doc.page_content for doc in docs}
    for collection, (ref_field, key_field) in REFERENCE_FIELDS.items():
        if collection not in collections:
            continue
        for doc in docs:
            if doc.metadata.get("role") != "hr" or doc.metadata.get("source") != "hr_data.csv":
                continue
            for ref_id in re.findall(rf"(?m)^{ref_field}: ({_ID})\s*$", doc.page_content):
                row = bm25_retriever.find_row(collection, key_field, ref_id)
                if row is None or row.page_content in present:
                    continue
                present.add(row.page_content)
                resolved.append(row)
                if len(resolved) >= MAX_RESOLVED_ROWS:
                    return resolved
    return resolved
