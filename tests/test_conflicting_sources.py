"""Two reports can disagree about the same figure (here: FY2024 vendor services
is $30M in financial_summary.md and about $510M in quarterly_financial_report.md).
The pipeline must put both sources in front of the LLM and instruct it to report
the disagreement instead of silently choosing one figure."""

import pytest
from langchain_core.documents import Document

from app import rag

QUESTION = "What were the total vendor services expenses in 2024 and what drove them?"

SUMMARY_CHUNK = (
    "Financial Report 2024 › Expense Breakdown by Category\n\n"
    "Vendor Services - A total of $30M, an 18% increase. Drivers of vendor services expenses: "
    "marketing-related expenses 40%, training and education 30%, software subscriptions 25%."
)
QUARTERLY_CHUNK = (
    "Quarterly Financial Report 2024 › 2024 Annual Summary\n\n"
    "Total vendor services expenses and vendor costs for 2024 rose to $510 million."
)


@pytest.fixture
def conflicting_finance_corpus(monkeypatch):
    """Finance holds exactly the two disagreeing chunks; every other collection is empty."""
    chunks = {
        "finance": [
            Document(page_content=SUMMARY_CHUNK, metadata={"source": "financial_summary.md", "role": "finance"}),
            Document(
                page_content=QUARTERLY_CHUNK,
                metadata={"source": "quarterly_financial_report.md", "role": "finance"},
            ),
        ]
    }

    class Store:
        def __init__(self, name):
            self.name = name

        def similarity_search(self, query, k=4):
            return [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in chunks.get(self.name, [])]

    monkeypatch.setattr(rag, "_get_vectorstore", lambda name, embeddings: Store(name))
    monkeypatch.setattr(
        rag.bm25_retriever,
        "retrieve",
        lambda question, collection, top_k: Store(collection).similarity_search(question, top_k),
    )


def test_system_prompt_requires_reporting_conflicting_figures():
    prompt = rag.SYSTEM_PROMPT.lower()

    assert "different values for the same figure" in prompt
    assert "do not choose one silently" in prompt
    assert "state that the documents differ" in prompt


def test_both_conflicting_sources_reach_the_llm_and_are_cited(conflicting_finance_corpus, mock_llm):
    mock_llm.invoke.return_value.content = "financial_summary.md says $30M; the quarterly report says $510 million."

    result = rag.query_rag(QUESTION, "finance")

    system_message, human_message = mock_llm.invoke.call_args.args[0]
    assert system_message.content == rag.SYSTEM_PROMPT
    # The annual-summary evidence for the expected figure, its growth rate and its drivers:
    assert "$30M" in human_message.content
    assert "18% increase" in human_message.content
    for driver in ("marketing-related expenses 40%", "training and education 30%", "software subscriptions 25%"):
        assert driver in human_message.content
    # ... and the conflicting figure, each labelled with its own document.
    assert "Source: financial_summary.md" in human_message.content
    assert "Source: quarterly_financial_report.md" in human_message.content
    assert "$510 million" in human_message.content
    assert result["sources"] == ["financial_summary.md", "quarterly_financial_report.md"]


def test_conflicting_sources_stay_inside_the_authorized_department(conflicting_finance_corpus, mock_llm):
    """Surfacing both figures must not widen access: HR still gets neither."""
    rag.query_rag(QUESTION, "hr")

    for call in mock_llm.invoke.call_args_list:
        assert "$30M" not in call.args[0][1].content
        assert "$510" not in call.args[0][1].content
