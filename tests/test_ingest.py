import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest import (
    CONTEXT_HEADER_SEPARATOR,
    add_context_headers,
    build_context_header,
    dominant_section,
    load_documents_by_role,
    markdown_outline,
    save_bm25_docs,
    split_at_section_boundaries,
    title_from_filename,
)


def _quarterly_report(quarter: str, filler: int = 700) -> str:
    # Same section names in every quarter, like the real marketing reports.
    return (
        f"# Comprehensive Marketing Report - {quarter} 2024\n\n"
        f"## Executive Summary\n{'Summary text. ' * (filler // 14)}\n\n"
        f"## Campaign Highlights\n### Influencer Partnerships\n"
        f"{'Influencers drove sign-ups this quarter. ' * (filler // 42)}\n\n"
        f"## Conclusion\n{'Closing remarks. ' * (filler // 17)}\n"
    )


def _write_data(tmp_path):
    root = tmp_path / "data"
    (root / "marketing").mkdir(parents=True)
    (root / "hr").mkdir()
    (root / "marketing" / "marketing_report_q1_2024.md").write_text(_quarterly_report("Q1"), encoding="utf-8")
    (root / "marketing" / "marketing_report_q2_2024.md").write_text(_quarterly_report("Q2"), encoding="utf-8")
    (root / "hr" / "hr_data.csv").write_text(
        "employee_id,full_name,leave_balance\nFINEMP1005,Shaurya Joshi,4\nFINEMP1010,Vihaan Chowdhury,10\n",
        encoding="utf-8",
    )
    return root


# --- header building blocks -----------------------------------------------------


def test_title_from_filename():
    assert title_from_filename("hr_data.csv") == "HR Data"
    assert title_from_filename("marketing_report_q2_2024.md") == "Marketing Report Q2 2024"
    assert title_from_filename("employee_handbook.md") == "Employee Handbook"


def test_markdown_outline_reads_title_atx_and_setext_sections():
    text = (
        "# Financial Report - 2024\n\n"
        "Executive Summary:\n-------------------------------------------\nBody.\n\n"
        "## Q3 - July to September 2024\n### Quarterly Financial Overview\nBody.\n"
        "- list item\n---\n"
    )
    title, sections = markdown_outline(text)
    assert title == "Financial Report - 2024"
    # Setext and ATX H2 are sections; H3 and dash-underlined list items are not.
    assert [name for _, name in sections] == ["Executive Summary", "Q3 - July to September 2024"]


def test_dominant_section_uses_where_most_of_the_chunk_lies():
    sections = [(0, "Strategic Objectives"), (100, "Campaign Highlights"), (1000, "Conclusion")]
    # Starts with 20 chars of overlap from the previous section, then 500 of the next.
    assert dominant_section(sections, 80, 600) == "Campaign Highlights"
    assert dominant_section(sections, 0, 90) == "Strategic Objectives"
    assert dominant_section([], 0, 500) is None


def test_csv_row_header_uses_file_title_and_first_field():
    row = "employee_id: FINEMP1005\nfull_name: Shaurya Joshi"
    assert build_context_header(row, row, 0, "hr_data.csv", ".csv") == f"HR Data{CONTEXT_HEADER_SEPARATOR}employee_id: FINEMP1005"


def test_add_context_headers_prefixes_text_and_keeps_metadata():
    source = Document(page_content="# Handbook\n\n## Leave Policies\nSick leave is 12 days.", metadata={})
    chunk = Document(
        page_content="## Leave Policies\nSick leave is 12 days.",
        metadata={"source": "employee_handbook.md", "role": "employee", "start_index": 12},
    )
    [out] = add_context_headers(source, [chunk], "employee_handbook.md", ".md")

    assert out.page_content == "Handbook › Leave Policies\n\n## Leave Policies\nSick leave is 12 days."
    assert out.metadata == {
        "source": "employee_handbook.md",
        "role": "employee",
        "start_index": 12,
        "context_header": "Handbook › Leave Policies",
    }


# --- full ingestion path --------------------------------------------------------


def test_repeated_sections_get_distinct_quarter_headers(tmp_path):
    docs = load_documents_by_role(_write_data(tmp_path))

    highlights = {
        d.metadata["source"]: d
        for d in docs["marketing"]
        if d.metadata["context_header"].endswith("Campaign Highlights")
    }
    assert set(highlights) == {"marketing_report_q1_2024.md", "marketing_report_q2_2024.md"}
    q2 = highlights["marketing_report_q2_2024.md"]
    assert q2.page_content.startswith("Comprehensive Marketing Report - Q2 2024 › Campaign Highlights\n\n")
    assert q2.metadata["role"] == "marketing"
    # Every chunk carries a header, and the header is the first thing embedded.
    for d in docs["marketing"]:
        assert d.page_content.startswith(d.metadata["context_header"] + "\n\n")


def test_csv_rows_keep_loader_metadata_and_get_row_headers(tmp_path):
    docs = load_documents_by_role(_write_data(tmp_path))["hr_docs"]

    assert [d.metadata["context_header"] for d in docs] == [
        "HR Data › employee_id: FINEMP1005",
        "HR Data › employee_id: FINEMP1010",
    ]
    assert docs[0].metadata["row"] == 0  # CSVLoader metadata preserved
    assert docs[0].metadata["source"] == "hr_data.csv" and docs[0].metadata["role"] == "hr"
    assert "full_name: Shaurya Joshi" in docs[0].page_content


def test_bm25_json_matches_chroma_text_and_keeps_only_source_and_role(tmp_path):
    docs_by_collection = load_documents_by_role(_write_data(tmp_path))
    save_bm25_docs(docs_by_collection, str(tmp_path / "db"))

    for collection, docs in docs_by_collection.items():
        if not docs:
            continue
        records = json.loads((tmp_path / "db" / "bm25_docs" / f"{collection}.json").read_text(encoding="utf-8"))
        # BM25 indexes exactly the text Chroma embeds.
        assert [r["page_content"] for r in records] == [d.page_content for d in docs]
        assert all(set(r["metadata"]) == {"source", "role"} for r in records)


# --- chunks never straddle two sections ---------------------------------------


def _handbook_like() -> str:
    # Two sections that together fit in one chunk: the splitter packs them
    # together, as it packed the end of "Compensation & Payroll" and all of
    # "Reimbursement Policies" in the real handbook.
    return (
        "# Handbook\n\n"
        f"## Payroll\n{'Salary is paid monthly. ' * 20}\n\n"
        "## Reimbursement\n- Hotel stay up to Rs 3,000/night.\n- Meals up to Rs 500/day.\n"
        f"{'Submit claims within 30 days. ' * 8}\n"
    )


def test_chunk_spanning_two_sections_is_cut_at_the_section_start():
    text = _handbook_like()
    doc = Document(page_content=text, metadata={"source": "handbook.md"})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    # Premise: without the cut, one chunk mixes both sections.
    assert any("Salary is paid" in c.page_content and "Hotel stay" in c.page_content for c in splitter.split_documents([doc]))

    chunks = add_context_headers(doc, split_at_section_boundaries(doc, splitter, ".md"), "handbook.md", ".md")

    hotel = [c for c in chunks if "Hotel stay" in c.page_content]
    assert len(hotel) == 1
    assert hotel[0].metadata["context_header"] == f"Handbook{CONTEXT_HEADER_SEPARATOR}Reimbursement"
    assert "Salary is paid" not in hotel[0].page_content
    for c in chunks:
        start = c.metadata["start_index"]
        assert text[start : start + 20] in text  # start_index stays relative to the document
        if "Salary is paid" in c.page_content:
            assert c.metadata["context_header"].endswith("Payroll")
            assert "Hotel stay" not in c.page_content


def test_split_keeps_non_markdown_and_single_section_chunks_unchanged():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    row = Document(page_content="employee_id: FINEMP1005\nfull_name: Shaurya Joshi", metadata={})
    assert [c.page_content for c in split_at_section_boundaries(row, splitter, ".csv")] == [row.page_content]

    doc = Document(page_content="## A\n" + "word " * 100 + "\n\n## B\nshort tail\n", metadata={})
    plain = splitter.split_documents([doc])
    # "## B" starts within min_part of the end, so the chunk is left as the splitter made it.
    assert [c.page_content for c in split_at_section_boundaries(doc, splitter, ".md")] == [c.page_content for c in plain]
