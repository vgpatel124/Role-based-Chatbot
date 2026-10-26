"""The answer-evaluation dataset must be checkable against the documents it is about.

A question whose reference or keywords are not in the source file cannot be answered correctly
by any model, and it would silently drag the scores down. These tests keep every question honest:
keywords and the numbers in the reference must appear verbatim in the expected source files, new
questions carry a verbatim `evidence` excerpt, and a role is only asked about files it may read.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from evaluation.redteam_cases import COLLECTION_FILES, EXPECTED_ACCESS

ROOT = Path(__file__).resolve().parent.parent
DATASET = json.loads((ROOT / "evaluation" / "answer_dataset.json").read_text(encoding="utf-8"))
SOURCES = {
    p.name: re.sub(r"\s+", " ", p.read_text(encoding="utf-8").replace("\u00a0", " ")).lower()
    for p in (ROOT / "resources" / "data").rglob("*")
    if p.suffix in (".md", ".csv")
}
ANSWERABLE = [r for r in DATASET if r["type"] == "answerable"]
OUT_OF_SCOPE = [r for r in DATASET if r["type"] == "out_of_scope"]

# Questions written before the `evidence` field existed. Everything added later must carry evidence.
LEGACY_IDS = frozenset(
    {
        "clevel_handbook",
        "clevel_hr",
        "clevel_instantpay",
        "clevel_revenue",
        "eng_db_scaling",
        "eng_docker",
        "eng_microservices",
        "eng_pipeline_tools",
        "eng_prod_deploy",
        "eng_sprint",
        "fin_annual_revenue",
        "fin_dso",
        "fin_gross_margin_yoy",
        "fin_q1_revenue",
        "fin_q2_vendor_services",
        "fin_q3_net_income",
        "fin_q3_revenue",
        "fin_q4_gross_margin",
        "fin_software_subscriptions",
        "fin_vendor_drivers",
        "gen_claims",
        "gen_hotel_meals",
        "gen_maternity",
        "gen_overtime",
        "gen_referral",
        "gen_sick_leave",
        "gen_tuition",
        "gen_wfh",
        "hr_1005_attendance",
        "hr_1010_manager",
        "hr_1020_leave",
        "hr_1020_role_location",
        "hr_1033_role",
        "mkt_q1_conferences",
        "mkt_q1_cpa",
        "mkt_q1_instantpay",
        "mkt_q2_influencers",
        "mkt_q3_loyalty",
        "mkt_q3_merchants",
        "mkt_q3_targets",
        "mkt_q4_budget",
        "oos_clevel_weather",
        "oos_employee_ceo_home",
        "oos_eng_cafeteria",
        "oos_finance_stock",
        "oos_hr_missing_employee",
        "oos_marketing_general",
    }
)
MIN_ANSWERABLE = 100


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).lower()


def _ids(rows):
    return [r["id"] for r in rows]


def test_ids_and_questions_are_unique():
    assert len(set(_ids(DATASET))) == len(DATASET)
    questions = Counter((r["role"], _norm(r["question"])) for r in DATASET)
    assert not [q for q, n in questions.items() if n > 1]


def test_the_dataset_is_large_enough_to_be_statistically_useful():
    assert len(ANSWERABLE) >= MIN_ANSWERABLE
    roles = Counter(r["role"] for r in ANSWERABLE)
    assert set(roles) == set(EXPECTED_ACCESS)
    assert min(roles.values()) >= 8  # every role is measured, not just the big ones


def test_the_legacy_ids_still_exist():
    assert LEGACY_IDS <= set(_ids(DATASET))


@pytest.mark.parametrize("row", ANSWERABLE, ids=_ids(ANSWERABLE))
def test_answerable_question_is_complete_and_inside_the_roles_documents(row):
    assert row["role"] in EXPECTED_ACCESS
    assert row["question"].strip() and row["reference"].strip()
    assert row["keywords"] and row["expected_sources"]
    allowed_files = set().union(*(COLLECTION_FILES[c] for c in EXPECTED_ACCESS[row["role"]]))
    for source in row["expected_sources"]:
        assert source in SOURCES, f"{source} is not a document under resources/data"
        assert source in allowed_files, f"role {row['role']} may not read {source}"


@pytest.mark.parametrize("row", ANSWERABLE, ids=_ids(ANSWERABLE))
def test_every_keyword_appears_verbatim_in_an_expected_source(row):
    text = " ".join(SOURCES[s] for s in row["expected_sources"])
    for keyword in row["keywords"]:
        # "a|b" means either spelling is accepted, as in the scorer; one of them must be in the document.
        assert any(_norm(alt) in text for alt in keyword.split("|")), f"{keyword!r} is not in {row['expected_sources']}"


@pytest.mark.parametrize("row", ANSWERABLE, ids=_ids(ANSWERABLE))
def test_every_number_in_the_reference_appears_in_the_documents(row):
    mentioned = [name for name in SOURCES if name in row["reference"]]
    text = " ".join(SOURCES[s] for s in [*row["expected_sources"], *mentioned])
    for number in re.findall(r"\d[\d,]*(?:\.\d+)?", row["reference"]):
        number = number.rstrip(",.")
        assert number in text, f"{number!r} in the reference is not in the source documents"


@pytest.mark.parametrize("row", ANSWERABLE, ids=_ids(ANSWERABLE))
def test_evidence_is_quoted_verbatim_from_the_source(row):
    evidence = row.get("evidence")
    if row["id"] in LEGACY_IDS and evidence is None:
        pytest.skip("legacy question without an evidence excerpt")
    assert evidence, f"{row['id']} needs an `evidence` list of verbatim excerpts"
    text = " ".join(SOURCES[s] for s in row["expected_sources"])
    for excerpt in evidence:
        assert _norm(excerpt) in text, f"not verbatim in {row['expected_sources']}: {excerpt!r}"


@pytest.mark.parametrize("row", OUT_OF_SCOPE, ids=_ids(OUT_OF_SCOPE))
def test_out_of_scope_questions_have_no_expected_answer(row):
    assert not row["expected_sources"] and not row["keywords"] and not row["reference"]
    assert row["role"] in EXPECTED_ACCESS


def test_the_test_catches_a_keyword_missing_from_its_source():
    row = {"keywords": ["$999 trillion"], "expected_sources": ["quarterly_financial_report.md"]}
    text = SOURCES[row["expected_sources"][0]]
    assert not any(_norm(alt) in text for alt in row["keywords"][0].split("|"))
