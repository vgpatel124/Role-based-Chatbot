import pytest

from app import rag
from app.rag import get_collections_for_role

ALL_COLLECTIONS = ["finance", "marketing", "hr_docs", "engineering", "general"]


@pytest.mark.parametrize(
    "role, expected",
    [
        ("finance", ["finance"]),
        ("hr", ["hr_docs"]),
        ("marketing", ["marketing"]),
        ("engineering", ["engineering"]),
        ("employee", ["general"]),
    ],
)
def test_single_department_roles_get_only_their_collection(role, expected):
    assert get_collections_for_role(role) == expected


def test_c_level_gets_all_five_collections():
    collections = get_collections_for_role("c_level")
    assert sorted(collections) == sorted(ALL_COLLECTIONS)
    assert len(collections) == 5


def test_unknown_role_falls_back_to_general_only():
    assert get_collections_for_role("intern") == ["general"]


def test_finance_user_asking_hr_question_gets_no_hr_data(client, auth_header, mock_chromadb, mock_llm):
    response = client.post(
        "/chat",
        json={"question": "What is the leave balance of employee FINEMP1010?"},
        headers=auth_header("finance"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "finance"
    assert body["sources"] == []
    assert "don't have access" in body["answer"] or "unable to find" in body["answer"]
    assert "FINEMP1010" not in body["answer"]
    # Authorization happened before retrieval: only the finance collection was searched.
    assert mock_chromadb.touched() == {"finance"}
    # Nothing relevant passed the threshold, so the LLM was never called.
    mock_llm.invoke.assert_not_called()


def test_c_level_chat_searches_all_collections(client, auth_header, mock_chromadb, mock_llm):
    response = client.post("/chat", json={"question": "Kubernetes microservices"}, headers=auth_header("c_level"))
    assert response.status_code == 200
    assert mock_chromadb.touched() == set(ALL_COLLECTIONS)


def test_chat_with_invalid_jwt_returns_401(client, mock_chromadb, mock_llm):
    response = client.post(
        "/chat", json={"question": "Q3 revenue?"}, headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"
    assert mock_chromadb.dense_calls == []


def test_chat_without_token_returns_401(client, mock_chromadb, mock_llm):
    response = client.post("/chat", json={"question": "Q3 revenue?"})
    assert response.status_code == 401
    assert mock_chromadb.dense_calls == []


def test_chat_with_expired_jwt_returns_401_token_has_expired(client, expired_jwt, mock_chromadb, mock_llm):
    response = client.post(
        "/chat", json={"question": "Q3 revenue?"}, headers={"Authorization": f"Bearer {expired_jwt}"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Token has expired"
    assert mock_chromadb.dense_calls == []


def test_role_comes_from_server_not_from_request(client, auth_header, mock_chromadb, mock_llm):
    # A client cannot widen access by sending a role in the body.
    response = client.post(
        "/chat",
        json={"question": "leave balance FINEMP1010", "role": "c_level"},
        headers=auth_header("finance"),
    )
    assert response.status_code == 200
    assert mock_chromadb.touched() == {"finance"}


def test_retrieval_is_scoped_before_llm_prompt(client, auth_header, mock_chromadb, mock_llm):
    response = client.post(
        "/chat", json={"question": "What was the gross margin in 2024?"}, headers=auth_header("finance")
    )
    assert response.status_code == 200
    prompt = mock_llm.invoke.call_args[0][0][1].content
    assert "62%" in prompt  # finance context reached the LLM
    for other in ("FINEMP", "Kubernetes", "influencer", "sick leave"):
        assert other not in prompt
    assert rag.get_collections_for_role("finance") == ["finance"]
