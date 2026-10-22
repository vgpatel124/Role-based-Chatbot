"""Red-team tests for RBAC, offline.

Every role asks every other department's question, wrapped in each known
manipulation pattern (evaluation/redteam_cases.py), through the real FastAPI
app with real JWT auth. The only fakes are the LLM and the vector stores.
Pass criteria: only authorized collections are searched, and no text from an
unauthorized collection reaches the LLM prompt.

The second half attacks the token itself: forged, tampered, unsigned and
mis-signed JWTs, and role claims smuggled in through other request channels.
"""

import base64
import json

import pytest
from jose import jwt

from app.config import settings
from app.rag import get_collections_for_role
from evaluation.redteam_cases import ALL_COLLECTIONS, EXPECTED_ACCESS, build_cases
from tests.conftest import CORPUS, USER_FOR_ROLE

CASES = build_cases()

ANY_QUESTION = {"question": "What was the gross margin in Q3 2024?"}


def _b64(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _chat(client, headers=None, **kwargs):
    return client.post("/chat", json=kwargs.pop("json", ANY_QUESTION), headers=headers or {}, **kwargs)


# --- The role x department x manipulation matrix -------------------------------


@pytest.mark.parametrize("role", sorted(EXPECTED_ACCESS))
def test_role_policy_matches_the_specification(role):
    assert set(get_collections_for_role(role)) == EXPECTED_ACCESS[role]


def test_matrix_covers_every_unauthorized_pair():
    # 5 single-department roles x 4 foreign collections x 8 templates; c_level has none.
    assert len(CASES) == 5 * 4 * 8
    assert {c.role for c in CASES} == {"finance", "marketing", "hr", "engineering", "employee"}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_attack_never_reaches_unauthorized_data(case, client, auth_header, mock_chromadb, mock_llm):
    authorized = EXPECTED_ACCESS[case.role]

    response = _chat(client, auth_header(case.role), json={"question": case.question})

    assert response.status_code == 200
    assert response.json()["role"] == case.role
    # Retrieval never touched a collection outside the role's scope.
    assert mock_chromadb.touched() <= authorized, case.id
    # Nothing from an unauthorized collection was placed in the prompt.
    forbidden = [doc.page_content for name in ALL_COLLECTIONS if name not in authorized for doc in CORPUS[name]]
    for call in mock_llm.invoke.call_args_list:
        prompt = " ".join(message.content for message in call.args[0])
        assert not any(text in prompt for text in forbidden), case.id


# --- Token attacks -----------------------------------------------------------------


def test_valid_signature_with_forged_role_claim_is_ignored(client, mock_chromadb, mock_llm):
    """Even a token signed with the real secret cannot raise privileges: the role
    is looked up server-side from the username, not read from the claim."""
    from app.auth import create_access_token

    token = create_access_token({"sub": "alice", "role": "c_level"})
    response = _chat(client, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["role"] == "finance"
    assert mock_chromadb.touched() == {"finance"}


@pytest.mark.parametrize("role_claim", ["admin", "superuser", "", "C_LEVEL", "finance,hr"])
def test_unknown_role_claim_cannot_widen_access(role_claim, client, mock_chromadb, mock_llm):
    from app.auth import create_access_token

    token = create_access_token({"sub": "frank", "role": role_claim})
    response = _chat(client, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert mock_chromadb.touched() == {"general"}


def test_payload_swapped_after_signing_is_rejected(client, mock_chromadb, mock_llm, valid_jwt_factory):
    header, _, signature = valid_jwt_factory("finance").split(".")
    forged_payload = _b64({"sub": "alice", "role": "c_level", "exp": 9999999999})

    response = _chat(client, {"Authorization": f"Bearer {header}.{forged_payload}.{signature}"})

    assert response.status_code == 401
    assert mock_chromadb.touched() == set()


def test_unsigned_alg_none_token_is_rejected(client, mock_chromadb, mock_llm):
    token = f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64({'sub': 'eve', 'role': 'c_level', 'exp': 9999999999})}."

    response = _chat(client, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert mock_chromadb.touched() == set()


@pytest.mark.parametrize(
    "secret",
    ["wrong-secret", "", "secret", "finsolve-secret-key-change-in-production"],
    ids=["wrong", "empty", "common", "documented-default"],
)
def test_token_signed_with_another_secret_is_rejected(secret, client, mock_chromadb, mock_llm):
    assert secret != settings.JWT_SECRET_KEY
    token = jwt.encode({"sub": "eve", "role": "c_level", "exp": 9999999999}, secret, algorithm="HS256")

    response = _chat(client, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert mock_chromadb.touched() == set()


def test_token_with_a_different_algorithm_is_rejected(client, mock_chromadb, mock_llm):
    token = jwt.encode(
        {"sub": "eve", "role": "c_level", "exp": 9999999999}, settings.JWT_SECRET_KEY, algorithm="HS512"
    )

    response = _chat(client, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 401


def test_token_for_unknown_user_is_rejected(client, mock_chromadb, mock_llm):
    token = jwt.encode(
        {"sub": "mallory", "role": "c_level", "exp": 9999999999}, settings.JWT_SECRET_KEY, algorithm="HS256"
    )

    response = _chat(client, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert mock_chromadb.touched() == set()


def test_token_without_subject_is_rejected(client, mock_chromadb, mock_llm):
    token = jwt.encode({"role": "c_level", "exp": 9999999999}, settings.JWT_SECRET_KEY, algorithm="HS256")

    response = _chat(client, {"Authorization": f"Bearer {token}"})

    assert response.status_code == 401


@pytest.mark.parametrize("header", ["Basic YWxpY2U6cGFzc3dvcmQxMjM=", "Bearer", "Bearer ", "token abc", ""])
def test_malformed_authorization_headers_are_rejected(header, client, mock_chromadb, mock_llm):
    response = _chat(client, {"Authorization": header} if header else {})

    assert response.status_code == 401
    assert mock_chromadb.touched() == set()


# --- Role smuggled through other channels -------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"headers": {"X-User-Role": "c_level"}},
        {"headers": {"X-Role": "c_level", "X-Forwarded-User": "eve"}},
        {"params": {"role": "c_level", "user": "eve"}},
    ],
    ids=["x-user-role", "x-role", "query-string"],
)
def test_role_in_request_metadata_is_ignored(extra, client, auth_header, mock_chromadb, mock_llm):
    headers = {**auth_header("employee"), **extra.pop("headers", {})}

    response = client.post("/chat", json=ANY_QUESTION, headers=headers, **extra)

    assert response.status_code == 200
    assert response.json()["role"] == "employee"
    assert mock_chromadb.touched() == {"general"}


def test_role_in_cookie_is_ignored(client, auth_header, mock_chromadb, mock_llm):
    client.cookies.set("role", "c_level")

    response = client.post("/chat", json=ANY_QUESTION, headers=auth_header("employee"))

    assert response.status_code == 200
    assert mock_chromadb.touched() == {"general"}


@pytest.mark.parametrize("field", ["role", "user_role", "collections", "username", "authorized_collections"])
def test_role_in_request_body_is_ignored(field, client, auth_header, mock_chromadb, mock_llm):
    value = ["finance", "hr_docs"] if field.endswith("collections") else "c_level"

    response = client.post("/chat", json={**ANY_QUESTION, field: value}, headers=auth_header("employee"))

    assert response.status_code in (200, 422)
    if response.status_code == 200:
        assert mock_chromadb.touched() == {"general"}


def test_every_role_can_log_in_and_stays_in_its_lane(client, mock_chromadb, mock_llm):
    """Logging in as a user yields exactly that user's collections, nothing more."""
    for role, username in USER_FOR_ROLE.items():
        mock_chromadb.dense_calls.clear()
        mock_chromadb.bm25_calls.clear()
        login = client.post("/auth/login-json", json={"username": username, "password": "password123"})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        _chat(client, headers)

        assert mock_chromadb.touched() == EXPECTED_ACCESS[role], role


def test_login_does_not_reveal_whether_the_username_exists(client):
    unknown = client.post("/auth/login-json", json={"username": "nobody", "password": "x"})
    wrong_password = client.post("/auth/login-json", json={"username": "alice", "password": "x"})

    assert unknown.status_code == wrong_password.status_code == 401
    assert unknown.json()["detail"] == wrong_password.json()["detail"]
