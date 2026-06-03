import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="FinSolve Internal Assistant", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
    .main-header {color: #0B5ED7; font-weight: 700; font-size: 2rem;}
    .role-badge {
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 0.5rem;
        background-color: #DCEBFF; color: #0B5ED7; font-weight: 600; margin-left: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "token" not in st.session_state:
        st.session_state.token = None
    if "user" not in st.session_state:
        st.session_state.user = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def login(username: str, password: str) -> bool:
    response = requests.post(
        f"{API_BASE_URL}/auth/login-json",
        json={"username": username, "password": password},
        timeout=30,
    )
    if response.status_code != 200:
        return False

    token = response.json().get("access_token")
    me_resp = requests.get(
        f"{API_BASE_URL}/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if me_resp.status_code != 200:
        return False

    st.session_state.token = token
    st.session_state.user = me_resp.json()
    st.session_state.messages = []
    return True


def ask_question(question: str) -> dict:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": question},
            headers=headers,
            timeout=60,
        )
    except requests.exceptions.RequestException:
        st.error("Could not reach the server. Is the backend running?")
        return {"answer": "Request failed. Please try again.", "sources": []}

    if response.status_code != 200:
        return {"answer": "Request failed. Please try again.", "sources": []}
    return response.json()


def render_login() -> None:
    st.markdown('<div class="main-header">FinSolve Technologies</div>', unsafe_allow_html=True)
    st.subheader("Internal RBAC Chatbot Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if login(username.strip(), password.strip()):
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials or backend unavailable.")


def render_chat() -> None:
    user = st.session_state.user
    st.markdown('<div class="main-header">FinSolve Technologies</div>', unsafe_allow_html=True)
    st.markdown(
        f"Logged in as **{user['username']}** <span class='role-badge'>{user['role']}</span>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for source in msg["sources"]:
                        st.markdown(f"- `{source}`")

    prompt = st.chat_input("Ask a question about your allowed data...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = ask_question(prompt)
        answer = result.get("answer", "I don't have access to that information")
        sources = result.get("sources", [])
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Sources"):
                if sources:
                    for source in sources:
                        st.markdown(f"- `{source}`")
                else:
                    st.markdown("No source documents returned.")


init_state()
if st.session_state.token and st.session_state.user:
    render_chat()
else:
    render_login()
