# FinSolve Technologies — Enterprise RAG Chatbot with Role-Based Access Control

An internal assistant that answers employees' questions from company documents — finance reports, HR records, marketing reports, engineering docs and the employee handbook — while guaranteeing each person only sees data their role is entitled to. Access control is enforced on the server before any document is retrieved, and answers are grounded only in authorized, relevance-checked content; when nothing relevant exists, the assistant says so instead of guessing.

**Highlights**
- **RBAC at the retrieval layer** — six roles, five department collections, zero cross-department leakage in every evaluated configuration.
- **Accurate retrieval** — hybrid dense + BM25 search, Reciprocal Rank Fusion, cross-encoder reranking with a relevance threshold, and contextual chunk headers. Production pipeline: **Recall@5 0.917, MRR 0.875**.
- **Conversational** — per-user session memory and automatic rewriting of follow-up questions.
- **Fast repeats** — RBAC-namespaced semantic cache on Redis Stack (~0.2 s for a repeated question vs ~2 s uncached).
- **Verified** — 90 automated tests, a retrieval evaluation harness, and a live end-to-end smoke test across all six users.

**Stack:** FastAPI · Streamlit · LangChain · ChromaDB · rank-bm25 · sentence-transformers · Redis Stack + redisvl · Groq (`openai/gpt-oss-120b`) · JWT (python-jose) · bcrypt

## Architecture

```
 Streamlit UI ──► login ──► JWT (username, role, expiry)
      │
      ▼
 POST /chat  (question, session_id, Bearer JWT)
      │
      ├─ Authenticate ─────────── invalid / expired token → 401
      ├─ Authorize ────────────── role → allowed document collections
      ├─ Conversation memory ──── last 6 messages of this user's session
      ├─ Query rewriting ──────── follow-ups turned into standalone questions
      ├─ Semantic cache ───────── per role + collections ──── hit → answer
      ├─ Retrieval
      │    ├─ HR, Finance ─────── dense + BM25 → Reciprocal Rank Fusion
      │    └─ other departments ─ dense
      ├─ Cross-encoder reranker ─ best 6 chunks
      ├─ Relevance threshold ──── nothing relevant → safe answer, no LLM call
      └─ LLM (Groq) ───────────── answers only from the authorized chunks
      │
      ▼
 answer · sources · role · session_id · cache_hit   (+ X-Request-ID, structured request log)
```

## What's Built

| Component | Purpose |
|-----------|---------|
| **Server-side RBAC** | The role comes from the signed JWT and selects the collections to search *before* retrieval; unauthorized data never reaches the LLM, so it cannot leak through the prompt. |
| **Hybrid search (HR, Finance)** | BM25 matches exact identifiers — employee IDs, names, figures — that embeddings blur. |
| **Reciprocal Rank Fusion** | Merges dense and BM25 rankings without comparing their incompatible scores. |
| **Cross-encoder reranker** | Rescores candidates by reading question and chunk together; keeps the best 6. |
| **Relevance threshold** | Discards weak chunks; if none remain, returns a safe "unable to find" answer without calling the LLM. |
| **Contextual chunk headers** | Each chunk is indexed with its document and section (e.g. `Comprehensive Marketing Report - Q2 2024 › Campaign Highlights`), so identical sections in different reports stay distinguishable. |
| **Semantic cache** | Redis Stack vector cache, isolated per role and collection set, with TTL and version-based invalidation; the app runs uncached if Redis is unavailable. |
| **Conversation memory** | Last 6 messages per user and session, isolated between users. |
| **Query rewriting** | Rewrites context-dependent follow-ups ("How does that compare…?") only when needed. |
| **Observability** | Request ID per call and one structured log line per request, with no tokens, passwords or document text. |

**Retrieval policy by department**

| Department | Strategy | Why |
|------------|----------|-----|
| finance | dense + BM25 | Exact figures, quarters and vendor names |
| hr_docs | dense + BM25 | Employee IDs and names need exact-token matching |
| marketing | dense | Narrative reports; BM25 did not help in evaluation |
| engineering | dense | Conceptual technical prose |
| general | dense | Policy questions phrased in varied language |

Implementation details for every component are in **[docs/DESIGN.md](docs/DESIGN.md)**.

## Getting Started

Requires Python 3.10+.

**1. Clone and create a virtual environment**
```bash
git clone https://github.com/vgpatel124/Role-based-Chatbot.git
cd Role-based-Chatbot
python -m venv venv && source venv/bin/activate
```

**2. Configure**
```bash
cp .env.example .env
```
Set `GROQ_API_KEY` (from [console.groq.com/keys](https://console.groq.com/keys)) and change `JWT_SECRET_KEY`.

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start Redis Stack** (optional — without it the app runs uncached)

macOS:
```bash
brew tap redis-stack/redis-stack
brew trust --cask redis-stack/redis-stack/redis-stack-server
brew install --cask redis-stack-server
redis-stack-server
```
Docker:
```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```
Redis Stack is required rather than plain Redis, because the cache uses its vector search module.

**5. Index the documents**
```bash
python app/ingest.py
```
To re-index, delete `chroma_db/` first.

**6. Start the API**
```bash
uvicorn app.main:app --reload --port 8000
```
The first start downloads two small models (~90 MB each).

**7. Start the UI**
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501 and log in as one of the test users.

### Configuration

All settings live in `.env` (see [.env.example](.env.example)). The ones you are most likely to change:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(none)* | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Chat model for answers and query rewriting |
| `JWT_SECRET_KEY` | *(placeholder)* | Secret used to sign login tokens — change it |
| `REDIS_URL` | `redis://localhost:6379` | Redis Stack connection |
| `CACHE_ENABLED` | `true` | Turn the semantic cache on or off |
| `RERANKER_SCORE_THRESHOLD` | `0.1` | Minimum relevance (0–1) for a chunk to be used |

The full reference (retrieval depths, fusion, reranker, cache and conversation settings) is in [docs/DESIGN.md](docs/DESIGN.md#configuration-reference).

### Test users

| Username | Password    | Role        | Can access |
|----------|-------------|-------------|------------|
| alice    | password123 | finance     | Finance reports |
| bob      | password123 | marketing   | Marketing reports |
| carol    | password123 | hr          | HR employee records |
| dave     | password123 | engineering | Engineering documentation |
| eve      | password123 | c_level     | All departments |
| frank    | password123 | employee    | Employee handbook |

Users are defined in `app/auth.py` for demonstration.

## Evidence

### Live end-to-end results

Run against the running system with real authentication, retrieval, Redis and Groq:

| User | Question | Result |
|------|----------|--------|
| alice (finance) | What was the gross margin in Q3 2024? | "62%" — `quarterly_financial_report.md` |
| alice (finance) | *same question again* | Served from cache in ~0.2 s |
| alice (finance) | What is the leave balance of FINEMP1005? | Blocked — safe answer, no sources, no LLM call |
| carol (hr) | What is the leave balance of FINEMP1005? | "4 days" — `hr_data.csv` |
| carol (hr) | What was the gross margin in Q3 2024? | Blocked |
| bob (marketing) | Which marketing campaign performed best in Q2 2024? | Southeast Asia influencer partnerships, ~40,000 new sign-ups — `marketing_report_q2_2024.md` |
| dave (engineering) | What are the stages of the SDLC at FinSolve? | Scrum ceremonies — `engineering_master_doc.md` |
| frank (employee) | What is the company leave policy? | Leave types and entitlements — `employee_handbook.md` |
| frank (employee) | What was the gross margin in Q3 2024? | Blocked |
| eve (c_level) | What was the total revenue in 2024? | "$9.4 billion" |
| eve (c_level) | How does that compare to the marketing spend? | Follow-up rewritten → "$9.4B revenue vs $2.3B marketing spend" |
| *(invalid token)* | — | 401 |

All 13 scenarios passed. The same question asked by four other roles never returned alice's cached answer.

### Automated tests

```bash
pytest tests/ -v
```

**90 tests, all passing, in ~3 seconds**, with no network, Redis or API key required. They cover RBAC and authentication, cache isolation and failure handling, conversation memory and rewriting, retrieval and fusion, reranking, ingestion, and the full login-to-answer flow. Breaking RBAC or cache isolation on purpose makes the suite fail (9 and 6 failing tests respectively). Per-file coverage: [docs/DESIGN.md](docs/DESIGN.md#test-coverage).

### Retrieval evaluation

```bash
python evaluation/evaluate_retrieval.py
```

Five retrieval configurations on 12 answerable questions across all departments, plus 2 unauthorized-access questions (measured, K = 5):

| Configuration | Recall@5 | MRR |
|---------------|----------|-----|
| A · Dense only (original) | 0.583 | 0.364 |
| B · Dense + reranker | 0.667 | 0.625 |
| C · Hybrid for HR/Finance | 0.833 | 0.517 |
| D · Hybrid for every department | 0.750 | 0.507 |
| **E · Full pipeline (production)** | **0.917** | **0.875** |

- Production retrieval finds a correct chunk in the top 5 for **11 of 12** questions, up from 9 of 12 before contextual chunk headers (Recall 0.750 → 0.917).
- **0 chunks from an unauthorized department** in every configuration.
- The evaluation set is small (12 questions), so treat the numbers as directional. Methodology, all metrics and before/after analysis: [docs/DESIGN.md](docs/DESIGN.md#evaluation-methodology-and-analysis).

## Limitations

- Users are stored in memory in `app/auth.py`; there is no user database.
- Conversation history is in process memory: lost on restart and not shared across API workers.
- Documents are indexed only when `python app/ingest.py` is run.
- Questions comparing two reports (e.g. Q2 vs Q3 marketing spend) are not yet reliably retrieved.
- The API is synchronous.

Details: [docs/DESIGN.md](docs/DESIGN.md#detailed-limitations).

## Roadmap

- PostgreSQL user store with user management.
- Redis-backed session store so conversations survive restarts and scale across workers.
- Async endpoints and streaming responses.
- Automatic re-indexing when source documents change.
- Per-document metadata filters for comparison questions; exact-identifier boosting for HR/finance lookups.
- A larger evaluation set with answer-level metrics.

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/login` | – | OAuth2 form login → JWT |
| POST | `/auth/login-json` | – | JSON login → JWT (used by the UI) |
| GET | `/auth/me` | Bearer | Current username and role |
| POST | `/chat` | Bearer | `{question, session_id?}` → `{answer, sources, role, session_id, cache_hit}` |
| GET | `/health` | – | Liveness check |
