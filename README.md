# FinSolve Technologies — Enterprise RAG Chatbot with Role-Based Access Control

## 1. Project Overview

An internal assistant that answers employees' questions from FinSolve's own documents — finance reports, HR records, marketing reports, engineering docs and the employee handbook — while guaranteeing that each person only ever sees data their role is entitled to. Access control is enforced on the server, at the retrieval layer, before any document is read. On top of that sits a retrieval pipeline built for accuracy (hybrid dense + BM25 search, Reciprocal Rank Fusion, cross-encoder reranking with a relevance threshold), conversational follow-ups (session memory and conditional query rewriting), and an RBAC-namespaced Redis semantic cache. Answers are grounded only in authorized, relevance-checked chunks; when nothing relevant is found the assistant says so instead of guessing.

**Stack:** FastAPI · Streamlit · LangChain · ChromaDB · rank-bm25 · sentence-transformers (cross-encoder) · Redis Stack + redisvl · Groq (`openai/gpt-oss-120b`) · python-jose (JWT) · bcrypt

## 2. Architecture Diagram

```
 Streamlit UI ── login ──► POST /auth/login-json ──► JWT (sub, role, exp)
      │                                                   │ stored in st.session_state
      ▼                                                   ▼
 POST /chat  {question, session_id}   +   Authorization: Bearer <JWT>
      │
      ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 1. JWT validation (get_current_user)          → 401 if invalid / expired │
 │ 2. RBAC: role → authorized collections        (decided BEFORE retrieval) │
 │ 3. Session: last 6 messages for user:session_id                          │
 │ 4. Conditional query rewriting  (only if follow-up AND history exists)   │
 │ 5. Semantic cache lookup   namespace = role : collections : version      │──► HIT → return
 │ 6. Retrieval policy per authorized collection                            │
 │      ├─ finance, hr_docs   → Dense (Chroma, 15) + BM25 (15) ─► RRF (15)  │
 │      └─ marketing, eng, general → Dense (Chroma, 15)                     │
 │    c_level: RRF across all per-collection lists                          │
 │ 7. Cross-encoder reranker (15 candidates → top 6)                        │
 │ 8. Relevance threshold (score ≥ 0.1)  ── nothing passes ──► safe answer  │──► no LLM call
 │ 9. LLM (Groq) with ONLY reranked, authorized chunks                      │
 │10. LLM declined? → sources = [] and skip cache                           │
 │11. Save turn to session history · store answer in cache (TTL 1h)         │
 └──────────────────────────────────────────────────────────────────────────┘
      │
      ▼
 {answer, sources, role, session_id, cache_hit}  +  X-Request-ID header
 One structured JSON log line per request (no tokens, passwords or document text)
```

## 3. Why Each Component Exists

**RBAC (server-side, at the retrieval layer).** The role comes from the signed JWT, never from the request body, and is mapped to a fixed list of ChromaDB collections before any search runs. Unauthorized collections are never queried — not queried-then-filtered, and not "hidden" by telling the LLM to ignore them. Prompt-level restrictions can be bypassed by prompt injection; data the model never receives cannot leak. Caching, reranking and query rewriting all operate strictly inside the already-authorized collection list.

**Hybrid Search (Dense + BM25) for HR and Finance.** Embeddings capture meaning but blur exact tokens: `FINEMP1005` and `FINEMP1010` look nearly identical to an embedding model, as do specific figures and names. HR records and financial reports are full of such identifiers. BM25 matches them lexically. In the evaluation, dense-only retrieval found **1 of 3** HR questions (only the lookup by name); hybrid found **3 of 3**, including both lookups by employee ID.

**Reciprocal Rank Fusion (RRF).** Dense cosine scores and BM25 scores live on unrelated scales, so concatenating or adding them lets one retriever dominate arbitrarily. RRF uses only ranks — `score = Σ 1/(k + rank)` with `k = 60` — so a chunk ranked well by both retrievers rises to the top, duplicates are merged, and no score calibration is needed. The same fusion merges per-collection lists for c_level users.

**Cross-Encoder Reranker.** First-stage retrieval embeds the question and each chunk separately (fast, approximate). A cross-encoder reads question and chunk *together* and judges relevance much more precisely, but is too slow to run over a whole collection — so it runs as a second stage on the 15 fused candidates and keeps the best 6. In the evaluation it raised MRR from 0.517 to 0.875 and Recall@5 from 0.833 to 0.917 (config C → E). Scores are forced through a sigmoid so they are always in [0, 1], whatever model is configured.

**Relevance Threshold.** Chunks scoring below `RERANKER_SCORE_THRESHOLD` (0.1) are discarded. If nothing passes, the API returns *"I was unable to find sufficiently relevant information in your authorized documents to answer this question."* and **the LLM is not called** — no context, no chance to hallucinate. This is also what a user sees when asking about another department: their own collection simply has nothing relevant (in the evaluation, the best-scoring chunk for the two unauthorized questions scored 0.0000 and 0.0087).

A second guard sits after the LLM: the system prompt tells it to reply exactly *"I don't have access to that information."* when the context does not directly answer the question. When it does (tolerating curly apostrophes, quotes and a missing period), the response carries `sources: []` — no document was actually used, so listing the retrieved ones would be misleading — and the refusal is **not cached**, so a later retrieval fix or re-ingest is not masked for an hour.

**Redis Stack Semantic Cache.** Repeated or reworded questions (cosine similarity ≥ 0.92) are answered from Redis in milliseconds instead of re-running retrieval and the LLM. Every entry is tagged with the namespace `role:sorted_collections:RETRIEVAL_VERSION`, and lookups filter on that exact namespace, so a finance user and an HR user never share an entry even for identical questions. Entries expire after `CACHE_TTL` (1 hour); bumping `RETRIEVAL_VERSION` invalidates everything at once. If Redis is down or lacks the search module, the chatbot keeps working uncached and logs a warning.

**Conversational Memory.** Each conversation keeps its last 6 messages (3 exchanges) in memory, keyed by `username:session_id`. Including the username means a user who sends someone else's `session_id` still gets an empty, separate history. The store implements a `SessionStore` protocol, so a Redis or PostgreSQL backend can replace it without changing callers.

**Conditional Query Rewriting.** Follow-ups like *"How does that compare to the marketing spend?"* are meaningless to a retriever on their own. When — and only when — there is history **and** the question contains a context signal (`it`, `that`, `those`, `compared to`, …, matched as whole words) or starts with a phrase like *"what about"*, the LLM rewrites it into a standalone question used for retrieval, reranking and the cache key. Self-contained questions skip the extra LLM call entirely. The user's original wording is still what gets answered.

**Contextual Chunk Headers.** Documents are split into ~1000-character chunks, and a chunk on its own loses where it came from: every quarterly marketing report has a `## Campaign Highlights` section, and the Q2 one contains neither "Q2" nor the report's name. At ingest every chunk is therefore prefixed with `<document title> › <section>`, e.g. `Comprehensive Marketing Report - Q2 2024 › Campaign Highlights` or `HR Data › employee_id: FINEMP1005`. The title is the file's H1 (or the file name for CSVs); the section is the level-2 heading — `## …` or a dash-underlined line — that covers most of the chunk. The header is placed first because the embedding model reads only ~256 tokens, and the identical text is written to ChromaDB and the BM25 JSON. Chunk boundaries are unchanged. This moved the Q2 Campaign Highlights chunk from dense rank 17 to 8 for *"Which marketing campaign performed best in Q2 2024?"*.

## 4. Department Retrieval Policy Table

| Department  | Strategy | Dense | BM25 | Why |
|-------------|----------|-------|------|-----|
| finance     | hybrid   | 15    | 15   | Exact figures, quarters and vendor names (`$2.4 billion`, `Q3`, `Vendor Services`) benefit from lexical matching |
| hr_docs     | hybrid   | 15    | 15   | Employee IDs (`FINEMP1005`), names and field values need exact-token matching that embeddings blur |
| marketing   | semantic | 15    | 0    | Narrative campaign reports; adding BM25 here did not help (config D scored below C in the evaluation) |
| engineering | semantic | 15    | 0    | Conceptual technical prose (architecture, SDLC) suits semantic search |
| general     | semantic | 15    | 0    | Policy handbook questions are phrased in varied natural language |

Depths come from `DENSE_TOP_K` / `BM25_TOP_K`; strategies live in `app/services/retrieval_policy.py`.

## 5. Setup Instructions

Requires Python 3.10+ (developed on 3.13).

**Step 1: Clone the repo**
```bash
git clone https://github.com/vgpatel124/Role-based-Chatbot.git
cd Role-based-Chatbot
python -m venv venv && source venv/bin/activate
```

**Step 2: Create `.env` from `.env.example`**
```bash
cp .env.example .env
```
Set `GROQ_API_KEY` to a real key from [console.groq.com/keys](https://console.groq.com/keys) (it starts with `gsk_`), and change `JWT_SECRET_KEY`.

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Install and start Redis Stack** (optional — without it the app runs uncached)

macOS (Homebrew). Recent Homebrew versions require trusting third-party taps first:
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
Plain Redis will not work for caching — the semantic cache needs the RediSearch vector module that Redis Stack includes.

**Step 5: Ingest documents**
```bash
python app/ingest.py
```
Builds the five ChromaDB collections (each chunk prefixed with its contextual header) and writes `chroma_db/bm25_docs/{collection}.json` for BM25. `Chroma.from_documents` appends, so to re-ingest cleanly delete `chroma_db/` first.

**Step 6: Start the API**
```bash
uvicorn app.main:app --reload --port 8000
```
The first start downloads the embedding model (~90 MB) and the reranker (~90 MB). Look for `Redis Stack cache connected successfully` and `Application startup complete.`

**Step 7: Start the UI**
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501.

## 6. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(none)* | Groq API key used for answers and query rewriting |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model for answers and query rewriting |
| `JWT_SECRET_KEY` | `finsolve-secret-key-change-in-production` | HMAC secret used to sign JWTs — change it |
| `JWT_EXPIRE_MINUTES` | `30` | Token lifetime in minutes |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB directory; BM25 JSON lives in `bm25_docs/` inside it |
| `RESOURCES_PATH` | `./resources` | Root of the source documents (`data/<department>/`) |
| `REDIS_URL` | `redis://localhost:6379` | Redis Stack connection URL |
| `CACHE_TTL` | `3600` | Seconds a cached answer lives |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Minimum cosine similarity between questions for a cache hit |
| `CACHE_ENABLED` | `true` | Set `false` to bypass the semantic cache |
| `DENSE_TOP_K` | `15` | Chunks fetched per collection by dense search |
| `BM25_TOP_K` | `15` | Chunks fetched per collection by BM25 (hybrid collections only) |
| `RRF_K` | `60` | RRF smoothing constant in `1 / (k + rank)` |
| `RRF_CANDIDATE_COUNT` | `15` | Candidates kept after fusion |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face cross-encoder used for reranking (English, ~90 MB) |
| `RERANKER_CANDIDATE_COUNT` | `15` | Candidates passed to the cross-encoder |
| `RERANKER_FINAL_TOP_K` | `6` | Chunks sent to the LLM after reranking |
| `RERANKER_SCORE_THRESHOLD` | `0.1` | Minimum reranker score (sigmoid, 0–1) for a chunk to be used |
| `CONVERSATION_HISTORY_LENGTH` | `6` | Messages of history kept per session |
| `QUERY_REWRITING_ENABLED` | `true` | Set `false` to disable follow-up rewriting |
| `RETRIEVAL_VERSION` | `v2` | Part of every cache namespace — bump to invalidate all cached answers |

## 7. Test Users

| Username | Password    | Role        | Can Access |
|----------|-------------|-------------|------------|
| alice    | password123 | finance     | `finance` collection |
| bob      | password123 | marketing   | `marketing` collection |
| carol    | password123 | hr          | `hr_docs` collection |
| dave     | password123 | engineering | `engineering` collection |
| eve      | password123 | c_level     | all five collections |
| frank    | password123 | employee    | `general` collection (employee handbook) |

Users are defined in memory in `app/auth.py` for demonstration only.

## 8. Example Queries by Role

These were run against the live system during development:

| User | Question | Result |
|------|----------|--------|
| alice (finance) | What was the gross margin in Q3 2024? | "62%" — source `quarterly_financial_report.md` |
| alice (finance) | *same question again* | Served from cache (`⚡ cached`), ~0.2 s |
| alice (finance) | What is the leave balance of FINEMP1005? | Safe "unable to find…" answer, no sources, LLM not called |
| carol (hr) | What is the leave balance of FINEMP1005? | "4 days" — source `hr_data.csv` |
| dave (engineering) | What are the stages of the SDLC at FinSolve? | Scrum ceremonies — source `engineering_master_doc.md` |
| frank (employee) | What is the company leave policy? | Leave types and entitlements — source `employee_handbook.md` |
| frank (employee) | What was the gross margin in Q3 2024? | Safe "unable to find…" answer |
| eve (c_level) | What was the total revenue in 2024? | "$9.4 billion" |
| eve (c_level) | How does that compare to the marketing spend? | Rewritten to a standalone question → "$9.4B revenue vs $2.3B marketing spend" |
| bob (marketing) | Which marketing campaign performed best in Q2 2024? | "The Southeast Asia influencer-partnership campaign … ≈ 40,000 new customer sign-ups" — matches the Q2 Campaign Highlights chunk (declined with `sources: []` before contextual headers) |

Other good questions: *"What were the total vendor services expenses in 2024?"* (alice), *"Where is Krishna Verma located?"* (carol), *"What results did the Q2 2024 influencer partnerships achieve?"* (bob), *"How are FinSolve's microservices deployed and scaled?"* (dave), *"How many days of sick leave do employees get?"* (frank).

## 9. Running Tests

```bash
pytest tests/ -v
```

90 tests, ~3 seconds. Every external dependency (Groq, ChromaDB, BM25 files, Redis, both Hugging Face models) is mocked; authentication and the RBAC mapping are real. The suite needs no network, Redis or API key.

| File | Covers |
|------|--------|
| `test_rbac.py` | Collections per role; cross-department questions return no data and never call the LLM; invalid / missing / expired JWT → 401 |
| `test_cache.py` | Hit skips LLM and retrieval; miss stores result; per-role namespaces; TTL; Redis down / erroring / disabled |
| `test_conversation.py` | When rewriting triggers; 6-message cap; session isolation (including across users) |
| `test_retrieval.py` | Policies; BM25 fallback when JSON is missing/corrupt; RRF scoring and dedup; finance retrieval never touches other collections |
| `test_reranker.py` | Ordering, threshold, `top_k`, metadata, sigmoid scores, fail-closed on errors |
| `test_ingest.py` | Contextual headers: title/section parsing (ATX and setext), majority-section choice, CSV row headers, metadata preserved, distinct Q1/Q2 headers for the same section, BM25 JSON text identical to Chroma text |
| `test_e2e.py` | Login → chat → answer with sources/session_id; second request served from cache; no cross-role cache sharing; LLM refusals return `sources: []` and are not cached |

## 10. Running Evaluation

```bash
python evaluation/evaluate_retrieval.py            # add --verbose for per-question detail, --k N to change the cut-off
```

Compares five retrieval configurations on `evaluation/eval_dataset.json` (12 answerable questions across all departments + 2 unauthorized-access questions). Retrieval only — the LLM is not called. A chunk counts as relevant only if it comes from the expected source **and** contains the expected passage (e.g. `employee_id: FINEMP1005`), so single-file collections cannot score trivially.

Measured results with contextual chunk headers (K = 5, reranker `ms-marco-MiniLM-L-6-v2`, threshold 0.1):

| Config | Description | Recall@5 | Precision@5 | MRR | Keyword coverage | ms/query |
|--------|-------------|----------|-------------|-----|------------------|----------|
| A | Dense only (original) | 0.583 | 0.117 | 0.364 | 0.653 | 13.3 |
| B | Dense + Reranker | 0.667 | 0.167 | 0.625 | 0.778 | 139.5 |
| C | Hybrid for HR/Finance, semantic elsewhere | 0.833 | 0.167 | 0.517 | 0.875 | 6.8 |
| D | Hybrid + RRF on every collection | 0.750 | 0.150 | 0.507 | 0.806 | 5.8 |
| E | **Full pipeline (C + Reranker) — production** | **0.917** | **0.217** | **0.875** | **1.000** | 105.1 |

Before contextual headers (same dataset and settings):

| Config | Recall@5 | Precision@5 | MRR |
|--------|----------|-------------|-----|
| A | 0.500 | 0.117 | 0.340 |
| B | 0.583 | 0.117 | 0.500 |
| C | 0.833 | 0.183 | 0.461 |
| D | 0.833 | 0.167 | 0.475 |
| E | 0.750 | 0.150 | 0.667 |

Accuracy metrics are deterministic across runs; `ms/query` varies with machine load.

Unauthorized-access samples: **0 chunks from a wrong department in every configuration**, before and after; with the reranker (B, E) both return 0 chunks, i.e. the safe answer without an LLM call.

Reading the results:
- **Production (E) improved** from 9 to 11 of 12 questions (Recall 0.750 → 0.917, MRR 0.667 → 0.875). The two gains are the Q2 influencer-campaign question and Q3 revenue; the only remaining E miss is the Q2-vs-Q3 marketing-spend comparison.
- Hybrid retrieval remains the largest single gain for HR/finance lookups (A → C).
- **Not every configuration improved.** C is unchanged overall (gained the Q2 campaign question, lost *"gross margin 2024 vs 2023"*: its chunk fell from inside the top 15 to dense rank 16, behind quarterly chunks whose headers now contain "2024", so it only survives via BM25; the reranker restores it to rank 1 in E). **D regressed** 0.833 → 0.750: running BM25 on marketing/engineering now matches the document title repeated in every chunk (e.g. "FinSolve" appears in all 38 engineering chunks), which pushes the Kubernetes chunk out of the top 5. D is not the production policy — this reinforces keeping BM25 to HR/finance.
- Keyword coverage now also counts header text (e.g. a "Q3" keyword can match a header), so it is slightly inflated relative to the earlier run. Recall, Precision and MRR are not affected: relevance is decided by the chunk's body text (`expected_chunk_contains`) and source.
- With only 12 scored questions each one moves Recall by 0.083 — treat these as directional. Precision@5 is capped at 0.2 for questions with a single relevant chunk.

## 11. Known Limitations

- Users are stored in memory in `app/auth.py` (no database, no user management).
- Conversation history lives in process memory: lost on restart and not shared between multiple API workers.
- The BM25 index is rebuilt only when `python app/ingest.py` is re-run; document changes are not picked up automatically.
- Re-running ingestion appends to ChromaDB — delete `chroma_db/` first to avoid duplicate chunks.
- The English `ms-marco-MiniLM-L-6-v2` reranker cannot distinguish between CSV rows by an exact ID (all rows score ~0.99 for "leave balance of …"); hybrid retrieval compensates, but the right row is not always ranked first.
- **Repeated sections across reports (partly resolved).** Contextual chunk headers fixed the Q2 campaign question (the Campaign Highlights chunk moved from dense rank 17 to 8 and is ranked first by the reranker), but *"How did the marketing spend in Q2 2024 compare to Q3 2024?"* still misses in every configuration — the Q4 report states the same "$2.5 million" figure, and a comparison question needs two specific chunks from two files in the top 5.
- Headers also enter BM25, so the document title repeated in every chunk adds a shared term to each one. This does not affect production (BM25 runs only on HR/finance) but lowered the BM25-everywhere configuration D.
- The LLM cites sources in its own format and sometimes cites the header (e.g. `【HR Data › employee_id: FINEMP1005】`) or invents line ranges (`†L1-L5`). The `sources` field in the API response is the reliable list.
- Endpoints are synchronous; each request holds a worker thread for the duration of the LLM call.

## 12. Future Improvements

- PostgreSQL user store with proper user management.
- Redis-backed `SessionStore` so conversations survive restarts and scale across workers.
- Async FastAPI endpoints and a streaming LLM response.
- Automatic document sync: re-index a collection when its source files change.
- Exact-identifier boost after reranking for HR/finance lookups.
- Per-quarter / per-document metadata filters (e.g. a question naming "Q2" and "Q3" retrieves from exactly those reports) for comparison questions.
- A larger evaluation set, plus answer-level metrics (e.g. faithfulness) with an LLM judge.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/login` | – | OAuth2 form login → JWT |
| POST | `/auth/login-json` | – | JSON login → JWT (used by Streamlit) |
| GET | `/auth/me` | Bearer | Current username and role |
| POST | `/chat` | Bearer | `{question, session_id?}` → `{answer, sources, role, session_id, cache_hit}` + `X-Request-ID` header |
| GET | `/health` | – | Liveness check |
