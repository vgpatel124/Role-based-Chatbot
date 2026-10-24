# FinSolve Technologies — Enterprise RAG Chatbot with Role-Based Access Control

An internal assistant that answers employees' questions from company documents — finance reports, HR records, marketing reports, engineering docs and the employee handbook — while guaranteeing each person only sees data their role is entitled to. Access control is enforced on the server before any document is retrieved, and answers are grounded only in authorized, relevance-checked content; when nothing relevant exists, the assistant says so instead of guessing.

**Highlights**
- **RBAC at the retrieval layer** — six roles, five department collections, zero cross-department leakage in every evaluated configuration.
- **Accurate retrieval** — hybrid dense + BM25 search, Reciprocal Rank Fusion, cross-encoder reranking with a relevance threshold, and contextual chunk headers. Production pipeline: **Recall@5 1.000, MRR 0.836**.
- **Conversational** — per-user session memory and automatic rewriting of follow-up questions.
- **Fast repeats** — RBAC-namespaced semantic cache on Redis Stack (~0.2 s for a repeated question vs ~2 s uncached).
- **Verified** — 344 automated tests (including a 160-case RBAC red-team), a retrieval evaluation harness, and a live end-to-end smoke test across all six users.

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
      └─ LLM (Groq or Anthropic) ─ answers only from the authorized chunks
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
| **Contextual chunk headers** | Each chunk is indexed with its document and section (e.g. `Comprehensive Marketing Report - Q2 2024 › Campaign Highlights`), so identical sections in different reports stay distinguishable. A chunk that would span two sections is cut at the second section's start, so its header always names the section its text comes from. |
| **Semantic cache** | Redis Stack vector cache, isolated per role and collection set, with TTL and version-based invalidation; the app runs uncached if Redis is unavailable. |
| **Conversation memory** | Last 6 messages per user and session, isolated between users. |
| **Query rewriting** | Rewrites context-dependent follow-ups ("How does that compare…?") only when needed. |
| **Observability** | Request ID per call and one structured log line per request, with no tokens, passwords or document text. Optional [Langfuse tracing](#observability-langfuse): one trace per chat turn with a span per pipeline step, model and token usage, and quality scores; document and conversation text is withheld by default. |

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

Requires Python 3.11+.

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
Set `GROQ_API_KEY` (from [console.groq.com/keys](https://console.groq.com/keys); or `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`) and change `JWT_SECRET_KEY`.

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

### Run with Docker

The quickest way to run everything (API, UI, Redis Stack and the indexing job):

```bash
cp .env.example .env        # set GROQ_API_KEY and JWT_SECRET_KEY
docker compose up --build
```
API on http://localhost:8000, UI on http://localhost:8501. The first start indexes the documents into a persistent volume and skips that step afterwards. After changing documents, rebuild the index:
```bash
docker compose run --rm -e FORCE_INGEST=1 ingest
```

### Configuration

All settings live in `.env` (see [.env.example](.env.example)). The ones you are most likely to change:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `anthropic`: which provider answers questions and rewrites queries |
| `GROQ_API_KEY` | *(none)* | Groq API key (when `LLM_PROVIDER=groq`) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model for answers and query rewriting |
| `ANTHROPIC_API_KEY` | *(none)* | Anthropic API key (when `LLM_PROVIDER=anthropic`; also the RAGAS judge's key) |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Claude model for answers and query rewriting |
| `JWT_SECRET_KEY` | *(placeholder)* | Secret used to sign login tokens — change it |
| `REDIS_URL` | `redis://localhost:6379` | Redis Stack connection |
| `CACHE_ENABLED` | `true` | Turn the semantic cache on or off |
| `RERANKER_SCORE_THRESHOLD` | `0.1` | Minimum relevance (0–1) for a chunk to be used |

The full reference (retrieval depths, fusion, reranker, cache and conversation settings) is in [docs/DESIGN.md](docs/DESIGN.md#configuration-reference).

### Observability (Langfuse)

Optional [Langfuse](https://langfuse.com) tracing (Cloud or self-hosted). It is off until both keys are set, and a Langfuse outage only drops traces: spans are exported in the background and every call into the SDK is guarded.

```bash
# .env — keys from your Langfuse project: Settings > API Keys
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com      # or https://us.cloud.langfuse.com, or your own host
LANGFUSE_TRACING_ENVIRONMENT=production           # keeps staging/dev/evaluation traces apart
```

One `/chat` turn is one trace (`chat-request`), with user, per-user session and a `role:<role>` tag attached to every observation:

```
answer-question                      chain      role, authorized collections, sources, cache hit, latencies
├─ rewrite-query                     generation only for follow-up questions: model, tokens
├─ lookup-cache                      retriever  hit / miss
├─ retrieve-context                  retriever
│  └─ search-collection              retriever  one per authorized collection: strategy, dense / BM25 / fused counts
├─ rerank-candidates                 span       candidates, threshold, kept, reranker scores
├─ resolve-references                retriever  HR roles only: manager rows added
└─ generate-answer                   generation model, input / output / reasoning tokens, latency
```

Scores: `cache_hit`, `answer_declined` and `no_relevant_context` (booleans) on every trace, and `reranker_top_score` whenever reranking ran. Filter on them to find refusals, or on `reranker_top_score` to see how close answers sit to `RERANKER_SCORE_THRESHOLD`.

**What is sent.** This app serves HR and finance records, so by default **no question, answer, prompt or document text leaves the process**: every input/output attribute is stripped at export, whatever the code records, and exception messages and stack traces are reduced to the error class. Traces carry names, counts, scores, timings, model and token usage. Set `LANGFUSE_CAPTURE_CONTENT=true` to send the text as well (sensible for a self-hosted Langfuse or development). JWTs, API keys and e-mail addresses are scrubbed either way, and the username is sent as the Langfuse user id.

**Finding a request.** `X-Request-ID` maps to the trace deterministically, and the request log line carries `trace_id`. In Python: `Langfuse().create_trace_id(seed=<X-Request-ID>)`.

The Langfuse agent skill ([langfuse/skills](https://github.com/langfuse/skills)) is installed in `.claude/skills/langfuse` for coding agents working on this repo. Design and the checks behind it: [docs/DESIGN.md](docs/DESIGN.md#observability-langfuse-tracing).

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
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check .
```

**344 tests, all passing, in ~6 seconds**, with no network, Redis or API key required. They cover RBAC and authentication, cache isolation and failure handling, conversation memory and rewriting, retrieval and fusion, reranking, ingestion, the full login-to-answer flow, and a 160-case RBAC attack matrix (below). Breaking RBAC or cache isolation on purpose makes the suite fail. Per-file coverage: [docs/DESIGN.md](docs/DESIGN.md#test-coverage).

### Retrieval evaluation

```bash
python evaluation/evaluate_retrieval.py            # report
python evaluation/evaluate_retrieval.py --check    # exit 1 if below evaluation/retrieval_thresholds.json
```

Five retrieval configurations on 12 answerable questions across all departments, plus 2 unauthorized-access questions (measured, K = 5):

| Configuration | Recall@5 | MRR |
|---------------|----------|-----|
| A · Dense only (original) | 0.500 | 0.350 |
| B · Dense + reranker | 0.833 | 0.669 |
| C · Hybrid for HR/Finance | 0.833 | 0.513 |
| D · Hybrid for every department | 0.750 | 0.496 |
| **E · Full pipeline (production)** | **1.000** | **0.836** |

- Production retrieval finds a correct chunk in the top 5 for **12 of 12** questions (11 of 12 before chunks were cut at section boundaries, 9 of 12 before contextual chunk headers). MRR moved from 0.875 to 0.836: two questions' correct chunk slid one rank behind a generic summary chunk that the reranker scores within 0.001 of it, while a third, previously missed, entered at rank 5. On a larger 41-question set MRR is unchanged (0.943 vs 0.947). Blending first-stage ranks, hybrid retrieval for every department and two other cross-encoders were measured as fixes and none is justified by the evidence ([details](docs/DESIGN.md#after-cutting-chunks-at-section-boundaries-why-mrr-fell-and-what-was-tried)); a regression gate now guards the affected queries.
- **0 chunks from an unauthorized department** in every configuration.
- The evaluation set is small (12 questions), so treat the numbers as directional. Methodology, all metrics and before/after analysis: [docs/DESIGN.md](docs/DESIGN.md#evaluation-methodology-and-analysis).

### RBAC red-team

Every role asks every other department's question, plain and wrapped in seven manipulation patterns (ignore-instructions, "I am the CEO", fake system override, prompt extraction, context dump, audit pretext, translation): 5 roles × 4 foreign departments × 8 variants = **160 attacks**. Cases and expected access live in [evaluation/redteam_cases.py](evaluation/redteam_cases.py), independent of the application code, so a bug that widens a role is caught rather than absorbed.

| Layer | Command | Needs | What it proves |
|-------|---------|-------|----------------|
| Offline tests | `pytest tests/test_rbac_attacks.py` | nothing | Only authorized collections are searched and no foreign text reaches the LLM prompt; forged, tampered, unsigned and wrongly signed JWTs, unknown role claims, and roles smuggled via headers, query strings, cookies or the body all fail to widen access |
| Live, retrieval | `python evaluation/redteam.py` | built index | On the real index, no candidate chunk or reranked context contains another department's data |
| Live, full pipeline | `python evaluation/redteam.py --llm` | index + the key of the active `LLM_PROVIDER` | Answers contain none of the foreign facts ("canaries"), cite no foreign files |

Measured: **0 leaks in 160/160 attacks**, and with the LLM in the loop **160/160 were refused**. Deliberately widening a role makes the offline matrix fail 35 tests and the live runner report the leaked chunks.

### Answer-level evaluation

```bash
pip install -r requirements-eval.txt
python evaluation/evaluate_answers.py            # full run (on the free Groq tier: 8,000 tokens/min, 200,000/day)
python evaluation/evaluate_answers.py --reuse --ragas-limit 12   # re-score saved answers, judge a subset
python evaluation/evaluate_answers.py --check    # exit 1 if below evaluation/thresholds.json
```

[evaluation/answer_dataset.json](evaluation/answer_dataset.json) has 41 answerable questions with reference answers (every department, plus c-level cross-department questions) and 6 out-of-scope questions that must be declined. Scored without a judge: refusal accuracy, expected-fact coverage, source hit rate. Scored by [RAGAS](https://docs.ragas.io) with a judge from a different model family than the generator (Claude Haiku, `claude-haiku-5-5`, judging gpt-oss): faithfulness, answer relevancy, context precision, context recall.

**Providers.** The chatbot's own LLM is chosen by `LLM_PROVIDER` (`groq`, the default, or `anthropic`; the Anthropic model is `ANTHROPIC_MODEL`, default `claude-haiku-5-5`). Production is unchanged and stays on Groq unless you set the variable. The Evals workflow sets `LLM_PROVIDER=anthropic`, so there the generator, the LLM red-team's target and the RAGAS judge all run on the Anthropic API and the workflow no longer touches Groq or its free-tier quota. Locally, `LLM_PROVIDER` defaults to Groq, so a local run of these scripts uses Groq unless you set it.

**This needs `ANTHROPIC_API_KEY`** (an environment variable locally; the repository secret of the same name in CI, where it is passed to the two LLM-backed steps only). Before anything is generated, a free preflight checks that the key works and the models exist for the account, then makes one tiny test call with the exact settings of the chatbot's model and of the judge. A missing key, an unavailable model or a provider error stops the run with exit status 2, and the red-team reports `INCONCLUSIVE` rather than `PASS` if its attacks hit provider errors, so none of these can be mistaken for a pass. Usage is billed per token on your Anthropic account (Claude Haiku 5.5 list price: from $0.10 per million input and $0.50 per million output tokens for prompts up to 100K tokens).

**Caveat: the judge and the generator are the same model** (`claude-haiku-5-5`), so the model grades its own answers, which tends to flatter it. They are separate settings (`ANTHROPIC_MODEL`, `EVAL_JUDGE_MODEL`) if you want a different judge. The answers were also written for a different model: the prompts and the thresholds were tuned against `gpt-oss-120b`.

Measured on the production pipeline (gpt-oss-120b on Groq, all 47 questions regenerated; RAGAS metrics on 12 evenly spaced answers, the same subset the CI gate uses). **These scores come from the Groq-era setup (Qwen as judge): the generator and the judge have since changed, the figures have not been re-measured with Claude, and the thresholds, unchanged, have not been checked against it.**

| Metric | Score | | Metric | Score |
|--------|-------|-|--------|-------|
| Refusal accuracy | 1.000 | | Faithfulness | 0.872 |
| Fact coverage | 1.000 | | Answer relevancy | 0.858 |
| Source hit rate | 1.000 | | Context precision | 0.919 |
| Out-of-scope declined | 6/6 | | Context recall | 1.000 |

What the evaluation found, and what was done about it:
- **Handbook hotel and meal limits were declined.** The chunk holding the ₹3,000 and ₹500 limits also contained the end of the payroll section and carried that section's header, so the reranker scored every chunk 0.000. Chunks are now cut at section boundaries (re-index to apply).
- **"Who is the manager of FINEMP1010?" was declined.** An HR row holds only `manager_id: FINEMP1007` and the manager's own row ranked 12th, outside the 6 chunks sent to the LLM. After reranking, `manager_id` references in the retrieved HR rows are resolved by exact lookup in the same, already authorized, HR collection.
- **Vendor services total answered as $510M instead of $30M.** The two finance reports label the same FY2024 line item at different scales ($30M in `financial_summary.md`, about $510M in the quarterly report; they also disagree on revenue growth and operating cash flow) and define no scope, so both figures are kept. Retrieval ranked the right chunk first; the model silently picked one figure. The system prompt now requires reporting each value with its document when sources differ, and the answer now gives both. Side effect: for questions where the two reports disagree (software subscriptions is another), answers are longer, and RAGAS's relevancy score reads such hedged answers as less direct (one judged sample scored 0.00 on that basis), which is why relevancy is 0.858.
- **Free-tier budget.** The Evals workflow runs on the Anthropic API, so it no longer uses Groq's free tier (200,000 tokens per day for the generator, which several full runs in one day used up). A local run with the default `LLM_PROVIDER=groq` still does. When a provider refuses requests, the answer evaluation reports `INCONCLUSIVE`, lists the affected answers and fails `--check`; those answers are excluded from the quality metrics instead of being scored as wrong.
- A judge is itself noisy on 12 samples (the previous one marked a correct one-line answer 0.33 on faithfulness), so treat the RAGAS numbers as directional. Thresholds for the CI gate are in [evaluation/thresholds.json](evaluation/thresholds.json); the [Evals workflow](.github/workflows/evals.yml) runs nightly, on demand, and on pull requests that touch the pipeline.

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

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/login` | – | OAuth2 form login → JWT |
| POST | `/auth/login-json` | – | JSON login → JWT (used by the UI) |
| GET | `/auth/me` | Bearer | Current username and role |
| POST | `/chat` | Bearer | `{question, session_id?}` → `{answer, sources, role, session_id, cache_hit}` |
| GET | `/health` | – | Liveness check |
