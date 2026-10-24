# Design Notes

Implementation details behind the [README](../README.md): how each component works, the full configuration reference, test coverage, evaluation methodology and analysis, and detailed limitations.

## Request pipeline

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

## Components

**RBAC (server-side, at the retrieval layer).** The role comes from the signed JWT, never from the request body, and is mapped to a fixed list of ChromaDB collections (`get_collections_for_role` in `app/rag.py`) before any search runs. Unauthorized collections are never queried — not queried-then-filtered, and not "hidden" by telling the LLM to ignore them. Prompt-level restrictions can be bypassed by prompt injection; data the model never receives cannot leak. Caching, reranking and query rewriting all operate strictly inside the already-authorized collection list; `retrieve_for_collections` makes no access decisions of its own.

**Hybrid search (dense + BM25) for HR and finance.** Embeddings capture meaning but blur exact tokens: `FINEMP1005` and `FINEMP1010` look nearly identical to an embedding model, as do specific figures and names. HR records and financial reports are full of such identifiers, so BM25 (`rank_bm25.BM25Okapi` over `chroma_db/bm25_docs/{collection}.json`, loaded once per process) matches them lexically. A small stopword list keeps question scaffolding ("what is the …") from dominating; chunks with zero term overlap are dropped. If a BM25 file is missing or corrupt, a warning is logged and the collection falls back to dense-only. In the evaluation, dense-only retrieval found 1 of 3 HR questions (only the lookup by name); hybrid found 3 of 3, including both lookups by employee ID.

**Reciprocal Rank Fusion.** Dense cosine scores and BM25 scores live on unrelated scales, so concatenating or adding them lets one retriever dominate arbitrarily. RRF uses only ranks — `score = Σ 1/(k + rank)`, `k = 60` — so a chunk ranked well by both retrievers rises to the top. Duplicates are merged by an MD5 of `page_content`. Each fused chunk records `rrf_score`, `dense_rank`, `bm25_rank` and `retrieval_source` (`dense` / `bm25` / `both`). For c_level users the per-collection lists are fused again into one global ranking (`global_rrf_score`).

**Reference resolution.** An HR row holds `manager_id: FINEMP1007`, not the manager's details, and that row shares no words with "who is the manager of FINEMP1010?", so it ranked 12th and never reached the LLM. After reranking, `app/services/reference_resolver.py` reads `manager_id` from the retrieved HR rows and adds the referenced rows (at most 2) by exact lookup in the persisted BM25 chunks. It adds no access: it only looks in the collection the referencing chunk came from, only when that collection is in the role's already-filtered list, and only for `hr_data.csv` rows.

**Cross-encoder reranker.** First-stage retrieval embeds question and chunk separately (fast, approximate). A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads them *together* and judges relevance more precisely, but is too slow for a whole collection, so it runs on the 15 fused candidates and keeps the best 6. The model loads once at startup. Scores are always passed through a sigmoid: `ms-marco-*` models ship with an identity activation and return raw logits, so without it the threshold would mean something different per model. If scoring fails, the reranker fails closed (no chunks → safe answer). In the evaluation it raised MRR from 0.517 to 0.875 and Recall@5 from 0.833 to 0.917 (config C → E).

**Relevance threshold and declined answers.** Chunks scoring below `RERANKER_SCORE_THRESHOLD` (0.1) are discarded. If nothing passes, the API returns *"I was unable to find sufficiently relevant information in your authorized documents to answer this question."* and the LLM is not called. This is also what a user sees when asking about another department: their own collection has nothing relevant (in the evaluation, the best chunk for the two unauthorized questions scored 0.0000 and 0.0087). A second guard sits after the LLM: the system prompt tells it to reply *"I don't have access to that information."* when the context does not directly answer the question. That reply is detected (tolerating curly apostrophes, surrounding quotes and a missing period), returned with `sources: []` — no document was actually used — and not cached, so a retrieval fix or re-ingest is not masked for the cache TTL.

**Redis Stack semantic cache.** Questions with cosine similarity ≥ 0.92 to a cached one are answered from Redis instead of re-running retrieval and the LLM. Entries are stored with redisvl in a vector index named `rbac_cache_{RETRIEVAL_VERSION}`; each is tagged with the namespace `role:sorted_collections:RETRIEVAL_VERSION`, and lookups use an escaped exact-match tag filter on it (a returned entry whose namespace differs is rejected as a second check). A finance user and an HR user therefore never share an entry. TTL is applied to the keys redisvl actually writes. Follow-up questions are looked up by their rewritten, standalone form, so one conversation's context cannot leak into another's cached answer. Safe answers and refusals are never cached. If Redis is down or lacks the search module, caching is disabled with a warning and reconnection is retried at most every 30 s.

**Conversational memory.** Each conversation keeps its last 6 messages in memory, keyed by `username:session_id`, so a user who sends someone else's `session_id` still gets an empty, separate history. Returned history is a copy. `InMemorySessionStore` implements the `SessionStore` protocol, so a Redis or PostgreSQL backend can replace it without changing callers.

**Conditional query rewriting.** Rewriting runs only when there is history **and** the question contains a context signal (`it`, `that`, `those`, `compared to`, …, matched as whole words after punctuation is stripped) or starts with a phrase such as *"what about"*. The LLM rewrites it into a standalone question used for retrieval, reranking and the cache key; self-contained questions skip the extra LLM call. The original question is what gets answered, with the standalone form shown to the LLM as "Interpreted as: …". On any LLM error or empty output the original question is used.

**Contextual chunk headers.** Documents are split into ~1000-character chunks (200 overlap). On its own a chunk loses where it came from: every quarterly marketing report has a `## Campaign Highlights` section, and the Q2 one contains neither "Q2" nor the report's name. At ingest every chunk is prefixed with `<document title> › <section>`:

- Title: the file's H1, or a name derived from the file name (`hr_data.csv` → `HR Data`).
- Section: the level-2 heading — `## …` or a dash-underlined line — covering most of the chunk's characters, so a chunk that starts with overlap from the previous section is labelled by its main subject. Chunks are first cut where a new level-2 section starts more than 200 characters before the chunk ends (a shorter stub is repeated by the next chunk's overlap), because a chunk mixing two sections carries only one header: the handbook's hotel and meal limits sat in a chunk labelled "Compensation & Payroll" and the reranker scored it 0.000. CSV rows use their first field (`employee_id: FINEMP1005`).
- The header is placed first because `all-MiniLM-L6-v2` reads only ~256 tokens.
- Chunk boundaries are unchanged; the header is added after splitting. Original metadata is kept and `context_header` is added.
- The identical text is written to ChromaDB and to the BM25 JSON.

For *"Which marketing campaign performed best in Q2 2024?"* this moved the Q2 Campaign Highlights chunk from dense rank 17 (outside the top 15) to 8; the reranker then ranks it first. `RETRIEVAL_VERSION` was bumped to `v2` with this change so answers cached from the old chunks are not served.

**Observability.** Every `/chat` request gets a UUID returned in `X-Request-ID` and produces one JSON log line with role, authorized collections, cache hit, rewrite, per-stage counts, reranker scores and per-stage latency. Tokens, passwords, document text and employee PII are never logged; logged queries are truncated to 300 characters.

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `anthropic`; built in `app/utils/llm.py`, which also reads Claude's block-list replies and sends no sampling parameters to Claude (Claude Haiku 5.5 rejects them) |
| `GROQ_API_KEY` | *(none)* | Groq API key used for answers and query rewriting (`LLM_PROVIDER=groq`) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model for answers and query rewriting |
| `ANTHROPIC_API_KEY` | *(none)* | Anthropic API key (`LLM_PROVIDER=anthropic`, and the RAGAS judge) |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Claude model for answers and query rewriting |
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

## Test coverage

90 tests; every external dependency (Groq, ChromaDB, BM25 files, Redis, both Hugging Face models) is mocked, while authentication (bcrypt + JWT) and the RBAC mapping are real. Two deliberate breakages were used to confirm the suite catches regressions: giving the finance role every collection failed 9 tests, and dropping the role from the cache namespace failed 6.

| File | Covers |
|------|--------|
| `test_rbac.py` | Collections per role; cross-department questions return no data and never call the LLM; a `role` field in the request body is ignored; invalid / missing / expired JWT → 401 |
| `test_cache.py` | Hit skips LLM and retrieval; miss stores result; per-role namespaces; escaped namespace filter; similarity threshold; TTL; Redis down / erroring / disabled |
| `test_conversation.py` | When rewriting triggers; rewrite drives retrieval while the original question is answered; 6-message cap; session isolation (including across users) |
| `test_reference_resolver.py` | Manager rows resolved only from an authorized HR collection, nothing looked up for other roles or non-HR sources, de-duplication and cap, exact-line BM25 lookup |
| `test_retrieval.py` | Policies; BM25 fallback when JSON is missing/corrupt; exact-ID ranking; RRF scoring and dedup; finance retrieval never touches other collections |
| `test_reranker.py` | Ordering, threshold, `top_k`, metadata, sigmoid scores, fail-closed on errors |
| `test_ingest.py` | Contextual headers: title/section parsing (ATX and setext), majority-section choice, chunks cut at section boundaries, CSV row headers, metadata preserved, distinct Q1/Q2 headers for the same section, BM25 JSON text identical to Chroma text |
| `test_e2e.py` | Login → chat → answer with sources/session_id; second request served from cache; no cross-role cache sharing; LLM refusals return `sources: []` and are not cached |

## Evaluation methodology and analysis

`evaluation/evaluate_retrieval.py` compares five retrieval configurations on `evaluation/eval_dataset.json`: 12 answerable questions (3 finance, 3 HR, 2 marketing, 2 engineering, 2 general) and 2 unauthorized-access questions. It evaluates retrieval only; the LLM is not called.

- **Relevance:** a chunk counts as relevant only if it comes from an expected source **and** contains the expected passage (`expected_chunk_contains`, e.g. `employee_id: FINEMP1005`). Without the second condition every chunk of a single-file collection would count and Recall would be trivially 100%.
- **Metrics (top K, default 5):** Recall@K = share of questions with ≥ 1 relevant chunk; Precision@K = relevant chunks / K (capped at 0.2 for single-answer questions); MRR = mean 1/rank of the first relevant chunk; keyword coverage = share of expected answer keywords present in the top-K text.
- **RBAC check:** the dataset's expected collections must match the live role mapping, and every configuration must return 0 chunks from an unauthorized department.
- An untimed warm-up runs first so one-off model loading is not charged to config A. Accuracy metrics are deterministic; `ms/query` varies with machine load.

**Configurations:** A dense only · B dense + reranker · C hybrid for HR/finance, semantic elsewhere · D hybrid + RRF on every collection · E = C + reranker (production).

**With contextual chunk headers:**

| Config | Recall@5 | Precision@5 | MRR | Keyword coverage | ms/query |
|--------|----------|-------------|-----|------------------|----------|
| A | 0.583 | 0.117 | 0.364 | 0.653 | 13.3 |
| B | 0.667 | 0.167 | 0.625 | 0.778 | 139.5 |
| C | 0.833 | 0.167 | 0.517 | 0.875 | 6.8 |
| D | 0.750 | 0.150 | 0.507 | 0.806 | 5.8 |
| **E** | **0.917** | **0.217** | **0.875** | **1.000** | 105.1 |

**Before contextual headers (same dataset and settings):**

| Config | Recall@5 | Precision@5 | MRR |
|--------|----------|-------------|-----|
| A | 0.500 | 0.117 | 0.340 |
| B | 0.583 | 0.117 | 0.500 |
| C | 0.833 | 0.183 | 0.461 |
| D | 0.833 | 0.167 | 0.475 |
| E | 0.750 | 0.150 | 0.667 |

**Analysis:**
- Production (E) improved from 9 to 11 of 12 questions. The two gains are the Q2 influencer-campaign question and Q3 revenue; the remaining miss is the Q2-vs-Q3 marketing-spend comparison.
- Hybrid retrieval remains the largest single gain for HR/finance lookups (A → C).
- Not every configuration improved. C is unchanged overall: it gained the Q2 campaign question and lost *"gross margin 2024 vs 2023"*, whose chunk fell to dense rank 16 behind quarterly chunks whose headers now contain "2024"; it survives only via BM25, and the reranker restores it to rank 1 in E. D regressed (0.833 → 0.750): BM25 on marketing/engineering now matches the document title repeated in every chunk ("FinSolve" is in all 38 engineering chunks), pushing the Kubernetes chunk out of the top 5. D is not the production policy; this supports keeping BM25 to HR/finance.
- Keyword coverage now also counts header text, so it is slightly inflated relative to the earlier run. Recall, Precision and MRR are unaffected because relevance is decided by body text and source.
- Unauthorized-access questions: 0 wrong-department chunks in every configuration, before and after; with the reranker (B, E) both return 0 chunks.
- With 12 scored questions each one moves Recall by 0.083 — the results are directional.

### After cutting chunks at section boundaries: why MRR fell, and what was tried

Cutting chunks where a new `##` section starts (so a chunk's header always names the section it belongs to) raised production Recall@5 from 0.917 to 1.000 and lowered MRR from 0.875 to 0.836. Per question (config E, first relevant rank, before → after): the Q2 influencer-campaign question 1 → 2, Q3 revenue 2 → 3, the Q2-vs-Q3 marketing comparison *missed* → 5. The other nine are unchanged.

**Cause.** The relevant chunks' reranker scores did not change (0.9982 and 0.9996 before and after). The chunks *around* them did: sections that used to be glued onto a neighbour are now short, clean, generic summary chunks ("Conclusion", "2024 Annual Summary") that the cross-encoder scores 0.9994 and 0.9998 for a question about one specific quarter. The model's scores are saturated near 1.0 for all of them, so the order among them is decided in the fourth decimal and the model does not separate "revenue in 2024" from "revenue in Q3 2024". On the 41-question set derived from `evaluation/answer_dataset.json` (relevant = right file and all expected facts in the chunk), MRR is 0.943 after against 0.947 before and Recall@5 is 1.000 for both, so the effect is two near-tie flips, not a retrieval regression.

**Alternatives measured on both indexes and both question sets** (Recall@5 / MRR; 12 = retrieval set, 41 = answer set):

| Change | New index, 12 | New index, 41 | Old index, 12 | Old index, 41 |
|--------|---------------|---------------|---------------|---------------|
| Current (reranker alone orders the final list) | 1.000 / 0.836 | 1.000 / 0.943 | 0.917 / 0.875 | 1.000 / 0.947 |
| Blend first-stage rank into the final score (weights 0.5, 1, 2; range shown) | 0.917 / 0.642-0.764 | 0.927-1.000 / 0.683-0.852 | 0.833-0.917 / 0.653-0.771 | 0.951-1.000 / 0.704-0.856 |
| Hybrid (BM25 + RRF) for every department | 0.917 / 0.806 | 1.000 / 0.939 | 0.917 / 0.833 | 1.000 / 0.935 |
| `ms-marco-MiniLM-L-12-v2` | 1.000 / 0.757 | 1.000 / 0.923 | 0.917 / 0.792 | 1.000 / 0.951 |
| `bge-reranker-base` | 1.000 / 0.917 | 1.000 / 0.939 | 1.000 / 0.896 | 1.000 / 0.963 |

Blending rank signals and widening hybrid retrieval are worse everywhere, and the larger MiniLM is worse too. `bge-reranker-base` is better on the 12 questions but equal within noise on the 41, and adopting it would mean a ~280M-parameter model (about 1.1 GB, against ~23M for MiniLM-L6) in every image, heavier inference, and recalibrating `RERANKER_SCORE_THRESHOLD` (the "no relevant context, no LLM call" safety path), so it is not adopted on this evidence. No pipeline change was made. Instead, `python evaluation/evaluate_retrieval.py --check` now fails the build if config E drops below `evaluation/retrieval_thresholds.json` (Recall 1.0, MRR 0.80, 0 RBAC leaks) or if the three near-tie queries (`fin_q3_revenue`, `mkt_campaign_performance`, `mkt_quarter_comparison`) fall below the ranks recorded as `max_first_rank` in the dataset.

### Conflicting figures across documents

`financial_summary.md` gives FY2024 vendor services as $30M (+18%; marketing 40%, training 30%, software 25%); `quarterly_financial_report.md` gives about $510M (four quarterly "Vendor Services" lines of $120M-$135M). They are the same line item at different scales (the two reports also disagree on revenue growth, 25% vs 28%, and operating cash flow, $50M vs $1.5B) and neither defines a scope, so both are kept. Retrieval already ranks the correct annual-breakdown chunk first; what failed was generation, where the model silently chose $510M. The system prompt now requires that when sources give different values for the same figure, each value is reported with its document and the difference is stated. `fin_vendor_drivers` in the answer evaluation checks the amount, growth rate, all three drivers and the cited source; `tests/test_conflicting_sources.py` checks the prompt rule and that both sources reach the LLM with their labels while staying inside the role's authorized collections.

## Observability (Langfuse tracing)

Implemented in `app/utils/tracing.py` and wired into `app/rag.py` and `app/main.py`; the trace layout and settings are in the [README](../README.md#observability-langfuse). Decisions worth knowing:

- **Content is withheld by default, enforced at export.** A `mask_otel_spans` hook deletes every input/output attribute (Langfuse and the OpenTelemetry GenAI names) from every span unless `LANGFUSE_CAPTURE_CONTENT=true`, so privacy does not depend on each call site remembering not to record a prompt, and it covers the LangChain callback handler too. Call sites record the question, answer and prompts normally; the hook decides what leaves. Metadata set by the code is limited to names, counts, numbers and file names. The same hook scrubs JWTs, API keys, e-mail addresses and provider account ids from whatever is exported.
- **Exception text never reaches Langfuse.** The SDK exports an exception's message and stack trace as span events and the status description, which the masking hook cannot alter, and messages can echo question or document text. The context manager in `tracing.py` and a subclass of the LangChain handler therefore record only the error class (and HTTP status for provider errors) and mark the observation `ERROR`; the full detail stays in the application log.
- **One trace per chat turn, one session per user conversation.** Following Langfuse's guidance for chatbots; the session id is `<username>:<session_id>`, the same key as the conversation history, so one user cannot attach traces to another's session.
- **Fixed, verb-first names; run-specific values in metadata** (`search-collection` with a `collection` field, not one name per collection). Names are what evaluators and dashboards target, so they should not change.
- **Integration where it helps, manual where it must.** The LLM calls use Langfuse's LangChain callback handler (model, tokens including reasoning tokens, latency); retrieval steps are manual `retriever` spans because they are custom code.
- **Trace id derived from `X-Request-ID`**, so any request can be found without storing an id, and the request log line carries it.
- **Fail-open.** No keys means no SDK import and no overhead; every SDK call is guarded; Langfuse being unreachable only drops traces. Steps that run outside a trace (scripts calling `retrieve_for_collections` directly) create nothing.
- **Evaluation runs** (`evaluate_answers.py`, `redteam.py`) default `LANGFUSE_TRACING_ENVIRONMENT` to `evaluation`.

Tests (`tests/test_tracing.py`) run the real pipeline against an in-memory exporter and assert on the exported spans: structure, types, nesting, attributes, scores, the withheld-content guarantee, scrubbing, exception handling and fail-open behaviour. The trace layout was also verified against a live Langfuse project by fetching the traces back through the v2 observations API (the older traces endpoint is being removed on 2026-11-16).

Requires Python 3.11+ (the Langfuse SDK's floor); `langfuse>=5,<6` needs OpenTelemetry 1.45, and the ChromaDB gRPC exporter pins its own OpenTelemetry version. When installing into an environment that already has an older exporter, `pip check` reports the conflict and upgrading `opentelemetry-exporter-otlp-proto-grpc` to the same 1.45.x line clears it (this was needed locally).

## Detailed limitations

- The English `ms-marco-MiniLM-L-6-v2` reranker cannot distinguish CSV rows by an exact ID (all rows score ~0.99 for "leave balance of …"); hybrid retrieval compensates, but the right row is not always ranked first.
- *"How did the marketing spend in Q2 2024 compare to Q3 2024?"* is only just retrieved (first relevant chunk at rank 5 in configs B and E, a miss in the others): the Q4 report states the same "$2.5 million" figure, and a comparison question needs two specific chunks from two files in the top 5.
- Headers also enter BM25, so the document title repeated in every chunk adds a shared term to each one. This does not affect production (BM25 runs only on HR/finance) but lowered configuration D.
- The LLM cites sources in its own format and sometimes cites the header (e.g. `【HR Data › employee_id: FINEMP1005】`) or invents line ranges (`†L1-L5`). The `sources` field in the API response is the reliable list.
- `Chroma.from_documents` appends on every ingest run, so `chroma_db/` must be deleted before re-ingesting.
