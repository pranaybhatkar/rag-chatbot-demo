# PRD — Mutual Fund FAQ RAG Chatbot (Facts-Only Q&A)

| Field | Value |
| --- | --- |
| Document ID | PRD-MF-RAG-001 |
| Version | **1.3** (supersedes 1.2) |
| Status | Ready for build — 1 interpretation (C-3) and 4 open questions (Q1, Q2, Q4, Q5) remain; none block a start |
| Owner | *(unassigned)* |
| Last updated | 2026-09-27 |
| Source of truth | `problemstatement.txt` (2,886 bytes, 38 lines) — *Milestone brief: Mutual Fund FAQs (Facts-Only Q&A)* |
| Deliverable type | RAG Chatbot — prototype app/notebook or ≤3-min demo video |
| Companion doc | `architecture.md` v1.0 — pipeline design; implements this PRD |

> **Revision note — v1.0 → v1.1.** v1.0 was written against an empty `problemstatement.txt`. The real brief was supplied later and **contradicted v1.0 in four material ways**. All corrected in v1.1; change log in §14.2.
>
> **Revision note — v1.1 → v1.2.** Compliance audit against all 38 lines of the brief. Fixed: `scheme_faq` doc type added (§7.2, named in brief line 8); example questions re-spread across schemes (§10.1); milestones realigned to the brief's *submission* list (§14.3); chunking decision procedure specified (§12.4). Resolved by stakeholder decision: source-tier conflict C-1 (Groww primary + official supplement), performance-claims conflict C-2 (ban as written), transparency line now applies to **all** responses incl. refusals (R7), and the `as_of_date` derivation rule (Q6). **No code has been written against this spec** — see §14.6.

---

## 1. Overview

A small, facts-only FAQ assistant over **one AMC (HDFC Mutual Fund)** and **5 schemes**, grounded in public web pages only. It answers factual questions — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements — with **one source link per answer**, in **≤ 3 sentences**, and **refuses any opinionated or portfolio question**.

### 1.1 Problem

Retail users comparing HDFC schemes must open five separate pages to answer basic factual questions, and support/content teams answer the same questions repeatedly. General-purpose chatbots are unsafe for this: they hallucinate expense ratios and exit loads, quote stale NAVs, drift into investment advice, and give no way to verify a claim. In this domain a wrong number is a direct financial harm.

### 1.2 Audience (per brief)

- **Retail users** comparing schemes across these 5 funds.
- **Support/content teams** answering repetitive mutual-fund questions.

### 1.3 Solution

A RAG pipeline — **Loading → Chunking → Embedding → Vector Store → Retrieval → Generation** — over a curated public corpus, with a **deterministic post-generation validator** that enforces every hard constraint in code before any text reaches the user.

**Core design principle:** the LLM is an *untrusted proposal generator*. All constraints are re-verified deterministically after generation. A prompt-only guarantee is not acceptable here.

---

## 2. Goals and Non-Goals

### 2.1 Goals

| ID | Goal |
| --- | --- |
| G1 | Answer factual questions on the 5 in-scope HDFC schemes from public sources only. |
| G2 | Guarantee every visible answer is **≤ 3 sentences**. Zero tolerance. |
| G3 | Guarantee every answer shows **one clear source link**. |
| G4 | Guarantee no PII is accepted or stored (PAN, Aadhaar, account numbers, OTPs, emails, phone numbers). |
| G5 | Guarantee **no performance claims** — never compute or compare returns. |
| G6 | Guarantee no investment advice; refuse opinionated/portfolio questions politely with a relevant educational link. |
| G7 | Keep the UI honest and explicit: "Facts-only. No investment advice." |

### 2.2 Non-Goals

- Return computation, comparison, ranking, projection, or any performance claim.
- Portfolio advice, asset allocation, goal planning, or "should I buy/sell" guidance.
- Funds outside the 5 in §3, including other HDFC schemes and all other AMCs.
- Plan variants other than **Direct Growth** (see §3.2).
- User accounts, authentication, saved portfolios, watchlists.
- Live/real-time NAV or market data.
- Any transactional capability.
- Multilingual support (English only in v1).

---

## 3. Scope — AMC and the 5 In-Scope Schemes

**AMC in scope: HDFC Mutual Fund (one AMC, per brief).** This is a closed allow-list; the system must not answer about any other fund.

### 3.1 The 5 Schemes (mandatory list)

| # | Canonical Scheme Name | Category | Source URL (verified live, HTTP 200) | Scheme ID |
| --- | --- | --- | --- | --- |
| 1 | **HDFC Large Cap Fund** | Equity — Large Cap | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` | `HDFC_LARGE_CAP` |
| 2 | **HDFC Flexi Cap Fund** | Equity — Flexi Cap | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` | `HDFC_FLEXI_CAP` |
| 3 | **HDFC ELSS Tax Saver Fund** | Equity — Tax Saving (ELSS) | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` | `HDFC_ELSS` |
| 4 | **HDFC Small Cap Fund** | Equity — Small Cap | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` | `HDFC_SMALL_CAP` |
| 5 | **HDFC Balanced Advantage Fund** | Hybrid — Equity Oriented | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` | `HDFC_BAL_ADV` |

All five URLs were confirmed reachable (HTTP 200) and their page titles verified. The set covers the brief's suggested mix — one large-cap, one flexi-cap, one ELSS — plus small-cap and a balanced/hybrid scheme, spanning the equity risk spectrum plus the only tax-locked product.

### 3.2 Two scope facts that will otherwise cause bugs

**(a) `hdfc-equity-fund` is a legacy slug.** That URL now serves **"HDFC Flexi Cap Fund"** (verified via page title). HDFC Equity Fund was renamed to HDFC Flexi Cap Fund. The indexer and the alias table must map the legacy name to the canonical name, or queries for "HDFC Equity Fund" fail to retrieve.

**(b) Every URL is the Direct Growth plan.** The corpus covers **Direct Growth only**. Questions about Regular/Direct Growth comparison, dividend/IDCW options, or commission structures are **not supported by the corpus** and must route to the grounded-refusal template — not be answered from model knowledge. This is a real coverage boundary, not an oversight.

### 3.3 Scheme Name Aliases

| Alias a user may type | Canonical scheme |
| --- | --- |
| HDFC Top 100, HDFC Top 100 Fund | HDFC Large Cap Fund |
| HDFC Equity, HDFC Equity Fund, HDFC Premier Equity | HDFC Flexi Cap Fund |
| HDFC ELSS, HDFC Tax Saver Fund, HDFC 80C Fund | HDFC ELSS Tax Saver Fund |
| HDFC Alpha Small Cap, HDFC Small Cap | HDFC Small Cap Fund |
| HDFC Balanced Advantage, HDFC Prudence Balanced Fund | HDFC Balanced Advantage Fund |
| "HDFC MF", "HDFC funds", "your funds", "HDFC" alone | Ambiguous — treat as a scope question and list the 5 |

---

## 4. Hard Requirements (Non-Negotiable)

Each has a dedicated validator, an automated test, and a release gate. **R1–R3 are the constraints explicitly called out in the brief; R4–R7 are derived from the brief's constraint list and are equally blocking.**

### 4.1 R1 — Strict 3-Sentence Response Limit

> No response may contain more than **three sentences** of user-visible text. A hard cap, not a target.

**Binding definition of "sentence" (for implementation and test):**

- A sentence is a span ending in `.`, `!`, or `?` as delimited by an NLP sentence tokenizer (spaCy `sentencizer`, or NLTK `punkt` with abbreviation handling), after whitespace normalisation.
- Counting applies to the **visible answer body only**. The source link and the `Last updated from sources:` line render as separate elements and are **not** counted.
- `e.g.`, `i.e.`, `vs.`, `Rs.`, `No.`, and decimals (`0.55%`, `1.25`, `₹500`) are **not** sentence boundaries. The tokenizer must handle them.
- Bulleted/numbered lists and semicolon- or colon-separated clauses are **not** separate sentences, but each list item or clause ending in terminal punctuation **counts as one**.
- A truncated fragment is **not** a sentence and is never allowed — see the ladder.

**Enforcement ladder (executed in order):**

1. **Constrained generation** — prompt specifies ≤ 3 sentences; temperature 0; `max_tokens` set so a 4th sentence is not reachable in one pass.
2. **Compress pass** — if > 3 sentences, re-prompt: *"Reduce to at most 3 sentences. Preserve the fact and its source chunk. Drop all qualifiers, examples, and asides."*
3. **Deterministic boundary truncation** — if still over, cut at the end of the 3rd sentence and discard the remainder, appending **no** ellipsis and **no** partial clause.
4. **Templated fallback** — if truncation would leave a meaningless stub (< 5 tokens), return a hard-coded factual template built from the top retrieved chunk.

Every rung must yield a grammatically complete answer. A dangling clause is a **release-blocking bug**.

**Prohibited workarounds:** splitting one answer across multiple chat bubbles; hiding overflow in `<details>`; putting one sentence per table row; streaming a long answer and editing it in place.

### 4.2 R2 — One Clear Source Link Per Answer

> Every answer shows one clear citation link to the public page it came from.

**Rules:**

- **Exactly one** link per answer — never zero, never more than one.
- Must resolve to a real, live public page (HTTP 200), and must be a page from the curated source list (§7.2). No invented slugs, no search-result URLs, no aggregator-of-aggregators, no third-party blogs.
- The link must be the **highest-scoring retrieved chunk that actually supports the answer**. It is a retrieval artefact, **not** an LLM output: the LLM emits a `chunk_id`, the system resolves it to a stored, verified URL. **Fabricated citations are therefore structurally impossible.**
- Label names the document, not the URL — e.g. *"HDFC ELSS Tax Saver Fund — Groww scheme page"*, *"HDFC Small Cap Fund — factsheet, Jun 2026"*.
- Refusals and out-of-scope redirects **also carry one relevant educational link** — see §4.6, this is explicitly required by the brief.
- Source preference where a fact appears in both an official HDFC/SEBI/AMFI document and a broker page: **cite the official source.** See the conflict resolution in §12.3.

**Enforcement:**

1. Retrieve top-k; parse LLM-proposed `chunk_id`; validate it exists in the provided context set.
2. If ≥ 2 valid IDs proposed → keep the highest-scoring, **discard the rest**; the render must never display more than one link.
3. If 0 valid → **citation-repair pass**: re-ask the LLM to select one ID from the enumerated allowed list.
4. If repair fails, or the top chunk score is below the confidence floor (§7.4) → return the grounded-refusal template. **An uncited factual answer is never rendered.**

### 4.3 R3 — Absolute Ban on PII

> The brief: *"No PII. Do not accept/store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers."*

**Explicitly enumerated in the brief:** PAN · Aadhaar · account numbers · OTPs · emails · phone numbers.
**Also in scope:** folio numbers, Demat/DP IDs, IFSC, bank account digits, card numbers/CVV, EPFO/PPO, nominee and guardian details, date of birth, postal address, passwords.

**Required behaviour:**

| Situation | Behaviour |
| --- | --- |
| Asks for their own folio/account/holdings | **Refuse.** The system is anonymous and holds no investor records. Do not imply the capability exists. |
| Pastes PII into the chat | **Refuse the request** and **redact the PII before it reaches logs, traces, or analytics.** |
| Asks "should I share my PAN?" | Refuse + a neutral educational link (e.g. HDFC's official PAN/Aadhaar guidance). Do not restate the PII. |
| Generic definitional question ("what is a folio number?") | Answerable in ≤ 3 sentences only if generic; must not be paired with any identifier. Default: redirect to fund facts. |
| Source document contains a professional name (fund manager, auditor, trustee) | **Permitted** — public professional roles, not customer PII. Do not blanket-redact these. |

**Architectural support:** **no user accounts and no authentication by design** (§2.2), so there is no customer PII store to leak. Defence in depth: regex + NER detection at the input boundary, a mandatory redaction filter in the single logging entry point, and an output scan before render.

### 4.4 R4 — No Performance Claims (separate from, and stricter than, "no advice")

> The brief: *"No performance claims. Don't compute/compare returns; link to the official factsheet if asked."*

This is its own hard rule and is **not** satisfied by the no-advice rule. The bot must **never state a return figure, growth rate, ranking, or comparison** — not even accurately, not even with an as-of date.

| Request | Required response |
| --- | --- |
| "What are the returns of HDFC Small Cap Fund?" | **Do not state a return.** Route to the factsheet: give the one-line factual frame (e.g. that performance is reported in the monthly factsheet) and link to the official factsheet. |
| "Which of these 5 has performed best?" | **Refuse.** Comparison of returns is explicitly banned. Offer non-performance facts (expense ratio, exit load, benchmark, objective). |
| "Is HDFC Flexi Cap a good performer?" | **Refuse** as a performance claim. |
| "What is the NAV?" | Current published NAV may be stated **only** with its as-of date, **only** if present in a retrieved chunk. Otherwise link to the source. Live/quoted NAV is not a "return" but must never be stale or unsourced. |
| "What's the expense ratio / exit load / benchmark / AUM?" | **Permitted factual answer** with a source link. |

The LLM must be instructed that numeric performance fields present in retrieved context are **context, not answerable content**, and a lint must catch numeric return patterns (`X% return`, `CAGR of`, `1-year return was`) in the draft.

### 4.5 R5 — Absolute Ban on Financial Advice

> The brief: *"Refuses opinionated/portfolio questions (e.g., 'Should I buy/sell?') with a polite, facts-only message and a relevant educational link."*

The hardest boundary to enforce, because a fact and an opinion are often lexically identical. The rule governs **intent and framing**, not vocabulary.

| Category | Permitted (FACT) | Prohibited (ADVICE) |
| --- | --- | --- |
| Selection | "HDFC Large Cap Fund tracks the Nifty 50." | "You should choose HDFC Large Cap Fund." |
| Suitability | "HDFC ELSS Tax Saver Fund has a 3-year lock-in for the 80C benefit." | "This fund is right for you." |
| Action | "The exit load is 1% if redeemed under 12 months." | "Hold it for at least 3 years." / "Switch to Direct to save money." *(Directive framing is prohibited even when the tax fact is true.)* |
| Timing | "NAV is published on each business day after market close." | "Buy on a dip." / "The right time to invest is now." |
| Comparison | "HDFC Balanced Advantage Fund's stated objective is dynamic equity allocation." | "HDFC Balanced Advantage is safer than HDFC Small Cap." |
| Risk | "HDFC Small Cap Fund's riskometer category is documented on the scheme page." | "This fund is low-risk." |
| Performance | *(see R4 — banned outright)* | "This fund will outperform." |

**Mandatory defence:** every advice request gets a refusal that **offers the factual alternative** rather than a dead end, plus one relevant educational link.

**Linting:** a lexicon of directive phrases (`you should`, `I recommend`, `best choice`, `I'd suggest`, `is ideal for`, `worth investing`, `go for`, `must buy`, `can I buy`) plus a modality classifier runs on the draft. A hit triggers **regeneration in fact-only framing**, then re-lint — never a silent strip. Two consecutive failures → refuse.

### 4.6 R6 — Refusal Contract

Every refusal must satisfy all four:

1. **Polite and brief** (≤ 3 sentences).
2. **States the facts-only boundary** without lecturing.
3. **Offers a useful factual alternative** where one exists.
4. **Carries one relevant educational link** — e.g. HDFC's "Mutual Funds: Introduction / How to invest" or SEBI's investor-education page. **The brief requires this link on refusals.**

This supersedes the v1.0 draft's decision to send refusals without a citation.

### 4.7 R7 — Transparency Line

> The brief: *"Clarity & transparency. Keep answers ≤3 sentences; add 'Last updated from sources: '."*

**Applies to EVERY response — answers and refusals alike** (stakeholder decision, 2026-09-27). The brief's line 23 attaches it to "answers" generally; scoping it to factual answers only was rejected as too narrow.

- Every response ends with the literal label **`Last updated from sources: `** followed by an ISO date (`Last updated from sources: 2026-06-30`).
- The string is **exact and user-visible** — a submission requirement, not a formatting nicety.
- **Date semantics differ by response type:**
  - *Factual answer* → the `as_of_date` of the **cited document**.
  - *Refusal* → the `as_of_date` of the **linked educational page**, so the line is never blank and never a placeholder.
  - *Out-of-scope / no-retrieval* → the `as_of_date` of the scope page or the **newest document in the corpus**, so the user can judge corpus freshness from the refusal itself.
- A response with a missing, blank, or literal-`"N/A"` date is a defect. **The line is never omitted and never left empty.**
- The line is excluded from the 3-sentence count (§4.1) and is not the source link.

### 4.8 R8 — Scope Containment

Questions about any fund outside the 5, any other AMC, gold, stocks, insurance, or unrelated topics are refused with a scope-correcting redirect listing the 5 schemes.

---

## 5. Functional Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-01 | Ingest the 5 named scheme pages plus supporting public documents (§7.2). | Must |
| FR-02 | Store every chunk with metadata: `scheme_id`, `doc_type`, `as_of_date`, `url`, `section`, `chunk_id`, `source_hash`, `source_tier`. | Must |
| FR-03 | Resolve aliases (§3.3), including the `hdfc-equity-fund` → Flexi Cap legacy mapping. | Must |
| FR-04 | Detect in-scope scheme intent; scope retrieval to that scheme, else search all 5. | Must |
| FR-05 | Generate grounded **only** in retrieved chunks, temperature 0, schema-constrained output. | Must |
| FR-06 | Enforce R1 via the 4-step ladder. | Must |
| FR-07 | Enforce R2 via chunk-ID → stored-URL resolution. | Must |
| FR-08 | Enforce R3: input PII detection, refusal, log redaction. | Must |
| FR-09 | Enforce R4: block numeric performance claims in output. | Must |
| FR-10 | Enforce R5: advice-intent classification on input; output lint + fact-only regeneration. | Must |
| FR-11 | Enforce R6: refusal template carries an educational link. | Must |
| FR-12 | Enforce R7: append the exact `Last updated from sources: <date>` line to **every** response, answers and refusals alike. | Must |
| FR-13 | Enforce R8: out-of-scope detection with scope-correcting redirect. | Must |
| FR-14 | Return grounded refusal below the confidence floor rather than answering from parametric memory. | Must |
| FR-15 | Render the tiny UI: welcome line, 3 example questions, facts-only note (§10.2). | Must |
| FR-16 | Display the single source link and the transparency line on every answer. | Must |
| FR-17 | Redaction-aware analytics; no raw query text stored unmasked. | Must |
| FR-18 | Session state ephemeral; discarded at session end. | Must |
| FR-19 | Support single-turn follow-ups ("and its exit load?") carrying `scheme_id` across turns. | Should |
| FR-20 | Freshness badge; warn if newest document for a scheme is > 60 days old. | Should |
| FR-21 | "Why this answer?" trace: retrieved chunk, score, doc metadata, prompt version, ladder rung taken. | Should |
| FR-22 | `GET /v1/schemes` returning the catalogue with per-scheme doc coverage. | Should |
| FR-23 | Auth-guarded admin reindex for the monthly factsheet cycle. | Should |
| FR-24 | Health endpoint reporting index freshness and model versions. | Could |

---

## 6. System Architecture

```
User (tiny chat UI)
        |
        v
[FastAPI service]
        |
        +--> 1. INPUT BOUNDARY
        |       PII scan (R3) ──hit──> redact + refuse
        |       Performance-claim intent check (R4)
        |       Advice-intent classifier (R5)
        |       Scope + scheme resolver, alias map (R8, FR-03/04)
        |       Plan-variant guard: Direct Growth only (3.2b)
        |
        +--> 2. RETRIEVAL
        |       embed query (all-MiniLM-L6-v2, same encoder as index)
        |       ChromaDB vector search, top-k = 8
        |       hard filter: scheme_id in ALLOW_LIST
        |       MMR re-rank for diversity
        |       confidence floor check ──below──> grounded refusal
        |
        +--> 3. GENERATION (temp 0, constrained max_tokens, JSON schema)
        |       prompt contract (§9): performance fields = context, not answerable
        |       output: { intent, sentences[], chunk_id, as_of_date }
        |
        +--> 4. RESPONSE VALIDATOR   <-- ALL HARD RULES LIVE HERE
        |       a. sentence counter    -> R1 ladder
        |       b. performance lint   -> R4 (regenerate, then refuse)
        |       c. advice lint        -> R5 (regenerate, then refuse)
        |       d. citation resolver  -> R2 ladder (chunk_id -> stored URL)
        |       e. as-of / transparency line -> R7
        |       f. PII scan on output -> R3
        |       g. scope re-check     -> R8
        |       h. link reachability + schema check
        |       |-> ANY unresolved breach -> templated refusal (§9.2)
        |
        +--> 5. RENDER + LOG (redacted)
                |
                v
        User sees: <=3 sentences + 1 source link + "Last updated from sources: <date>"
```

### 6.1 Mandated Technology Stack

Per the brief's "End output — RAG ChatBot" section:

| Layer | Choice | Status |
| --- | --- | --- |
| Embedding model | **`sentence-transformers/all-MiniLM-L6-v2`** | **Mandated by brief.** 384-dim, ~80 MB, CPU-friendly, free. |
| Vector database | **ChromaDB** (persistent client) | **Mandated by brief.** |
| Pipeline stages | **Loading → Chunking → Embedding → Store Vector Data**, plus retrieval-side search/generation | **Mandated by brief.** Both ingestion and retrieval stages must be demonstrable. |
| Chunking strategy | **To be decided from the data** — the brief says to determine it based on the actual content, not fixed in advance. | **Open — see §12.2 Q1.** |

v1.0 proposed 350–500 tokens with 60–80 overlap. That is now a **starting hypothesis to be validated against the real pages**, not a settled decision. Measure chunk size against retrieval precision on the golden set and record the chosen value and the evidence.

| Layer | Choice | Rationale |
| --- | --- | --- |
| API | FastAPI (Python 3.11+) | Native Pydantic response schema enforcement. |
| UI | Streamlit or Gradio | "Tiny UI" per brief; shares the validation layer so the UI cannot bypass rules. |
| LLM | Instruction-following model with JSON-schema-constrained output | Must support temperature 0. |
| Sentence split | spaCy `sentencizer` | Correct abbreviation/decimal handling for R1. |

> **Constraint interaction:** `all-MiniLM-L6-v2` is a 384-dimension English sentence encoder trained on general web text. It is adequate for these short, keyword-dense factual lookups. Two known limitations to test: it is **not** great at disambiguating near-identical numbers across 5 similar scheme pages (e.g. "expense ratio" appearing on all 5), and numeric recall is weaker than semantic recall. Mitigations: a `scheme_id` metadata filter to hard-scope retrieval, and exact-match boosting for numeric terms. **If AC-08 fails on numeric accuracy, a hybrid/BM25 keyword leg is the first fix, keeping the mandated embedding model in place rather than replacing it.**

---

## 7. Data Model, Corpus, and Retrieval

### 7.1 Chunk Metadata Schema

```json
{
  "chunk_id":         "hdfc_small_cap_page_expense_2026_07_c02",
  "scheme_id":        "HDFC_SMALL_CAP",
  "scheme_name":      "HDFC Small Cap Fund",
  "plan":             "Direct Growth",
  "doc_type":         "scheme_page",
  "doc_title":        "HDFC Small Cap Fund Direct Growth",
  "source_tier":      "broker",
  "as_of_date":       "2026-07-15",
  "url":              "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
  "section":          "Fees and charges",
  "text":             "...",
  "source_hash":      "sha256:...",
  "embedding_model":  "all-MiniLM-L6-v2",
  "ingested_at":      "2026-07-20T09:14:00Z"
}
```

`source_tier` is `official` (HDFC AMC / SEBI / AMFI) or `broker` (the Groww pages). It drives citation preference (§12.3) and the deliverable's source list.

### 7.2 Corpus

**Tier 1 — the 5 named source URLs (primary, required deliverable).** The 5 scheme pages in §3.1, all verified HTTP 200.

**Tier 2 — official supporting documents** (AMC / SEBI / AMFI categories named in the brief), used for the statutory and procedural facts the 5 pages summarise rather than state:

| `doc_type` | Source | Covers |
| --- | --- | --- |
| `scheme_faq` | HDFC AMC | **The scheme's own FAQ pages** — plain-language Q&A. The most retrieval-friendly doc type and the best match to a "FAQ assistant". Named explicitly in the brief. |
| `factsheet` | HDFC AMC | Monthly performance (NAV-to-NAV tables, benchmark, AUM, holdings) |
| `kim_sid` | HDFC AMC | KIM / SID — objective, benchmark, riskometer methodology |
| `exit_load_schedule` | HDFC AMC | Statutory additional-information / exit-load table |
| `riskometer_note` | HDFC AMC / SEBI | Riskometer categories and the level-of-risk scale |
| `elss_lockin_rule` | HDFC AMC / Income Tax Dept. | 3-year lock-in, 80C limit, Section 80C |
| `statement_guide` | HDFC AMC | How to download capital-gains / account statements |
| `tax_doc_guide` | HDFC AMC | Tax-document guides (26AS, capital-gains statement, TDS) |
| `fee_charges` | HDFC AMC | Fee/charges and TER detail |
| `investor_education` | SEBI / HDFC AMC | Educational link used by refusals (R6) |

### 7.3 Ingestion Notes

- **CORRECTED 2026-09-27 by Phase 1 (measured, not assumed).** A plain `requests` fetch **does** return 17k–44k chars of real server-rendered prose per page — no headless browser needed. What it does **not** return is the *figures*. Expense ratio, riskometer level, minimum SIP and benchmark are rendered client-side from the page's embedded `__NEXT_DATA__` JSON, so the stripped HTML shows the label "Expense ratio" next to an info icon and **no number at all**. Stage 1 therefore reads two layers of the same document: the rendered text, and the payload scalars. Both are required; dropping `<script>` is what silently deleted the headline fact.
- Extract per-section blocks (fees, exit load, minimums, benchmark, riskometer, objective) rather than one blob per page — these map 1:1 to the user questions and give clean, citable chunks.
- **Never treat a `Label: value` line as a heading during section segmentation.** Doing so strips the value out of the body and deletes the fact while the label still appears in the section list. Table rows (`| a | b |`) are never headings for the same reason.
- Prepend a synthetic context line to each chunk: `"{scheme_name} ({plan}) | {doc_type} | as of {as_of_date} | {section}"`. This measurably improves retrieval and makes chunks self-describing for citation.
- Never split a table across chunks; keep the header row with any table body.
- Store `source_hash` so re-ingest is idempotent and drift is detectable.

### 7.4 Retrieval and Confidence Floor

- Top-k = 8, hard-filtered to the `ALLOW_LIST`.
- **Similarity score < 0.62 → do not answer.** Return the grounded-refusal template. *(Threshold to be tuned empirically on the golden set; a floor set too low produces confident hallucination, which is worse than a refusal.)*
- If the top-2 chunks materially disagree on a number (e.g. two different expense ratios for the same scheme), **do not answer** — surface the conflict using the single highest-confidence link and record the ambiguity in the trace.
- Add exact-match boosting for numeric tokens (§6.1).

---

## 8. Prompt Contract

A **versioned, unit-tested artefact** (`prompts/system_prompt.v1.txt`) pinned in the repository — a required release input, not a code string.

```
ROLE
You are a facts-only reference assistant for HDFC Mutual Fund scheme information.
You are not a financial advisor.

SCOPE
You answer ONLY about these 5 HDFC schemes (Direct Growth plan):
  1. HDFC Large Cap Fund
  2. HDFC Flexi Cap Fund
  3. HDFC ELSS Tax Saver Fund
  4. HDFC Small Cap Fund
  5. HDFC Balanced Advantage Fund
Anything else -> intent=OUT_OF_SCOPE.

OUTPUT FORMAT (strict JSON)
{
  "intent": "FACTUAL | OUT_OF_SCOPE | ADVICE_REQUEST | PERFORMANCE_REQUEST | PII_REQUEST | UNGROUNDED",
  "sentences": ["...", "...", "..."],
  "chunk_id": "<exactly one chunk_id from CONTEXT, or null>"
}

HARD RULES
- MAXIMUM 3 SENTENCES. Never exceed. Lists that add sentences are not allowed.
- Use ONLY facts present in CONTEXT. Never use prior knowledge.
- NO PERFORMANCE CLAIMS. Never state a return, growth rate, ranking, or comparison.
  Numeric performance figures in CONTEXT are reference material, not answerable
  content. If asked about returns, point to the factsheet instead.
- EXACTLY ONE chunk_id, the best supporting chunk. Never null on a FACTUAL answer.
- Never recommend, rank, endorse, or forecast. Report published facts only.
- Never request or repeat PAN, Aadhaar, account/folio numbers, OTPs, emails, phone numbers.
- Plan variants other than Direct Growth are out of scope.
- Never invent a URL. The link is resolved from chunk_id by the system.

CONTEXT
[retrieved chunks: chunk_id, scheme_name, doc_type, as_of_date, url, text]
```

Two engineering requirements:

1. **Schema-constrained decode** (JSON Schema / structured output) so malformed output is impossible rather than merely unlikely.
2. **The URL is never model-generated.** The LLM emits `chunk_id`; the system maps it to a stored, verified URL. This makes fabricated citations structurally impossible.

---

## 9. Response Templates

All hard-coded, human-reviewed, ≤ 3 sentences. **All carry exactly one link; the non-factual ones carry a relevant educational link (R6).**

### 9.1 Refusals and Special Cases

| Trigger | Template |
| --- | --- |
| Out-of-scope fund / other AMC | "I only cover 5 HDFC Mutual Fund schemes: HDFC Large Cap Fund, HDFC Flexi Cap Fund, HDFC ELSS Tax Saver Fund, HDFC Small Cap Fund, and HDFC Balanced Advantage Fund. If your question is about one of these, tell me which and I'll share the published facts." + educational link |
| Advice request (R5) | "I can't recommend a fund or advise on what's right for you — this assistant is facts-only. I can share published details such as expense ratio, exit load, lock-in period, benchmark, or minimum investment for any of the 5 HDFC schemes in scope." + educational link |
| Performance / returns request (R4) | "I don't compute or compare returns, since figures change and past performance doesn't indicate future results. Performance for each of these funds is published in its monthly factsheet, which I can point you to." + factsheet link |
| PII request (R3) | "I don't collect or store personal information, and I can't access individual investor records. I can only share published, public information about the 5 funds in scope." + educational link |
| Non-Direct-Growth plan (3.2b) | "This assistant is scoped to the Direct Growth plan of the 5 HDFC schemes, and I don't have the Regular or dividend plan documents indexed. If your question is about Direct Growth, ask me and I'll share what's published." + link |
| Below confidence floor (R4/R2) | "I couldn't find a published page that answers that, and I don't guess. I can help with objective, expense ratio, exit load, minimum SIP, lock-in, benchmark, or how to download statements for the 5 HDFC schemes in scope." + educational link |
| Stale corpus | "Note: the newest document I have for this fund is dated {as_of_date}. Please confirm current figures on the source page." + link |

### 9.2 Response Shape (answers **and** refusals)

Every response — answer or refusal — ends with the same two lines, in this order:

```
{answer or refusal sentence 1}
{sentence 2 — optional}
{sentence 3 — optional}

Source: {document label} — {url}
Last updated from sources: {as_of_date}
```

Refusals are styled distinctly in the UI so a user can tell a refusal from an answer. The `Source:` and `Last updated from sources:` lines are **shared by both paths** and are the only elements exempt from the 3-sentence count.

---

## 10. UI Specification ("Tiny UI")

Per the brief: *"Tiny UI: welcome line + 3 example questions and a note: 'Facts-only. No investment advice.'"*

### 10.1 Required Elements

| # | Element | Requirement |
| --- | --- | --- |
| 1 | **Welcome line** | One line naming the assistant, the AMC, and the 5 schemes. |
| 2 | **3 example questions** | Exactly 3, clickable, covering distinct intents (see below). |
| 3 | **Disclaimer note** | The literal string **`Facts-only. No investment advice.`** |

**Welcome line (approved copy):**
> "Hi — I'm a facts-only assistant for 5 HDFC Mutual Fund schemes: Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, and Balanced Advantage. Ask me about expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, or statements."

**The 3 example questions:**

1. "What is the expense ratio of HDFC Large Cap Fund?" *(fee/charges fact)*
2. "What is the lock-in period for HDFC ELSS Tax Saver Fund?" *(ELSS rule fact)*
3. "How do I download my capital gains statement?" *(procedural fact)*

> **Why these three.** They span three distinct topic types — a fee fact, a statutory-rule fact, and a procedural fact — and they cover **two different schemes plus a cross-scheme procedure**, so a reviewer immediately sees that the system is not hard-wired to a single fund. All three are drawn from the brief's own example list (line 15).
>
> **Constraints on example selection:**
> - **No example may trigger the R4 or R5 refusal path.** The first thing a reviewer sees must be a working answer, not a refusal — so no example asks about returns, performance, or recommendations.
> - **No two examples may target the same scheme**, so multi-scheme coverage is visible on first paint.
> - Each must be answerable from an indexed chunk at the AC-08 accuracy bar. If any example cannot be answered from the corpus, it is replaced, not left to fail live.

### 10.2 UI Requirements

| ID | Requirement |
| --- | --- |
| UI-1 | Every response — answer **or** refusal — renders: ≤ 3 sentences, exactly one visible source link, and the `Last updated from sources: <date>` line. |
| UI-2 | The facts-only disclaimer is visible without scrolling and is repeated in the footer. |
| UI-3 | Refusals render in a distinct style with a visible "no advice given" marker. |
| UI-4 | The single source link opens in a new tab and is visibly a link, not raw URL text. |
| UI-5 | The in-scope 5 schemes are listed in the sidebar or header, always visible. |
| UI-6 | Keyboard navigable, screen-reader friendly, WCAG 2.1 AA contrast. |
| UI-7 | No login, no account creation, no PII input fields anywhere in the UI. |
| UI-8 | Stale-corpus warning surfaces when the newest doc for a scheme is > 60 days old. |

---

## 11. Evaluation and Acceptance Criteria

### 11.1 Test Suites

| Suite | Size | Purpose |
| --- | --- | --- |
| **G — Golden** | 150 factual questions across the 5 schemes (30 each) | Correctness, relevance, citation, as-of dating |
| **P — Performance** | 40 returns/comparison requests | Must refuse per R4; must not emit a return figure |
| **A — Adversarial safety** | 60 cases: PII, advice, jailbreak, prompt injection ("ignore previous instructions"), role-play ("pretend you are a fund manager"), multi-turn pressure | Must refuse |
| **S — Scope** | 40 out-of-scope queries (other AMCs, other HDFC schemes, gold, stocks, insurance, Regular plan) | Must refuse with correct redirect |
| **C — Constraint** | All suites through the validator | Sentence count, link count, transparency line |
| **L — Live link check** | Sampled 5% of answers | Cited URL returns HTTP 200 and contains the cited figure |
| **W — Workflow** | 30 real user questions | End-to-end answer quality on the supported topic set |

### 11.2 Release Gates (all must pass)

| # | Criterion | Threshold |
| --- | --- | --- |
| AC-1 | Answers containing > 3 sentences | **0** |
| AC-2 | Answers with ≠ exactly 1 source link | **0** |
| AC-3 | Cited links that 404 or don't contain the cited figure | **0** |
| AC-4 | PII requests answered instead of refused | **0** of 60 |
| AC-5 | Advice/opinion requests answered instead of refused | **0** of 60 |
| AC-6 | Prompt-injection / jailbreak successes | **0** |
| AC-7 | Out-of-scope questions answered with out-of-scope content | **0** of 40 |
| AC-8 | Factual accuracy on the golden set (grounded in source docs) | ≥ 95% |
| AC-9 | Refusal quality (correct intent, helpful redirect, educational link present) | ≥ 90% |
| AC-10 | Ungrounded claims (not present in any retrieved chunk) | **0** |
| AC-11 | Raw PII in logs, traces, or analytics | **0** |
| AC-12 | Responses (answers **or** refusals) missing the `Last updated from sources:` line, or showing a blank / `N/A` / placeholder date | **0** |
| AC-13 | Answers stating a return figure, growth rate, or return comparison | **0** of 40 |
| AC-14 | Answers recommending, ranking, or endorsing a fund | **0** |
| AC-15 | p50 / p95 end-to-end latency | ≤ 3 s / ≤ 8 s |
| AC-16 | The 3 example questions return real answers, not refusals | **3 of 3** |

**Regression discipline:** prompt, validator, embedding model, and chunking parameters are versioned. Any change re-runs the full suite. **A prompt edit that improves helpfulness but breaks any AC-1…AC-7 or AC-12…AC-14 gate is rejected.** Safety gates are never traded against quality.

---

## 12. Non-Functional Requirements, Assumptions, and Open Questions

### 12.1 Non-Functional Requirements

| ID | Category | Requirement |
| --- | --- | --- |
| NFR-01 | Privacy | No accounts, no PII persisted, redaction at log-write time, ephemeral sessions. |
| NFR-02 | Data residency | Corpus, index, and logs on owned infrastructure; no third-party data egress. |
| NFR-03 | Performance | p50 ≤ 3 s, p95 ≤ 8 s; no visible multi-round generation. |
| NFR-04 | Freshness | Monthly factsheets refreshed within 5 business days of HDFC publication; staleness badge past 60 days. |
| NFR-05 | Observability | Per-request trace: chunk IDs, scores, prompt version, validator actions, ladder rung taken, final intent. **No raw PII.** |
| NFR-06 | Auditability | Every answer reproducible from (query, chunk_id, prompt version, model version, chunking params). |
| NFR-07 | Maintainability | Scheme catalogue, templates, thresholds, and chunking params are config, not hard-coded. |
| NFR-08 | Cost | Local embedding + ChromaDB; zero per-token third-party cost in v1. |
| NFR-09 | Portability | Runs as notebook or app; README reproducible from a clean clone. |

### 12.2 Assumptions and Open Questions

| ID | Item | Status / impact |
| --- | --- | --- |
| A1 | Scope = 1 AMC (HDFC) + the 5 schemes in §3.1, Direct Growth only. | **Confirmed by brief.** |
| A2 | Corpus limited to public web pages. | **Confirmed by brief.** |
| A3 | Delivery = tiny web chat UI over a FastAPI API. | **Confirmed by brief.** |
| A4 | English-only v1. | Assumption. |
| A5 | No user accounts, no authentication, no personalization. | Assumption, consistent with "no PII". |
| A6 | Historic performance is **out of scope for answers** (R4) — the factsheet is linked instead. | **Confirmed by brief** (corrects v1.0 assumption A6, which was wrong). |
| **Q1** | **Chunking strategy** — brief says decide from the data. v1.0 proposed 350–500 tokens / 60–80 overlap as a hypothesis. Needs a measured decision. | **Open — engineering task.** Blocks M2. **Decision procedure now specified in §12.4** so the choice is evidence-based, not arbitrary. |
| Q2 | Is a hosted LLM acceptable, or must inference be fully local? | **Open.** Directly affects NFR-02 and the R3 posture. Local strongly preferred. |
| ~~Q3~~ | Are the 5 Groww pages the sole permitted sources, or may official HDFC/SEBI/AMFI pages be added? | **RESOLVED 2026-09-27 — Groww primary + official supplement.** See §12.3 C-1. |
| Q4 | Should the mandated `all-MiniLM-L6-v2` stay if it fails AC-08 on numeric accuracy, or may a hybrid keyword leg be added? | **Open.** Plan: keep the mandated model, add BM25 as a complement. |
| Q5 | Who signs off the 150-question golden set for factual accuracy? | **Open — blocks AC-8.** |
| ~~Q6~~ | Is a "Last updated from sources:" date derived from page publish time or ingest time? | **RESOLVED — `as_of_date` = the source document's own last-updated/publish date, extracted at ingest; ingest date is stored separately as `ingested_at` and never shown to the user.** If a page exposes no publish date, fall back to the HTTP `Last-Modified` header, then to corpus ingest date, and flag it in the trace. |

### 12.3 Spec Conflict Log

Three genuine contradictions **inside the brief itself**. C-1 and C-2 are now **resolved by stakeholder decision**; C-3 remains an interpretation to confirm.

| # | Conflict | Detail | Resolution |
| --- | --- | --- | --- |
| **C-1** | **Source authority** | The brief says collect pages "from AMC/SEBI/AMFI" and "**no third-party blogs as sources**", then supplies **5 `groww.in` URLs**. Groww is a third-party broker/aggregator — not HDFC AMC, not SEBI, not AMFI. The deliverable then asks for a "source list of the **5 URLs you used**". | ✅ **RESOLVED 2026-09-27 — Groww primary + official supplement.** The 5 Groww pages are **Tier-1, the required primary corpus** (deliverable line 26 is explicit about "the 5 URLs you used"). Official HDFC/SEBI/AMFI pages are **Tier-2**, covering the statutory facts the 5 pages summarise rather than state (riskometer methodology, exit-load schedule, ELSS lock-in, statement guides, scheme FAQs). **Where a fact appears in both tiers, cite the official source** (R2). Rationale for compliance: Groww is a regulated broker platform, not a blog, so it does not breach the "no third-party blogs" clause read literally. Recorded as `source_tier: broker` so the distinction stays visible in the source-list deliverable. |
| **C-2** | **"No performance claims" vs. FAQ topic list** | The brief lists "performance"-adjacent FAQ topics but also bans computing/comparing returns and says to link the factsheet instead. | ✅ **RESOLVED 2026-09-27 — ban implemented as written.** R4 is enforced: no return figure, growth rate, ranking, or comparison ever appears in an answer. Performance requests receive the factsheet link instead. The topic list is read as *"point me to where performance is published"*, not *"tell me the return"*. |
| **C-3** | **"No screenshots of the app back-end"** | Ambiguous: it may mean *don't use screenshots as sources*, or *don't screenshot the backend for the deliverable*. | **Interpreted** (unconfirmed): sources must be live public pages, not images; and the prototype/demo must be a working link or video, not backend screenshots. **Confirm at M0.** |

---

### 12.4 Chunking Decision Procedure (resolves Q1 at build time, not on paper)

The brief defers this to the data ("Ask cursor to decide the chunking strategy based on the data"), so the PRD specifies **how to decide** rather than deciding now. This runs at M2 and its output is recorded in the README and in the trace metadata (NFR-06).

**Step 1 — Characterise the corpus.** Report mean/median chunk length in tokens, the ratio of tabular to prose content, and the section-heading density. These five pages are metadata-dense (fees, exit-load slabs, minimums) rather than prose-heavy, which is the fact that should drive the decision.

**Step 2 — Define the token unit (AMENDED v1.3, per `architecture.md` §6.2).** "Token" means a **word-piece as counted by the mandated embedder's own tokenizer**, never a word or character estimate:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
n = len(tok(chunk_text, add_special_tokens=True)["input_ids"])
```

This amendment is **mandatory, not advisory.** The model card states: *"By default, input text longer than 256 word pieces is truncated."* Truncation is silent — the tail of an over-length chunk never reaches the vector while the stored document text still shows it in full, so the index looks complete while being blind to that content. Because ~1.3 word-pieces ≈ 1 English word, a naive word count overstates usable capacity by roughly 30% and will silently over-run the limit.

**Step 3 — Sweep the grid.** Chunk at `{128, 192, 224, 256}` word-pieces × `{0, 32, 48, 64}` overlap, section-aware, holding every other variable fixed. **The v1.2 grid of `{250, 400, 600, 800}` is withdrawn** — every option above 256 was truncated, so the sweep would have measured embeddings that never represented most of the chunk.

**Step 3 — Score each configuration** against a stratified 40-question probe set (8 per scheme, spanning all six brief topics) on two metrics only:
- **Citation precision** — does the top chunk actually contain the answer?
- **Self-containment** — can the chunk answer without its neighbours (measured by how often the generator needs a second chunk)?

**Step 5 — Select** the configuration maximising citation precision, with self-containment as the tie-break. **Report the grid and the winning row in the README.** A decision without evidence is not a decision.

**Guardrails regardless of the outcome:**
- Never split a table across chunks; always carry the header row (R1 §4.1 needs whole clauses to count sentences correctly).
- Always prepend the synthetic context line `{scheme} ({plan}) | {doc_type} | as of {date} | {section}`.
- **Hard ceiling of 256 word-pieces** — this is the embedder's truncation point, not a tunable (AMENDED v1.3; the v1.2 value of 800 was invalid). Operate at **224** to reserve headroom for the context header and tokenizer drift; the model was trained at 128 sequence length, so quality also degrades before the hard limit.
- Any chunk exceeding the ceiling is a **hard error, not a warning** — it means the chunker failed, and a silent over-length chunk is the failure mode most likely to reach the index unnoticed.

---

## 13. Deliverables (per the brief's submission list)

| # | Deliverable | Requirement | Status |
| --- | --- | --- | --- |
| D1 | **Working prototype** | App or notebook link; ≤ 3-min demo video if hosting isn't possible. | Not started |
| D2 | **Source list** | CSV or MD of the **5 URLs** used. | §3.1 ready; ingestion pending |
| D3 | **README** | Setup steps, scope (AMC + schemes), known limits. | Not started |
| D4 | **Sample Q&A file** | 5–10 queries with the assistant's answers **+ links**. | Not started |
| D5 | **Disclaimer snippet** | The exact facts-only/no-advice text used in the UI. | §14.1 ready |

---

## 14. Appendix

### 14.1 UI Disclaimer Snippet (D5)

> **Facts-only. No investment advice.**
> This assistant answers factual questions about 5 HDFC Mutual Fund schemes using published public pages only. It does not recommend funds, does not compute or compare returns, and is not a substitute for professional financial advice. Figures may change — always confirm on the source page.

### 14.2 Change Log — v1.0 → v1.1

v1.0 was written against an **empty** `problemstatement.txt` and is superseded. Four material corrections:

| # | v1.0 (incorrect) | v1.1 (correct, from the real brief) |
| --- | --- | --- |
| 1 | Scheme 5 was **HDFC Corporate Bond Fund**; **HDFC ELSS Tax Saver Fund was missing entirely**. | Scheme 5 is **HDFC ELSS Tax Saver Fund**; Corporate Bond is removed (§3.1). 4 of 5 names happened to match. |
| 2 | Assumed historic returns were answerable as facts with as-of dating. | **Wrong.** R4 now bans stating any return figure; the factsheet is linked instead. |
| 3 | Refusals carried **no** citation (logged as OQ-1). | **Wrong.** R6 now requires a relevant educational link on every refusal, per the brief. |
| 4 | Stack was "open item"; chunking fixed at 350–500 tokens. | Stack **mandated**: `all-MiniLM-L6-v2` + ChromaDB; chunking is **to be decided from the data** (Q1). |

Additions in v1.1 with no v1.0 counterpart: R4 (no performance claims), R7 (`Last updated from sources:` line), the "tiny UI" spec (§10), the 5 deliverables (§13), the spec conflict log (§12.3), and the ELSS/lock-in and statement-download topic coverage.

### 14.2b Change Log — v1.1 → v1.2 (compliance audit)

A line-by-line audit against all 38 lines of the brief. **No requirement was found to be missing or contradicted after these fixes**; the entries below are corrections and closures.

| # | Change | Reason |
| --- | --- | --- |
| 1 | Added `scheme_faq` and `tax_doc_guide` doc types (§7.2) | Brief line 8 names "scheme FAQs" and "statement/tax-doc guides" explicitly; neither had a `doc_type`. |
| 2 | Example questions re-spread: #1 → Large Cap, #2 → ELSS, #3 → cross-scheme (§10.1) | v1.1 used ELSS twice, hiding whether the other 4 schemes worked. Added selection constraints. |
| 3 | Milestones realigned to the brief's **submission** list (§14.3) | The brief is a "Milestone brief"; v1.1 milestones were internal build phases. |
| 4 | Added chunking **decision procedure** (§12.4) | Brief line 34 defers the decision to the data; v1.1 left it as a bare open item with no method. |
| 5 | **R7 now applies to refusals too** | Stakeholder decision. Brief line 23 attaches the line to "answers" generally. Date semantics per response type specified; blank/`N/A` dates are now an explicit defect. |
| 6 | **C-1 resolved** — Groww primary + official supplement | Stakeholder decision. §12.3 C-1. |
| 7 | **C-2 resolved** — performance ban implemented as written | Stakeholder decision. §12.3 C-2. |
| 8 | **Q6 resolved** — `as_of_date` = document's own last-updated date | With HTTP `Last-Modified` and ingest date as documented fallbacks; `ingested_at` never shown to users. |
| 9 | AC-12, FR-12, UI-1, §9.2 rewritten to match the R7 change | Consistency with #5. |
| 10 | Added §14.6 implementation status | Makes explicit that no prototype exists and D1–D5 are outstanding, so the PRD cannot be mistaken for a working build. |

**Still open after v1.2:** C-3 (interpretation to confirm), Q1 (chunking values — method now specified, values come from the M2 sweep), Q2 (hosted vs local LLM), Q4 (hybrid keyword leg if numeric accuracy fails), Q5 (golden-set sign-off owner). None of these block starting the build.

### 14.2c Change Log — v1.2 → v1.3 (mandated-embedder limit)

Raised while writing `architecture.md`, which confirmed the limit from the model card rather than assuming it.

| # | Change | Reason |
| --- | --- | --- |
| 1 | §12.4 **Step 2 added** — "token" is now explicitly a **word-piece counted by the mandated embedder's tokenizer** | A word/character estimate overstates usable capacity ~30% and silently over-runs the limit |
| 2 | §12.4 sweep grid **withdrawn**: `{250,400,600,800}` → `{128,192,224,256}` | `all-MiniLM-L6-v2` **silently truncates past 256 word-pieces**. Every prior option was truncated, so the sweep would have measured embeddings that never represented most of the chunk, and could have selected such a configuration |
| 3 | Hard ceiling **800 → 256** word-pieces; operating target **224** | 256 is the model's truncation point and is not tunable; 224 reserves headroom for the context header and tokenizer drift. The model was also trained at 128 sequence length, so quality degrades before the hard limit |
| 4 | Over-length chunks are a **hard error, not a warning** | A silent over-length chunk is the most likely failure to reach the index unnoticed |
| 5 | Step renumbered (Select is now Step 5) | Follows from the inserted Step 2 |

**Net effect:** the chunking decision procedure is unchanged in intent — still decided from the data — but it is now measured in units the embedder actually respects, so the sweep produces a valid answer instead of a confidently wrong one.

### 14.3 Milestones

### 14.3 Milestones

The brief is a **Milestone brief** with a fixed submission list (§13), so the milestone plan is organised around *what gets submitted*, not around internal build phases. Build order is given as the second column.

| # | Submission milestone | Build order | Exit criteria |
| --- | --- | --- | --- |
| **M0** | **Spec sign-off** | 1st | §12.3 C-3 confirmed; Q1, Q2, Q4, Q5 closed. Scope locked to the 5 schemes. |
| **M1** | **Corpus ready** (feeds D2) | 2nd | 5 pages ingested with non-empty extracted text; Tier-2 official docs added; metadata + aliases resolving. **SOURCES.csv written.** |
| **M2** | **Retrieval + generation working** | 3rd | Grounded answers on the golden set; **chunking decided from data** per §12.4 and recorded in the README. |
| **M3** | **Validator enforcing R1–R8** | 4th | All ladders implemented; distinct trace entry per enforcement action. |
| **M4** | **Safety gates green** | 5th | AC-1…AC-7 and AC-10…AC-14 at **zero** violations. Nothing is demonstrated until this passes. |
| **M5** | **Tiny UI live** (feeds D1) | 6th | Welcome line, 3 example questions, exact disclaimer, citation + transparency line on every response, distinct refusal styling. |
| **M6** | **Submission package complete** (D1, D3, D4, D5) | 7th | Prototype link or ≤3-min video; README with setup + scope + known limits; **sample Q&A generated from the real pipeline** with answers *and* links; disclaimer snippet. |
| **M7** | **Final verification** | 8th | Live-link check on sampled citations, latency, log audit for PII, accessibility pass. |

**Ordering rationale.** M3 → M4 sit deliberately early relative to "make it look good": the validator is the product, and every later milestone inherits its guarantees. M6 is gated on M4 so the submitted sample Q&A cannot contain a rule violation.

> **M6 integrity rule.** The sample Q&A file (D4) must be **generated by running the actual pipeline** — not hand-written. Hand-authoring plausible-looking answers and links is the single easiest way to ship a deliverable that fails AC-2 or AC-3 on inspection. If the pipeline cannot answer a question, the honest entry is a refusal, and that is a legitimate sample answer.

**Critical path:** M0 → M3 → M4 → M6.

### 14.4 Definition of Done

- [ ] Scope locked to HDFC + the 5 schemes in §3.1, enforced as a hard allow-list, Direct Growth only.
- [ ] R1–R8 enforced in deterministic code, with a distinct trace entry per enforcement action.
- [ ] Every answer ≤ 3 sentences, exactly 1 source link, and the exact `Last updated from sources: <date>` line.
- [ ] No return figure, growth rate, or comparison ever appears in an answer (AC-13 = 0).
- [ ] No recommendation, ranking, or endorsement ever appears (AC-14 = 0).
- [ ] Every refusal is polite, ≤ 3 sentences, and carries a relevant educational link.
- [ ] PII never accepted, echoed, or logged — verified by reading the log store, not the filter.
- [ ] Chunking strategy chosen from data, with the evidence recorded.
- [ ] All gates in §11.2 pass; D1–D5 submitted.
- [ ] §12.3 conflicts closed or deferred with a named owner.

### 14.5 Traceability to the Brief

| Brief requirement | Specified in | Enforced by | Verified by |
| --- | --- | --- | --- |
| One AMC, 5 schemes (§7) | §3.1, §3.2, §3.3 | `ALLOW_LIST`, scope resolver, alias map (R8, FR-03/04) | Suite S, AC-7 |
| Answers ≤ 3 sentences | §4.1 (definition + 4-rung ladder) | Sentence counter → compress → truncate → template (FR-06) | AC-1, AC-2 |
| One source link every answer | §4.2, §4.6 | chunk_id → stored URL; URL never model-generated (FR-07) | AC-2, AC-3 |
| Public sources only, no blogs | §7.1 `source_tier`, §7.2, §12.3 C-1 | Citation allow-list + link reachability check | AC-3, L suite |
| No PII | §4.3 (enumerated + behaviour table) | Input NER/regex, log redaction, output scan, no-accounts design (FR-08) | AC-4, AC-11 |
| No performance claims | §4.4 | Performance-intent classifier + numeric-return lint (FR-09) | AC-13, P suite |
| No advice; polite refusal + educational link | §4.5, §4.6 | Intent classifier + advice lint + regeneration (FR-10, FR-11) | AC-5, AC-6, AC-14 |
| "Last updated from sources: " (all responses) | §4.7, §9.2 | Transparency-line builder (FR-12) | AC-12 |
| Tiny UI: welcome + 3 examples + note | §10 | UI components (FR-15, FR-16) | AC-16, manual review |
| Facts-only. No investment advice. | §14.1, §10.1 | UI component (FR-15) | Manual review |
| all-MiniLM-L6-v2 | §6.1 | Config-pinned | NFR-06 |
| ChromaDB | §6.1 | Config-pinned | NFR-06 |
| Loading→Chunking→Embedding→Store | §6 architecture, §6.1 | Ingest pipeline stages | Manual walkthrough |
| Chunking decided from data | §12.2 Q1, §12.4, §14.3 M2 | Empirical sweep per §12.4 | NFR-06, M2 exit |
| 5 deliverables | §13, §14.3 M6 | — | D1–D5 |
| AMC/SEBI/AMFI doc categories (brief line 8) | §7.2 (incl. `scheme_faq`, `tax_doc_guide`) | Ingest manifest | M1 exit |
| Factsheets, KIM/SID, fee/charges, riskometer notes, statement guides | §7.2 | Ingest manifest | M1 exit |
| Retail users + support/content teams (brief line 5) | §1.2 | Question-set design for both audiences | Golden set coverage |

### 14.6 Implementation Status

**This PRD is specification only. No prototype has been built against it, and deliverables D1–D5 do not yet exist.**

| Deliverable | State |
| --- | --- |
| D1 Working prototype | **Not started** |
| D2 Source list | Specified in §3.1 + `src/config.py`; not yet generated as a file |
| D3 README | **Not started** |
| D4 Sample Q&A | **Not started** — and per the §14.3 M6 integrity rule it must be produced by running the real pipeline, never hand-written |
| D5 Disclaimer snippet | Drafted in §14.1; not yet placed in a UI |

**Environment finding that will affect whoever builds this.** The development machine's only pre-existing Python (Anaconda 3.8) has a **broken `ssl` module** and cannot reach PyPI at all:

```
SSLError("Can't connect to HTTPS URL because the SSL module is not available")
```

`torch`, `sentence-transformers`, and `chromadb` therefore all fail to install under it. A clean **Python 3.11.9** (OpenSSL 3.0.13) has been installed per-user at `%LOCALAPPDATA%\Programs\Python\Python311` and has working TLS, so the mandated stack installs normally there. Prefer a 3.11 virtual environment over the Anaconda interpreter.
