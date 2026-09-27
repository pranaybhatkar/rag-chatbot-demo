# Implementation Guide — Mutual Fund FAQ RAG Chatbot

| Field | Value |
| --- | --- |
| Document ID | IMPL-MF-RAG-001 |
| Version | 1.0 |
| Status | Ready to build |
| Last updated | 2026-09-27 |
| Implements | `PRD.md` v1.3 (requirements) + `architecture.md` v1.0 (design) |
| Mandate | Brief lines 30–36 — all RAG stages, ingestion **and** retrieval |

> **Read this first.** `PRD.md` says *what* must be true. `architecture.md` says *how the pipeline is shaped*. This document is the **build order** — which files to write, in what sequence, and how to know each phase is genuinely done rather than merely finished.
>
> **Do not skip Phase 4.** Guardrails are not a later hardening pass. Phase 4 builds the input boundary that every later request passes through, and it must exist before the UI is wired, or you will debug an unfiltered pipeline through a chat interface.

---

## 0. Before You Start

### 0.1 Environment (mandatory — do not skip)

The Anaconda 3.8 interpreter on this machine has a **broken `ssl` module** and cannot reach PyPI:

```
SSLError("Can't connect to HTTPS URL because the SSL module is not available")
```

`torch`, `sentence-transformers`, and `chromadb` all fail to install under it. A working **Python 3.11.9** is installed at `%LOCALAPPDATA%\Programs\Python\Python311` (OpenSSL 3.0.13).

```powershell
$env:PY = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $env:PY -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import chromadb, sentence_transformers; print('ok')"
```

**Pin exact versions.** `chromadb` and `sentence-transformers` both change embedding-function APIs between minor versions; an unpinned build is the most common cause of a "works on my machine" failure two days from the demo.

### 0.2 `requirements.txt`

```
chromadb==0.5.23
sentence-transformers==3.3.1
transformers==4.48.3
torch==2.6.0+cpu          # via --index-url https://download.pytorch.org/whl/cpu
streamlit==1.42.2
beautifulsoup4==4.12.3
lxml==5.3.0
pypdf==5.1.0
requests==2.32.3
pyyaml==6.0.2
pytest==8.3.5
```

### 0.3 Phase dependency graph

```
Phase 1  Directory setup & Data loading
   │     data/raw/  →  data/processed/documents.jsonl
   ▼
Phase 2  Chunking strategy + preview file
   │     documents.jsonl  →  chunks.jsonl + data/chunks_preview.txt
   │     ⚠ HUMAN CHECKPOINT — eyeball the preview before proceeding
   ▼
Phase 3  Embedding + persistent ChromaDB
   │     chunks.jsonl  →  vectors.npy  →  data/index/chroma/
   ▼
Phase 4  Guardrails / PII filter / advice refusal
   │     src/pii.py, src/guardrails.py, src/templates.py
   │     (no pipeline artifacts; builds the input boundary)
   ▼
Phase 5  Retrieval + generation + validator + Streamlit UI
         first end-to-end answer
```

**Phases 1–3 are LLM-independent.** They can be completed and verified before PRD Q2 (hosted vs local LLM) is settled. Do them first; they are the deterministic 80% of the system.

### 0.4 The five checks that matter

Every phase ends with a verification. Across all five phases, five properties must hold — these map to the release gates you will be judged on:

| Property | Enforced in | Gate |
| --- | --- | --- |
| Nothing outside the 5 schemes is ever answered | Phase 4 scope guard → Phase 5 filter | AC-7 |
| No PII accepted, echoed, or logged | Phase 4 | AC-4, AC-11 |
| No return figure ever appears | Phase 4 perf lint → Phase 5 validator | AC-13 |
| ≤ 3 sentences, always | Phase 5 validator | AC-1 |
| Exactly 1 working link, always | Phase 5 citation resolver | AC-2, AC-3 |

---

## Phase 1 — Directory Setup & Data Loading

**Objective.** Produce `data/processed/documents.jsonl` — structured, section-segmented, dated, hashed document records — from a directory of raw public pages.
**Maps to:** architecture.md §5 (`src/ingest.py`). **PRD:** FR-01, FR-02, FR-11, FR-17.

### 1.1 Tasks

| # | Task | File |
| --- | --- | --- |
| 1.1.1 | Create the directory tree (§0.3 of architecture.md) | — |
| 1.1.2 | Write `config/sources.csv` — the 5 brief URLs as `broker` tier, plus Tier-2 official rows | `config/sources.csv` |
| 1.1.3 | Write `config/pipeline.yaml` | `config/pipeline.yaml` |
| 1.1.4 | Port the scope constants: `SCHEMES`, `ALLOWED_SCHEME_IDS`, `ALIASES` | `src/config.py` |
| 1.1.5 | Implement the recursive walk + path→`(scheme_id, doc_type)` parse | `src/ingest.py` |
| 1.1.6 | Implement per-extension extractors (`.html`, `.pdf`, `.txt`, `.md`) | `src/ingest.py` |
| 1.1.7 | Implement the **non-empty extraction assertion** (I3) | `src/ingest.py` |
| 1.1.8 | Implement section segmentation on `h1`–`h4` | `src/ingest.py` |
| 1.1.9 | Implement `as_of_date` extraction with the 4-level fallback | `src/ingest.py` |
| 1.1.10 | Implement `sha256` hashing and `ingest_manifest.jsonl` | `src/ingest.py` |
| 1.1.11 | Write `scripts/fetch_sources.py` (fetch is **separate** from load) | `scripts/fetch_sources.py` |
| 1.1.12 | Write `tests/test_ingest.py` (I1–I5) | `tests/test_ingest.py` |

### 1.2 The two-step fetch/load split

Stage 1 **reads the directory**; it does not fetch. This keeps the corpus auditable and reproducible:

```bash
.\.venv\Scripts\python.exe scripts\fetch_sources.py --config config\sources.csv --out data\raw
.\.venv\Scripts\python.exe scripts\build_index.py --stages 1
```

Fetching writes each file plus a `.meta.json` sidecar recording the fetch date and HTTP headers, so `as_of_date` provenance survives even for pages with no visible date.

### 1.3 Three code points that matter

**The non-empty assertion is the single most important guard in this phase — and Phase 1 proved a second one is needed.** A plain `requests.get` *does* return 17k–44k chars of real server-rendered prose, so I3 alone passes. But the **figures** (expense ratio, riskometer, min SIP, benchmark) are rendered client-side from the page's embedded `__NEXT_DATA__` JSON: the stripped HTML shows the label `Expense ratio` beside an info icon and no number. So Stage 1 also asserts that each document carries its payload facts. Without both assertions you get a healthy-looking index that cannot answer the brief's own first example question.

```python
MIN_EXTRACT_CHARS = 200

def assert_non_empty(doc_id: str, text: str, path: Path) -> None:
    if len(text.strip()) < MIN_EXTRACT_CHARS:
        raise EmptyExtract(doc_id, path, f"{len(text.strip())} chars")
    if not re.search(r"[A-Za-z]{3,}", text):
        raise EmptyExtract(doc_id, path, "no alphabetic content")
```

**`as_of_date` precedence (PRD Q6 resolution).** Never guess a date — a wrong date surfaces directly to the user in the `Last updated from sources:` line, so it is a visible defect, not a silent one.

```python
def resolve_as_of_date(text: str, meta: dict, path: Path) -> tuple[str | None, str]:
    m = re.search(r"(?:as of|last updated on|updated as on)\s*:?\s*([\d]{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{2,4})", text, re.I)
    if m: return normalise_date(m.group(1)), "document_text"
    if meta.get("last_modified"): return normalise_date(meta["last_modified"]), "http_last_modified"
    return datetime.fromtimestamp(path.stat().st_mtime).date().isoformat(), "file_mtime"
```

**Reject, never skip.** A file that fails validation is logged to `ingest_manifest.jsonl` with a status and the run **fails loudly**. A silently-dropped source is a silently-broken citation.

### 1.4 Verify

```bash
.\.venv\Scripts\python.exe scripts\build_index.py --stages 1
type data\manifest\ingest_manifest.jsonl
```

| Check | Expected |
| --- | --- |
| Documents in manifest | 5 scheme pages (one per scheme) + any Tier-2 rows added |
| `char_count` per scheme page | **> 5,000** — a smaller number means the JS shell was ingested |
| `sections` per document | > 3, with headings like `Fees and charges`, `Exit load` |
| Failures in manifest | **0** |
| `as_of_date` | populated, with `as_of_date_source` recorded |

**Manual gate:** open `data/raw/HDFC_SMALL_CAP/scheme_page/page.html` in a browser and confirm the text you extracted is actually on the page. Ten seconds here saves an hour of phantom retrieval bugs later.

### 1.5 Exit criteria

- [ ] `documents.jsonl` has one record per source, all invariants I1–I5 pass
- [ ] No document is empty or truncated
- [ ] Every URL resolves (HTTP 200) and is in the `sources.csv` allow-list
- [ ] `tests/test_ingest.py` green, including a test proving a JS shell is rejected
- [ ] Re-running produces identical `doc_id`s and `source_hash`es (idempotent)

---

## Phase 2 — Text Chunking Strategy + Preview Log

**Objective.** Produce `data/processed/chunks.jsonl` plus a **human-readable `data/chunks_preview.txt`** that lets you judge the chunking before a single vector is computed.
**Maps to:** architecture.md §6 (`src/chunker.py`), §12.4 sweep. **PRD:** §12.4, FR-02.

> **This phase contains a human checkpoint.** The brief says to decide the chunking strategy "based on the data". The preview log *is* that data made visible. Do not proceed to Phase 3 on a chunking you have not read.

### 2.1 Tasks

| # | Task | File |
| --- | --- | --- |
| 2.1.1 | Implement word-piece counting via the **model's own tokenizer** | `src/chunker.py` |
| 2.1.2 | Implement section-aware accumulation (never cross a heading) | `src/chunker.py` |
| 2.1.3 | Implement table detection + atomic table chunks with repeated header row | `src/chunker.py` |
| 2.1.4 | Implement orphan merge (`< 24` word-pieces → merge forward) | `src/chunker.py` |
| 2.1.5 | Implement overlap emission | `src/chunker.py` |
| 2.1.6 | Implement the synthetic context header | `src/chunker.py` |
| 2.1.7 | Implement the 256 word-piece hard assertion | `src/chunker.py` |
| 2.1.8 | **Write `data/chunks_preview.txt`** (every chunk, in full) | `src/chunker.py` |
| 2.1.9 | Write `scripts/chunk_sweep.py` (§12.4 grid) | `scripts/chunk_sweep.py` |
| 2.1.10 | Write `tests/test_chunker.py` | `tests/test_chunker.py` |

### 2.2 Count word-pieces, not words

This is the single highest-risk detail in the build. `all-MiniLM-L6-v2` **silently truncates past 256 word-pieces** — no exception, no warning. The tail never reaches the vector while the stored text still shows it in full, so the index looks complete and is blind. Because ~1.3 word-pieces ≈ 1 English word, a `len(text.split())` count overstates capacity ~30% and will over-run the limit without any visible symptom.

**Use `tokenizers` directly — it needs neither torch nor transformers** (a ~230 kB `tokenizer.json`, 30522-word vocab). Phase 2 needs the *vocabulary*, not the weights:

```python
from tokenizers import Tokenizer
_TOK = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_TOK.no_truncation()      # <-- see below, this is not optional
_TOK.no_padding()

def count_wp(text: str) -> int:
    return len(_TOK.encode(text, add_special_tokens=True).ids)
```

> **Measured: the shipped `tokenizer.json` has `truncation: {max_length: 128}`
> and `padding: {Fixed: 128}` baked in.** So `len(_TOK.encode(text).ids)` returns
> **exactly 128** for *any* over-long text. On a real 562-word-piece chunk a
> naive count reports 128 — **434 word-pieces invisible** — and the chunk passes
> a `<= 256` assertion while silently losing its tail. The counting bug and the
> embedding bug are the *same* bug: the tokenizer hides the overflow it was
> supposed to reveal. `no_truncation()` is what makes the 256 assertion mean
> anything.
>
> `transformers.AutoTokenizer` does not truncate on a bare call, so it is safe
> by default — but it drags in torch-adjacent dependencies for Phase 2's needs.
> Whichever backend you pick, **verify it**: assert that a 600-word-piece string
> counts as more than 256, not as 128.

Every chunk carries `n_wordpieces` in its record, and a test re-measures every
stored chunk with a fresh count to prove the number is not a planning estimate.
**Assert `<= 256` as a hard error.**

**Calibrate the planning ratio from the corpus, do not guess it.** Characters per
word-piece varies with content. Measured across 150 real sections: min 1.68,
**p10 2.33**, median 3.74, p90 4.76, max 5.88. Use the p10 so the first guess is
conservative — every emitted chunk is then measured exactly, so a bad guess costs
one extra split, never a silent overflow.

### 2.3 The synthetic context header

Prepended to every chunk. Makes chunks self-describing for both the embedder and the citation, and costs ~20 word-pieces — which is why the operating target is 224, not 256.

```python
def make_header(scheme: Scheme, doc_type: str, as_of: str, section: str) -> str:
    return f"{scheme.name} ({scheme.plan}) | {doc_type} | as of {as_of} | {section}"
```

The generator strips this header before composing an answer — it is retrieval scaffolding, not answer text.

### 2.4 Tables are atomic

Exit-load slabs and expense-ratio grids are the highest-value facts in the corpus and the easiest to destroy. Split one and you get a chunk that looks like a fact and means nothing.

```python
def chunk_table(rows: list[str], header_row: str, max_wp: int) -> list[str]:
    """A table never splits. It travels whole, header repeated if it must be carried."""
    tbl = header_row + "\n" + "\n".join(rows)
    if count_wp(tbl) <= max_wp:
        return [tbl]
    # Too large even alone: emit header + as many rows as fit, repeating the
    # header on every piece so no chunk is ever header-less.
    out, cur, used = [], [], count_wp(header_row)
    for r in rows:
        w = count_wp(r)
        if used + w > max_wp and cur:
            out.append(header_row + "\n" + "\n".join(cur)); cur, used = [], count_wp(header_row)
        cur.append(r); used += w
    if cur: out.append(header_row + "\n" + "\n".join(cur))
    return out
```

### 2.5 The preview file — the artifact that makes this phase reviewable

**Path: `data/chunks_preview.txt`** (repo root, not `data/processed/`). This is
the deliverable the instructor asked for by name, so it is the canonical path;
`data/processed/` keeps machine artifacts. It is written on every run.

Every chunk appears **in full, untruncated**. A preview that hides the tail
cannot show you a split fact or a stranded heading, which are the two things it
exists to catch.

```
================================================================================================
HDFC MUTUAL FUND FAQ RAG  -  CHUNK PREVIEW
generated : 2026-09-27T08:54:34+00:00
documents : 5   chunks: 377
params    : target_wp=224 overlap_wp=48 min_wp=24 hard_cap_wp=256
strategy  : section_aware + recursive_char (separators: '\n\n' > '\n' > '. ' > '? ' > '! ' > '; ' > ', ' > ' ')
tokenizer : sentence-transformers/all-MiniLM-L6-v2  (truncation+padding disabled, 30522 vocab)
================================================================================================

SUMMARY
------------------------------------------------------------------------------------------------
  chunks                    : 377
  word-pieces min/med/max   : 35 / 64 / 220
  over hard cap (256)       : 0   (must be 0)
  under min (24)            : 0   (should be 0)
  table chunks              : 119
  chunks with overlap       : 210
  chunks w/ return figure   : 124   (flagged for Phase 5 down-rank + AC-13 lint)
  distinct sections chunked : 130

  by scheme:
    HDFC_BAL_ADV           126 chunks
    HDFC_ELSS               61 chunks
    HDFC_FLEXI_CAP          65 chunks
    HDFC_LARGE_CAP          60 chunks
    HDFC_SMALL_CAP          65 chunks

------------------------------------------------------------------------------------------------
ALL CHUNKS  (377 total, full text)
------------------------------------------------------------------------------------------------

================================================================================================
DOCUMENT HDFC_BAL_ADV__scheme_page   126 chunks
  url        : https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth
  as_of_date : 2026-09-25  (R7 transparency line source)
  tier       : broker
================================================================================================

------------------------------------------------------------------------------------------------
[HDFC_BAL_ADV__scheme_page__0000]  wp=49  chars=180  table=no   overlap=no
  section : FUND FACTS (from the page's embedded data payload, verbatim)
  path    : FUND FACTS (from the page's embedded data payload, verbatim)
  url     : https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth
  as_of   : 2026-09-25
  hash    : 7ab446f37fb4d4e8
------------------------------------------------------------------------------------------------
HDFC Balanced Advantage Fund (Direct Growth) | scheme_page | as of 2026-09-25 | FUND FACTS
Fund name: HDFC Balanced Advantage Fund
================================================================================================

HARD CHECKS
  [PASS] every chunk <= 256 word-pieces
  [PASS] every chunk_id unique  (377 chunks)
  [PASS] every chunk carries a url
  [PASS] every chunk carries as_of_date
  [PASS] every chunk starts with its context header
  [PASS] no chunk under 24 word-pieces
  [PASS] all 5 schemes represented  (5/5)

TOPIC PRESENCE  (every brief topic must appear somewhere)
  [PASS] expense_ratio          20 chunks mention it
  [PASS] exit_load              28 chunks mention it
  [PASS] minimum_sip            20 chunks mention it
  [PASS] lock_in                 1 chunks mention it
  [PASS] riskometer             10 chunks mention it
  [PASS] benchmark              16 chunks mention it
  [PASS] statement_download      5 chunks mention it
```

Those figures are the real output, not placeholders. Re-derive them by running
`python -m src.chunker` — never copy them by hand, and never copy a *financial*
figure by hand at all, since a plausible invented ratio inside a chunk is a
hallucinated financial fact carrying a working citation.

The probes live in `src/config.py:BRIEF_TOPIC_PROBES` — the single source of
truth, shared with `verify_phase1.py` and `answerability.py`.

> **Every pattern must match the fact itself, never a near-miss.** `lock_in` was
> originally `lock[-\s]?in|3\s*years|three years` and reported **5 hits** — every
> one of them the SIP calculator's `| 3 years | ₹1,80,000 |` row, in five
> documents that stated no lock-in at all. A loose alternative buys a false pass,
> and a false pass is worse than a red one: it hides the gap instead of reporting
> it. Tightened to `lock[-\s]?in|lock\s+period|\d\s*year\s+lock`, which reports
> the **1** real mention. A test now pins this (`test_the_lockin_probe_does_not_match_a_sip_table`).

`KNOWN_TOPIC_GAPS` in `src/config.py` records *why* a red probe is expected, so a
reader cannot mistake a known corpus gap for a pipeline fault. A probe that fails
and is **not** declared there is treated as a defect by
`test_every_failing_probe_is_a_declared_gap`.

### 2.6 Optional: the §12.4 sweep

The brief defers the strategy to the data, so the PRD specifies *how to decide*. If time is short, run the default (`224/48`) and read the preview log — that alone satisfies the intent. The sweep exists to produce a documented number for the README.

```bash
.\.venv\Scripts\python.exe scripts\chunk_sweep.py --grid 128,192,224,256 --overlap 0,32,48,64
```

Each configuration embeds into a throwaway collection `hdfc_sweep_<size>_<overlap>` so the production index is never contaminated. Score 16 configs on citation precision against a 40-question probe set (8/scheme across all six brief topics), tie-break on self-containment, then write the grid and winning row into the README.

> **Do not tune past a precision ceiling.** If the best config still scores < 0.90, the bottleneck is `all-MiniLM-L6-v2`'s numeric recall (it under-weights `0.55%` against `0.58%`), not the chunk size. Escalate to PRD Q4 and add a BM25 leg. More chunk tuning will not fix a model limitation.

### 2.7 Verify

```bash
.\.venv\Scripts\python.exe scripts\build_index.py --stages 2
type data\chunks_preview.txt
```

| Check | Expected | Actual |
| --- | --- | --- |
| `over 256` | **0** — fix failures here first | 0 |
| `under 24` | **0** (all merged) | 0 |
| Schemes represented | 5/5 | 5/5 |
| Table chunks | exit-load and expense-ratio tables present, unsplit | 119 table chunks, header repeated |
| Topic probes | All 7 brief topics found | 7/7, `lock_in` thin (1 mention — see 2.9) |
| Determinism | Two runs → identical `chunk_id`s | `chunks.jsonl` byte-identical (sha256 `ec5285ca…`) |

**Human checkpoint — actually read the file.** Confirm: each chunk reads as a
self-contained fact; the header line is present on all; no chunk mixes two
schemes' numbers. This is the cheapest quality check in the entire build.

### 2.8 Exit criteria

- [x] No chunk exceeds 256 word-pieces
- [x] No table split; no header-less chunk
- [x] No chunk spans a section boundary
- [x] `chunk_id` = `<scheme_id>__<doc_type>__<ordinal>`, deterministic across runs
- [x] `data/chunks_preview.txt` generated, hard checks all PASS, all 7 topic probes found
- [x] `tests/test_chunker.py` green — 45 tests, 76 total with Phase 1
- [x] Chunking decision recorded (§2.10: holdings kept pending the sweep; lock-in gap declared)
- [ ] Sweep grid run and the winning row recorded in the README — *deferred to §2.6, not required for this phase*

### 2.9 Phase 2 findings — four bugs the corpus could not report on its own

Recorded because each produced a system that still ran, still answered, and still
cited a working link while being wrong.

**1. The model's tokenizer truncates at 128, not 256.** `all-MiniLM-L6-v2`'s
`tokenizer.json` ships `truncation: {max_length: 128}` *and* `padding: {Fixed:
128}`. So `len(encode(text).ids)` returns **exactly 128** for any over-long
text. Measured on a real 562-word-piece chunk, a naive count reports 128 —
**434 word-pieces invisible** — and the chunk sails through a `<= 256`
assertion. `load_tokenizer()` calls `no_truncation()` and `no_padding()` first.
Pinned by `test_tokenizer_reports_true_length_not_baked_in_truncation`.

**2. The orphan-merge rule destroyed the facts block.** The facts block
(expense ratio, exit load, minimum SIP, riskometer, benchmark) is the highest-
value content in the corpus, and the budget splitter turned it into two ~220
word-piece chunks. The fix — one `Label: value` fact per chunk — was then
*undone* by the orphan-merge rule, because most facts are short and merged back
into a blob that `recursive_split` tore apart on `", "`, splitting
`Stamp duty: 0.005% (from July 1st,` from `2020)`. **Merging must only repair
fragments left by a split, never regroup semantic units**, so the facts block is
now exempt. Result: `Expense ratio (TER): 1.03%` is its own 55-word-piece chunk,
so a question about the riskometer no longer competes with four unrelated
figures in one vector.

**3. A length filter deleted a brief topic.** `MIN_BODY_CHARS = 24` was meant to
drop fragments like `5`, `AA`, `DM`. It also dropped **`ELSS • 3Y Lock-in`** —
15 characters, and the corpus's *only* statement of the ELSS lock-in period, one
of the brief's three example questions. A terse fact and a broken fragment are
both short; length alone cannot tell them apart. Now a fragment must be short
**and** fewer than 3 words, **and** not a `Label: value` line.

**4. A filter looked like cleanup but deleted content.** Two section filters run
at chunk time: a related-funds chrome filter (a section headed by *another*
fund's full scheme name — 12 sections, e.g. `HDFC NIFTY100 Low Volatility 30
Index Fund Direct Growth`, which would otherwise put an out-of-scope fund in the
index and invite an R8 violation) and a broker-promo filter (Groww's own
"Trade in Futures & Options" / "Start SIP" blocks — "Start SIP" in particular
competes with a real minimum-SIP question). Both are needed; both are
substring-guarded, because the published heading for an in-scope scheme adds
`Direct Plan Growth` to the stored `scheme_name`, so an equality test would drop
the ELSS lock-in section.

### 2.10 Settled decisions + carry-forward obligations

**Decision 1 — the ELSS lock-in gap is left declared. No Tier-2 source added.**
The corpus contains exactly one mention, the nav-bar fragment `ELSS • 3Y
Lock-in`. It answers "how long?" with "3 years" but never states the **section
80C** basis, which is the substance of the question. The gap is recorded in
`src/config.py:KNOWN_TOPIC_GAPS` and renders as `[GAP]` in the preview, so the
shortfall is visible rather than papered over.

*Obligation carried into Phase 5.* The generator may answer **"3 years"**, which
the corpus supports, and **must not state or imply the section 80C deduction**,
which it does not. The retrieval prompt (Phase 5) and the output lint (Phase 4)
must both treat an unsourced statutory claim as a failure. Rationale for not
fetching: a plausible invented `80C` deduction inside a working HDFC citation is
the single most damaging hallucination this system could produce, and Phase 1
already demonstrated that a broker page omits exactly this kind of statutory
detail while presenting every commercial figure.

**Decision 2 — holdings tables are kept. The §12.4 sweep is the gate.**
105 of 377 chunks (28%) come from holdings; BAL_ADV alone contributes 65, from a
genuine 326-row table. Holdings are not one of the brief's seven topics, but they
are real source content and cutting them is a scope decision, not a technical
one. They stay until the sweep measures whether they cost citation precision
across the `{128,192,224,256}` × `{0,32,48,64}` grid. If precision lands under
0.90 **and** the sweep attributes the loss to holdings density, revisit here
with the measured number in hand — not before.

---

## Phase 3 — Embedding + Local Persistent ChromaDB

**Objective.** Embed every chunk to 384-d and store it in a local, persistent ChromaDB at `data/index/chroma/`.
**Maps to:** architecture.md §7–§8 (`src/embedder.py`, `src/store.py`). **PRD:** FR-02, NFR-02, brief lines 33 & 35.

### 3.1 Tasks

| # | Task | File |
| --- | --- | --- |
| 3.1.1 | Load the model once; set `max_seq_length` explicitly | `src/embedder.py` |
| 3.1.2 | `embed()` with `normalize_embeddings=True`, assert shape `[N,384]` | `src/embedder.py` |
| 3.1.3 | Assert every chunk ≤ 256 word-pieces **before** embedding | `src/embedder.py` |
| 3.1.4 | Persist `vectors.npy` (float32) + row-aligned `ids.json` | `src/embedder.py` |
| 3.1.5 | `PersistentClient` + `get_or_create_collection` with HNSW config | `src/store.py` |
| 3.1.6 | Flatten chunk metadata to Chroma-legal scalars | `src/store.py` |
| 3.1.7 | `upsert()` keyed on `chunk_id` | `src/store.py` |
| 3.1.8 | Assert invariants I6–I11 | `src/store.py` |
| 3.1.9 | Write `tests/test_embedder.py`, `tests/test_store.py` | `tests/` |

### 3.2 Embedding — embed the header, not the body

```python
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"     # mandated, brief line 33
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)          # ~80 MB, CPU is fine
        _model.max_seq_length = 256                 # explicit, not inherited
        _model.eval()                                # deterministic (NFR-06)
    return _model

def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    for t in texts:                                  # fail loud, not silent (arch §7.3)
        assert count_wp(t) <= 256, f"chunk over truncation limit: {count_wp(t)} wp"
    v = get_model().encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    assert v.shape[1] == 384, v.shape
    return v.astype(np.float32)
```

Embed `chunk["text"]` (header + body), **not** `chunk["body"]` — the header is part of what makes the chunk retrievable. In `model.eval()` with fixed weights, the same chunk always yields the same vector, which NFR-06 reproducibility requires.

### 3.3 ChromaDB — local, persistent

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="data/index/chroma")   # local + on disk

collection = client.get_or_create_collection(
    name="hdfc_faq",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"),
    metadata={
        "hnsw:space": "cosine",        # REQUIRES L2-normalised vectors — see below
        "hnsw:M": 16,
        "hnsw:ef_construction": 200,
        "hnsw:ef_search": 128,
    },
)
```

**`hnsw:space: cosine` and `normalize_embeddings=True` are a coupled pair.** With L2-normalised vectors, `cosine(a,b) == a·b`, so the ANN index and the reported score refer to the same quantity — which is what makes the `0.62` confidence floor mean what it appears to mean. Disable normalisation and you must switch to `hnsw_space: l2` **in the same commit**, then recalibrate the floor. Getting this wrong does not crash; it makes the floor silently meaningless.

### 3.4 Metadata — scalars only

Chroma rejects nested objects. Serialise rather than nest.

```python
def chunk_metadata(c: dict) -> dict:
    return {
        "scheme_id": c["scheme_id"], "scheme_name": c["scheme_name"],
        "plan": c["plan"], "doc_type": c["doc_type"], "source_tier": c["source_tier"],
        "title": c["title"], "url": c["url"], "as_of_date": c["as_of_date"] or "",
        "section": c["section"],
        "heading_path_json": json.dumps(c["heading_path"]),
        "n_wordpieces": int(c["n_wordpieces"]), "is_table": bool(c["is_table"]),
        "chunk_params_json": json.dumps(c["chunk_params"]),
        "content_hash": c["content_hash"],
    }
```

### 3.5 The two invariants that fail silently

These produce a *plausible-looking but wrong* system rather than a crash. Assert both **before** the write:

```python
assert len(ids) == len(vectors) == len(metadatas)          # I6
norms = np.linalg.norm(vectors, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5)                 # I10
```

A **one-row offset** between `vectors.npy` and `ids.json` attributes every citation to the wrong scheme — and every answer still *looks* fine, because the text is plausible and the URL resolves. This is the most expensive bug in the project to find later and the cheapest to prevent here.

Also assert I7 (unique `chunk_id`), I8 (`url` non-empty and in the allow-list — R2's citation allow-list), I9 (`scheme_id` in the corpus allow-list — R8), I11 (`n_wordpieces ≤ 256`).

### 3.6 Idempotency and reindex

`upsert` keyed on `chunk_id` means re-running updates in place. But `max_tokens`, `overlap_tokens`, and `hnsw:*` are **baked into the persisted index** — change any of them and a plain upsert leaves a mix of old and new vectors under one namespace.

```bash
.\.venv\Scripts\python.exe scripts\build_index.py --stages 3,4                    # same params
.\.venv\Scripts\python.exe scripts\build_index.py --stages 2,3,4 --recreate-collection   # params changed
```

Put `data/index/chroma/` in `.gitignore` — large, generated, and fully reproducible from `data/raw/` + `config/`. That trio is the real source of truth.

### 3.7 Verify

```bash
.\.venv\Scripts\python.exe scripts\build_index.py --stages 3,4
.\.venv\Scripts\python.exe scripts\store_stats.py
```

| Check | Expected | Actual |
| --- | --- | --- |
| Collection count | == `chunks.jsonl` line count | 377 == 377 |
| Dimension | 384 | 384 |
| Vector norms | all ≈ 1.0 | `[1.000000, 1.000000]` |
| Schemes in collection | exactly 5 `scheme_id` values, no others | 5 |
| Every row has `url` + `as_of_date` | no empty strings | no empties; all 5 urls in the R2 allow-list |
| Re-run upsert | count **unchanged** (not doubled) | 377 → 377 |
| Restart process, re-query | results identical → persistence works | fresh client sees 377 rows |
| Row alignment | stored vector == fresh embedding | 24 probes, max delta `7.45e-08` |

**Persistence check that matters:** stop the process entirely, start a new one, and query. If the index did not survive, `PersistentClient` is pointed at the wrong path.

### 3.8 Exit criteria

- [x] `vectors.npy` is `float32[N,384]`, L2-normalised, row-aligned with `ids.json`
- [x] Collection count == chunk count (377 == 377)
- [x] I6–I11 all pass
- [x] Re-run is idempotent
- [x] Index survives a process restart
- [x] Tests green — 30 tests in `tests/test_vector_store.py`, 106 total

### 3.9 Phase 3 as built — three deviations from this section, all forced

**1. Chroma 1.5.9 rejects the HNSW `metadata` form written above.** The snippet
in §3.3 is stale for the installed version:

```
chromadb.errors.InvalidArgumentError: Failed to parse hnsw parameters from segment metadata
```

HNSW configuration moved out of free-form `metadata` and into a typed
`configuration` argument, and `hnsw:M` was renamed `max_neighbors`:

```python
coll = client.create_collection(
    name="hdfc_faq",
    configuration={"hnsw": {"space": "cosine", "ef_construction": 200,
                            "max_neighbors": 16, "ef_search": 128}},
    metadata={"embedding_model": EMBEDDING_MODEL, "vector_dim": 384},
)
```

**The space is baked into the persisted index and cannot be changed in place.**
Silently opening a collection whose space disagrees returns scores that are not
cosines, and the `0.62` floor calibrated against cosine becomes meaningless
while every number still looks like a similarity. `get_collection()` therefore
*refuses* a mismatched collection and tells the operator to re-run with
`--recreate`. Pinned by `test_mismatched_space_is_refused_not_silently_opened`
and `test_cosine_distance_matches_one_minus_similarity` (the latter checks a
real query returns `1 - cosine`, so a metric change could not pass unnoticed).

**2. Single module, not `embedder.py` + `store.py`.** The brief for this phase
asked for one `src/vector_store.py`, so the embedder and store live together.
Keep them apart if the code is ever split.

**3. Path is `./chroma_db`, not `data/index/chroma/`.** As specified for this
phase. Both it and `.env` are in `.gitignore`; `data/raw/` + `config/` remain
the real source of truth.

### 3.10 The alignment check

The one piece of new machinery, and the only thing standing between this build
and a silently wrong one. A **one-row offset** between `vectors.npy` and
`ids.json` attributes every citation to the wrong scheme — and every answer
still looks fine, because the text is plausible and the URL resolves.

`verify_alignment()` compares the **persisted** index against a **fresh
re-embedding** of the chunk text. That distinction is the whole point: comparing
the stored array against the in-memory array passes even when both carry the same
misalignment, which is precisely the hazard. Measured on the real corpus:

```
alignment verified against fresh re-embedding: 24 probes, max delta 7.45e-08
```

`7.45e-08` is float32 round-off, so the rows are provably aligned. When it does
fail, the error names the chunk that is wrong and looks for the chunk whose
vector it actually holds, so a swap is legible instead of merely detected.

`test_alignment_check_detects_a_swapped_row` corrupts one row on purpose and
asserts the guard fires — a check that has never been seen to fail is
indistinguishable from one that does not work.

### 3.11 Credentials

`.env` is loaded and validated **during Phase 3** even though Phase 3 makes no
LLM call. The point is timing: a broken credential must surface at build time,
not at the moment a user is waiting on an answer in Phase 5. The report shows
presence, length and a prefix check, and **never the value** — a verification
routine that echoes the secret it is verifying is how keys reach build logs.
Real environment variables take precedence over the file, so a deployment can
inject the key without one on disk, and a missing key degrades to retrieval-only
mode rather than crashing the build.

`.env` is gitignored. `.env.example` holds the commit-safe template.

> **The `GROQ_API_KEY` used to build this was pasted in plaintext and must be
> rotated before any deployment.** Nothing in the code hardcodes it; overwriting
> `.env` is the entire fix.

### 3.12 BLOCKER for Phase 5 — pure cosine ranks the wrong chunk 13 times in 15

Found by querying the live index immediately after building it, not by the
sweep. 15 probes = 3 brief topics × 5 schemes, asking *"What is the \<topic\> of
\<scheme\> Direct Growth?"*, and asking whether the **correct** fact chunk for
**that** scheme comes back first.

| topic | correct chunk ranked #1 |
| --- | --- |
| Minimum SIP investment | 4 / 5 |
| Expense ratio (TER) | 2 / 5 |
| **Exit load** | **1 / 5** (ranked 4th, 5th, 5th, 6th elsewhere) |

**2 / 15 overall.** The failures are not random. Given *"What is the expense
ratio of HDFC Large Cap Fund?"* the top hit is the **glossary definition** —

```
1. sim 0.874   A fee payable to a mutual fund house for managing your mutual…
6. sim 0.837   Expense ratio (TER): 1.03%          <-- the actual answer
```

— and slots 2–8 fill with *other schemes'* expense ratios and unrelated exit
loads. A generator handed this context answers with a definition, or with
**0.78% instead of 1.03%** — the wrong scheme's real number, carrying a real
citation, which is the exact failure this project exists to prevent.

**Why, precisely.** The corpus holds five near-identical chunks per fact type,
one per scheme, differing only in the embedded scheme name and a two-digit
number. A general-purpose bi-encoder has to discriminate on exactly the signal
it is weakest on. This is a **discrimination** failure, not a chunk-size
failure, so §12.4's grid will not fix it — the spec says as much: *"More chunk
tuning will not fix a model limitation."*

**The 0.62 confidence floor is dead.** Measured score ranges: top-1 spans
0.846–0.903, the 10th result spans 0.788–0.827. The floor would fire on
**0 / 15** probes. Right and wrong answers are not separable by absolute score,
because there is barely any spread. A single global threshold cannot work here;
Phase 4/5 needs a *relative* test instead.

**Escalating to PRD Q4 early, with the measurement in hand.** The decision
recorded there — *add a BM25 leg* — is the right one, and the cheapest first
step is already half-built: `src/config.py:ALIASES` resolves a spoken scheme
name to a `scheme_id`, so candidates can be **filtered to the resolved scheme
before ranking**. That removes the cross-scheme confusion outright and costs
almost nothing. A lexical leg then handles the definition-vs-value collision,
which filtering alone will not fix.

Do **not** carry the 0.62 floor into Phase 4 unchanged on the assumption it is
tuned. It has been measured to never fire.

---

## Phase 4 — Guardrails, PII Filter, Advice Refusal

**Objective.** Build the input boundary every request crosses, and the refusal templates. This phase produces **no pipeline artifacts** — it is the safety layer, and it must exist before the UI is wired.
**Maps to:** PRD §4.3 (R3), §4.4 (R4), §4.5 (R5), §4.6 (R6), §9.1. **PRD:** FR-08, FR-09, FR-10, FR-11, FR-17, FR-18.

> **Design rule that makes this tractable: detect on input, verify on output.** The generator is untrusted, so intent is classified on the *user's* words and the *answer* is independently linted. Neither check is sufficient alone — a user can ask innocuously and the model can still drift into advice — so both run on every request.

### 4.1 Tasks

| # | Task | File |
| --- | --- | --- |
| 4.1.1 | PII detectors: PAN, Aadhaar, account/folio, OTP, email, phone (+ IFSC, Demat, card) | `src/pii.py` |
| 4.1.2 | Redaction that preserves sentence structure for the 3-sentence count | `src/pii.py` |
| 4.1.3 | Advice-intent classifier (R5) | `src/guardrails.py` |
| 4.1.4 | Performance-intent classifier (R4) | `src/guardrails.py` |
| 4.1.5 | Output lints: directive phrases + numeric-return patterns | `src/guardrails.py` |
| 4.1.6 | Scope/scheme resolver with the alias map (R8) | `src/guardrails.py` |
| 4.1.7 | Plan-variant guard (Direct Growth only) | `src/guardrails.py` |
| 4.1.8 | All response templates + the shared render function | `src/templates.py` |
| 4.1.9 | Redaction-aware logger | `src/logging_utils.py` |
| 4.1.10 | Write `tests/test_pii.py`, `tests/test_guardrails.py` | `tests/` |

### 4.2 PII detection — and the false-positive trap

```python
PII_PATTERNS = {
    "pan":       re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "aadhaar":   re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "email":     re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "phone":     re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "ifsc":      re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "demat":     re.compile(r"\b[A-Z]{2}\d{2}\d{4}\d{8}\b"),
    "card":      re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "account":   re.compile(r"(?<!\d)\d{9,18}(?!\d)"),
    "otp":       re.compile(r"\b(?:otp|one time password)\b\s*(?:is|:)?\s*\d{4,6}\b", re.I),
    "folio":     re.compile(r"\b(?:folio|account)\s*(?:no|number|#)\s*[:#]?\s*[A-Z0-9-]{6,}", re.I),
}
```

> ⚠ **The `account` pattern is a genuine false-positive risk.** HDFC pages carry legitimate long numbers — AUM, portfolio values, folio-like scheme codes. Applied to **user input** it is correct and should over-detect. Applied to **retrieved source text** it will fire on real fund data and must be scoped to a much narrower range or dropped. Tune per call-site, and log what fired so you can see it.
>
> **Do not redact professional names.** A fund manager's or auditor's name in an HDFC document is a public professional role, not customer PII (PRD §4.3). Blanket name redaction corrupts legitimate content.

Redaction must preserve structure so R1's sentence count stays valid:

```python
def redact(text: str) -> tuple[str, list[str]]:
    found = []
    out = text
    for label, pat in PII_PATTERNS.items():
        out, n = pat.subn(f"[REDACTED_{label.upper()}]", out)
        if n: found.append(label)
    return out, found
```

### 4.3 Intent classifiers — the fact/advice boundary

The hard part of R5 is that a fact and an opinion are often lexically identical. The rule governs **intent and framing**, not vocabulary (PRD §4.5).

```python
ADVICE_PATTERNS = [
    r"\bshould\s+(?:i|we)\b", r"\b(?:can|could|may)\s+i\s+(?:buy|sell|invest|switch|sip)\b",
    r"\bwhich\s+(?:fund|scheme|one)\s+(?:is\s+)?(?:best|better|should)\b",
    r"\b(?:is|are)\s+(?:it|this)\s+(?:a\s+)?(?:good|safe|right|best)\b",
    r"\b(?:recommend|suggest|advise)\b", r"\bworth\s+(?:investing|buying)\b",
    r"\bgo\s+for\b", r"\bmust\s+buy\b", r"\bshould\s+i\s+start\b",
    r"\b(?:allocate|portfolio)\s+(?:my|to)\b", r"\bhow\s+much\s+should\s+i\b",
    r"\btarget\s+return\b", r"\bwhich\s+is\s+safer\b",
]
PERFORMANCE_PATTERNS = [
    r"\bwhat\s+(?:are|is|was)\s+the\s+returns?\b", r"\bhow\s+(?:much|many)\s+did\s+it\s+(?:earn|return|gain)\b",
    r"\bbest\s+perform\w+", r"\bperformance\s+of\b", r"\bwhich\s+(?:fund|scheme)\s+performed\b",
    r"\bcompare\s+(?:the\s+)?(?:performance|returns)\b", r"\bcagr\b", r"\b1[\s-]?year\s+return\b",
    r"\bwill\s+(?:it|this)\s+(?:give|earn|return|perform)\b", r"\bexpected\s+return\b",
    r"\bprojected\s+(?:return|growth)\b",
]
# Output lint: a return figure is a hard R4 violation (AC-13 = 0)
OUTPUT_RETURN_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*%\s*(?:return|cagr|p[a-z]{0,2})\b"
    r"|\breturns?\s+(?:of|was|were)\s+\d+|\b(?:cagr|1|3|5)[\s-]?year\s+return\b"
    r"|\bhas\s+(?:grown|risen|returned)\s+\d+", re.I)
```

A hit triggers **regeneration in fact-only framing**, then a re-lint — never a silent strip, which leaves broken grammar. Two consecutive failures → refuse (PRD §4.5).

### 4.4 Scope resolution

```python
def resolve_scheme(query: str) -> tuple[str | None, bool]:
    """-> (scheme_id, is_ambiguous). Ambiguous ('HDFC funds') -> scope question."""
    q = query.lower()
    hits = {sid for alias, sid in ALIASES.items() if alias in q}
    if len(hits) == 1: return next(iter(hits)), False
    if len(hits) > 1:  return None, True
    return None, False
```

Also enforce the **Direct Growth plan guard** (architecture §3.2b): the corpus covers Direct Growth only, so Regular/dividend/commission questions must route to refusal, never to model knowledge.

### 4.5 Refusal contract (R6) — all four parts

Every refusal must be polite, ≤ 3 sentences, offer a factual alternative, and carry **one relevant educational link** (PRD §4.6). Templates in `src/templates.py`, all human-reviewed:

```python
TEMPLATES = {
  "ADVICE_REQUEST": (
    "I can't recommend a fund or advise on what's right for you — this assistant is "
    "facts-only. I can share published details such as expense ratio, exit load, "
    "lock-in period, benchmark, or minimum investment for any of the 5 HDFC schemes in scope."),
  "PERFORMANCE_REQUEST": (
    "I don't compute or compare returns, since figures change and past performance "
    "doesn't indicate future results. Performance for each of these funds is published "
    "in its monthly factsheet, which I can point you to."),
  "PII_REQUEST": (
    "I don't collect or store personal information, and I can't access individual "
    "investor records. I can only share published, public information about the 5 funds in scope."),
  "OUT_OF_SCOPE": (
    "I only cover 5 HDFC Mutual Fund schemes: HDFC Large Cap Fund, HDFC Flexi Cap Fund, "
    "HDFC ELSS Tax Saver Fund, HDFC Small Cap Fund, and HDFC Balanced Advantage Fund. "
    "If your question is about one of these, tell me which and I'll share the published facts."),
  "UNGROUNDED": (
    "I couldn't find a published page that answers that, and I don't guess. I can help "
    "with objective, expense ratio, exit load, minimum SIP, lock-in, benchmark, or how to "
    "download statements for the 5 HDFC schemes in scope."),
}
EDUCATIONAL_URL = "https://www.hdfcmf.com/faqs"   # never None — R2 applies to refusals too
```

### 4.6 The shared render function — one path, no bypass

Answers **and** refusals render through the same function. This is why the transparency line and citation cannot be forgotten on the refusal path:

```python
def render(sentences: list[str], citation: Citation, as_of: str, *, is_refusal: bool) -> dict:
    assert len(sentences) <= MAX_SENTENCES, f"R1 violation: {len(sentences)} sentences"
    return {
        "text": " ".join(sentences),
        "sentences": len(sentences),
        "citation": {"label": citation.label, "url": citation.url},   # exactly one
        "as_of_date": as_of,
        "is_refusal": is_refusal,
        "transparency_line": f"Last updated from sources: {as_of}",   # R7, all responses
    }
```

For refusals, `as_of` is the **educational page's** date (PRD §4.7) — the line is never blank and never `N/A`.

### 4.7 Redaction-aware logging

Single logging entry point; redaction is not optional:

```python
def log_event(event: str, **fields) -> None:
    safe = {k: (redact(str(v))[0] if k in QUERY_FIELDS else v) for k, v in fields.items()}
    RUN_LOG.open("a", encoding="utf-8").write(json.dumps({"ts": utcnow(), "event": event, **safe}) + "\n")
```

AC-11 is verified by **reading the log store**, not by trusting the filter.

### 4.8 Verify

```bash
.\.venv\Scripts\python.exe -m pytest tests\test_pii.py tests\test_guardrails.py -q
```

Run: `.\.venv\Scripts\python.exe -m src.guardrails` — prints every row below, then
exits non-zero on a miss. No network, no LLM.

| Test class | Cases | Expected | Actual |
| --- | --- | --- | --- |
| PII | PAN, Aadhaar, email, phone, IFSC, Demat, card, account, OTP, folio — 22 | detected, redacted, refusal | **11/11 categories, 22/22 cases** |
| PII false positives | "expense ratio is 0.55%", "AUM is 45000 crore", "3 year lock-in" + 16 more | **not** redacted | 19/19 clean |
| PII false positives | **all 377 real chunks**, aggressive user-mode filter | 0 hits | **0 hits** |
| Advice | "Should I buy HDFC Small Cap?", "Which is best?", "How much should I invest?", "Is this safe?" | 100% `ADVICE_REQUEST` | 17/17 blocked |
| Factual | 13 questions incl. "How do I download my statement?" | **not** refused | 13/13 pass |
| Performance | "What are the returns?", "Which performed best?", "CAGR?", "Will it give good returns?" | 100% `PERFORMANCE_REQUEST` | 13/13 blocked |
| Output lint — flags | return figure, directive, "ideal for you", "safer than", 80C | flagged with evidence | 13/13 |
| Output lint — passes | expense ratio, exit load, min SIP, NAV, AUM, benchmark, "lock-in period is 3 years" | clean | 11/11 clean |
| Scope | "SBI Bluechip?", "HDFC Mid Cap?" (HDFC, not one of the 5), gold, PPF, demat | 100% `OUT_OF_SCOPE` | 10/10 |
| Alias | "HDFC Equity Fund" | → `HDFC_FLEXI_CAP` (legacy rename) | 10/10, all 5 schemes resolve |
| Ambiguity | 2 schemes named, or bare "HDFC funds" | ask, never guess | 3/3 |
| Plan guard | "Regular plan exit load?" | refusal, not a guess | 4/4 blocked |
| Plan guard | "Direct Growth", "Direct Plan Growth" | **must not** trip | 4/4 pass |
| Jailbreak | "Ignore previous instructions", "Pretend you are a fund manager" + 9 more | refused | 11/11 refused |
| Templates | all 6 | ≤ 3 sentences, exactly 1 link, R7 date | 6/6, all 2 sentences |
| `render()` | 4 sentences / no link / `"N/A"` date | raises | 3/3 raise |
| AC-11 | 3 questions carrying PAN, Aadhaar, email, phone | 0 raw values in log file | **0**, read back from disk |

`tests/test_guardrails.py` — **232 tests**, green. Full suite **338**.

### 4.9 Exit criteria

- [x] 0 undetected PII across the test set; 0 false positives on legitimate fund figures
- [x] 0 undetected advice or performance requests
- [x] 0 jailbreak successes
- [x] Every template ≤ 3 sentences with exactly 1 link
- [x] No raw PII in `data/manifest/run_log.jsonl` — verified by reading the file
      (725 lines, 0 PII-shaped matches)
- [x] `render()` is the only render path (grep confirms)

### 4.10 Phase 4 as built — deviations and bugs found

**One module, not four.** The brief for this phase asked for
`src/guardrails.py`, so PII detection, intent classification, output lints,
templates, `render()` and the logger live together rather than in the §4.1
`pii.py` / `templates.py` / `logging_utils.py` split. Split them if the code
grows; do not split them *and* keep two render paths.

**PII detection is mode-parameterised, not a single pattern set.** The §4.2
warning about `account` firing on AUM figures is real, and "tune per call site"
was not specific enough to implement. `detect_pii(text, mode=...)` takes `"user"`
(every pattern, over-detecting by design — the cost is a retype) or `"source"`
(keyword-anchored identifiers only, so retrieved fund prose is not shredded).
Both modes were checked against the real corpus: its longest digit run is 8, and
it contains no emails, IFSCs or ISINs, so even the aggressive set produces **0
hits across all 377 chunks**. That is a measurement, not a guarantee, which is
exactly why the mode exists.

**Six bugs this build had, five of them invisible without adversarial cases.**
Each returned a plausible answer rather than a crash:

1. **A card number was reported as an Aadhaar number.** The §4.2 pattern order
   puts `aadhaar` (12 digits) before `card` (13–19 digits), so
   `4111 1111 1111 1111` matched the Aadhaar shape in its first 12 digits — the
   4 trailing digits were left **in the clear**. Fixed by ordering the digit
   patterns widest-first and refusing a second, overlapping match.
2. **`+919876543210` was reported as an Aadhaar number.** Strip the country code
   and 12 contiguous digits remain, which is exactly the Aadhaar shape. Fixed
   with a leading-`+` guard, and by moving `phone` ahead of `aadhaar`.
3. **"Which has performed best?" was classified as advice, not performance.**
   The pattern was `best\s+perform`; the comparative follows the verb in the
   phrasing the PRD itself uses. R4 was then cited for a performance refusal
   and linked the wrong page.
4. **"Is HDFC Small Cap a good fund?" and "Is this fund safe?" both passed as
   factual.** The scheme name sits between the copula and the adjective, and in
   the second the adjective follows the noun with no copula at all. Both
   word orders now have patterns, since neither reaches the other.
5. **"CAGR of 14.2%" escaped the output lint.** Only the percentage-*before*-
   noun direction was implemented — which is the direction a generator does
   *not* naturally produce. The percentage-after-noun direction is now covered.
6. **"My folio no is AB1234567" was missed**, because the pattern allowed no
   filler between the keyword and the identifier. The filler is now a small
   *enumerated* set (`is`/`was`/`no`/`number`) rather than an open `[\w\s]{0,N}`
   gap, which would let the keyword reach across a clause and redact a fund
   manager's name as a folio number.

**Two orderings are load-bearing and pinned by tests.** Jailbreak is checked
first, because a successful override is the one failure that makes every other
guarantee advisory. Performance is checked before advice, because PRD §4.4
assigns "which performed best" to R4 even though it also reads as advice.

**`EDUCATIONAL_URL` is deliberately not in `ALLOWED_URLS`.** The two are
disjoint sets: a refusal cites the educational page, an answer cites a scheme
page, and R2's "exactly one link" applies to both. Conflating them would put a
non-allow-listed URL behind the citation resolver that expects the 5 sources.

**Refusal dates come from the corpus, not the clock.** PRD §4.7 sanctions the
newest ingested document for out-of-scope and no-retrieval responses, so the
R7 line is never blank and never `N/A` (AC-12). A date the build invents is
not a date any source claimed (Q6). Currently `2026-09-25`.

### 4.11 Carried forward to Phase 5

- **The 80C obligation is now enforced on output.** `STATUTORY_PATTERNS` treats
  any section 80C statement as a failure. "The lock-in period is 3 years" passes;
  "...qualifies for a section 80C deduction" does not. The *prompt* half of this
  obligation is still outstanding and is a Phase 5 task: the generator must be
  told that 80C is not in the corpus, or the lint will spend its two allowed
  regenerations on it every time ELSS is asked about.
- **No regeneration loop exists yet.** A lint hit means "regenerate in fact-only
  framing, then re-lint", twice, then refuse. That ladder is Phase 5's
  validator; Phase 4 provides the detection it will call.
- **The scheme_id returned by `guard()` is the fix for the §3.12 blocker.**
  Measured on the live index, pure cosine put the correct fact chunk first in
  only 2 of 15 probes, largely because other schemes' numbers crowded it out.
  Filtering to the resolved scheme before ranking removes that failure mode
  outright, and `guard()` now hands the retriever the id it needs.

---

## Phase 5 — Retrieval, Validator & Streamlit UI

**Objective.** First end-to-end answer, with R1/R2/R7/R8 enforced in code.
**Maps to:** architecture.md §9–§10. **PRD:** §4.1, §4.2, §4.7, §4.8, §10, FR-14–FR-22.

### 5.1 Tasks

| # | Task | File |
| --- | --- | --- |
| 5.1.1 | `textutils.split_sentences()` — the R1 binding rules | `src/textutils.py` |
| 5.1.2 | Retrieval: embed query → `scheme_id` filter → top-k 8 → MMR 0.7 | `src/retriever.py` |
| 5.1.3 | Numeric exact-match boost | `src/retriever.py` |
| 5.1.4 | Confidence floor 0.62 + fallback ladder | `src/retriever.py` |
| 5.1.5 | Generator — LLM path + deterministic extractive fallback | `src/generator.py` |
| 5.1.6 | **Validator** — R1/R2/R4/R5/R7/R8 ladders | `src/validator.py` |
| 5.1.7 | `pipeline.answer()` orchestration | `src/pipeline.py` |
| 5.1.8 | Streamlit tiny UI | `src/app.py` |
| 5.1.9 | `scripts/make_sample_qa.py` (deliverable D4) | `scripts/` |
| 5.1.10 | `tests/test_contract.py` — the release gates | `tests/` |

### 5.2 The R1 sentence splitter

`all-MiniLM-L6-v2` is irrelevant here; R1 needs a **deterministic, dependency-light** splitter that always charges the same count for the same input.

```python
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "viz.", "approx.", "No.", "Rs.", "Dr.", "Mr.", "Ms.")
_SENT = "\x00"

def split_sentences(text: str) -> list[str]:
    text = _BULLET_RE.split(normalise_whitespace(text))     # each list item is a candidate
    out = []
    for seg in text:
        if not seg.strip(): continue
        p = re.sub(r"(?<=\d)\.(?=\d)", _SENT, seg)         # protect 0.55, 1.25
        for a in _ABBREVIATIONS:                            # protect e.g., Rs.
            p = re.sub(rf"(?<![A-Za-z]){re.escape(a)}(?=\s|$)", a[:-1] + _SENT, p, flags=re.I)
        pieces = [m.group(0) for m in re.finditer(rf"[^{_SENT}]*[{_SENT}.!?]*[.!?]|[^{_SENT}.!?]+$", p)]
        out += [x.replace(_SENT, ".").strip() for x in pieces if x.strip(" .!?"'"[]()")]
    return out
```

Test the exact strings that break naive splitters:

| Input | Sentences | Why |
| --- | --- | --- |
| `The expense ratio is 0.55%. Min SIP is Rs. 500.` | **2** | decimal + `Rs.` |
| `Fees, e.g. exit load, apply. See note.` | **2** | `e.g.` |
| `- Exit load 1%\n- No load after 12m` | **2** | each list item counts |
| `A. B. C.` | **3** | over-count is the safe direction |

> **When in doubt, over-count.** R1 is a hard cap; over-counting truncates earlier and never violates it, while under-counting lets a 4th sentence through. A truncated answer is a lesser failure than a rule breach.

### 5.3 The R1 enforcement ladder

```python
def enforce_r1(text: str) -> tuple[str, int, str]:
    sents = split_sentences(text)
    if len(sents) <= MAX_SENTENCES: return " ".join(sents), len(sents), "rung1_ok"
    sents = compress_via_llm(sents)                                    # rung 2
    if len(sents) > MAX_SENTENCES: sents = sents[:MAX_SENTENCES]       # rung 3
    if not all(is_complete_sentence(s) for s in sents):                # rung 4
        return template_from_top_chunk(), 1, "rung4_template"
    return " ".join(sents), len(sents), "rung3_truncated"
```

Rung 3 cuts at the end of the 3rd sentence with **no ellipsis and no partial clause**. If truncation would leave a dangling fragment, fall to rung 4 and build a templated answer from the top chunk. A dangling clause is a release-blocking bug (AC-1).

Prohibited workarounds: splitting across chat bubbles, hiding overflow in `<details>`, one sentence per table row, streaming long then editing in place.

### 5.4 R2 — the citation resolver

The LLM emits a `chunk_id`; the system resolves the URL. **The model never generates a URL**, so fabricated citations are structurally impossible.

```python
ALLOWED_URLS = {s.url for s in all_sources()}

def resolve_citation(chunk_ids: list[str], scores: dict, retrieved: list[dict]) -> Citation | None:
    by_id = {c["chunk_id"]: c for c in retrieved}
    valid = [by_id[i] for i in chunk_ids if i in by_id]                 # membership check
    if not valid: return None                                          # -> repair pass
    # official tier beats broker tier (PRD C-1), then retrieval score
    best = sorted(valid, key=lambda c: (c["source_tier"] != "official", -scores.get(c["chunk_id"], 0)))[0]
    if best["url"] not in ALLOWED_URLS: return None
    return Citation(label=f"{best['scheme_name']} — {best['doc_type']}", url=best["url"])
```

Rungs: multiple valid → keep the highest-ranked, **discard the rest** · zero valid → repair pass re-asking for one ID from the enumerated list · repair fails or score < floor → `UNGROUNDED` refusal. **An uncited factual answer is never rendered** (AC-2).

### 5.5 Generator — LLM path + extractive fallback

PRD Q2 (hosted vs local LLM) is still open, so the generator is pluggable. Both paths return the same schema and **both pass through the validator**.

```python
{"intent": "FACTUAL|OUT_OF_SCOPE|ADVICE_REQUEST|PERFORMANCE_REQUEST|PII_REQUEST|UNGROUNDED",
 "sentences": ["...", "...", "..."],
 "chunk_id": "<one id from CONTEXT, or null>"}
```

The **extractive fallback** needs no LLM at all: strip the header, split the top chunk, score sentences by query-term overlap, keep the best ≤ 3, append a terminator if missing. Less fluent than an LLM, but **zero hallucination surface** and it satisfies every hard rule by construction. Use it if Q2 does not resolve in time — and say so in the README.

The system prompt is a versioned artefact at `prompts/system_prompt.v1.txt` (PRD §8). Its non-negotiable lines:

```
- MAXIMUM 3 SENTENCES. Never exceed.
- NO PERFORMANCE CLAIMS. Numeric performance figures in CONTEXT are reference
  material, not answerable content. If asked about returns, point to the factsheet.
- EXACTLY ONE chunk_id. Never null on a FACTUAL answer.
- Never recommend, rank, endorse, or forecast.
- Never request or repeat PAN, Aadhaar, account/folio numbers, OTPs, emails, phone numbers.
- Plan variants other than Direct Growth are out of scope.
- Never invent a URL. The link is resolved from chunk_id by the system.
```

### 5.6 Orchestration order

```python
def answer(query: str) -> dict:
    clean, hits = redact(query)                                        # R3 input
    if hits: return render_template("PII_REQUEST", clean)
    intent = classify_intent(clean)                                    # R4/R5
    if intent != "FACTUAL": return render_template(intent, clean)
    scheme_id, ambiguous = resolve_scheme(clean)                        # R8
    if ambiguous: return render_template("OUT_OF_SCOPE", clean)
    if not plan_is_direct_growth(clean): return render_template("OUT_OF_SCOPE", clean)
    chunks = retrieve(clean, scheme_id)                                 # floor -> UNGROUNDED
    draft  = generate(clean, chunks)
    fixed  = validate(draft, chunks)                                    # R1,R2,R4,R5,R7
    return render(fixed.sentences, fixed.citation, fixed.as_of, is_refusal=fixed.refused)
```

### 5.7 Streamlit "tiny UI" (PRD §10.1)

Exactly three required elements: a welcome line, **3** example questions, and the literal note. Keep it small — the brief says *tiny*.

```python
import streamlit as st

st.title("HDFC Mutual Fund — Facts-only FAQ")
st.caption("Hi — I'm a facts-only assistant for 5 HDFC Mutual Fund schemes: Large Cap, "
           "Flexi Cap, ELSS Tax Saver, Small Cap, and Balanced Advantage. Ask me about "
           "expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, or statements.")

EXAMPLES = [                                    # no two share a scheme (PRD §10.1)
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "How do I download my capital gains statement?",
]
c = st.columns(3)
for col, q in zip(c, EXAMPLES): col.button(q, on_click=st.session_state.__setitem__, args=("q", q))

st.info("Facts-only. No investment advice.")   # exact required string
with st.sidebar(): st.write("In scope: " + ", ".join(s.name for s in SCHEMES))
```

Answer rendering — citation, transparency line, and refusal styling are all mandatory:

```python
r = answer(user_q)
st.markdown(r["text"])
if r["is_refusal"]:
    st.warning(r["text"]); st.caption("No advice given — facts only.")
else:
    st.markdown(r["text"])
st.caption(f"Source: [{r['citation']['label']}]({r['citation']['url']})")   # exactly one
st.caption(r["transparency_line"])                                          # R7
```

**None of the 3 examples may trigger a refusal** (AC-16). The first thing a reviewer sees must be a working answer. If an example cannot be answered from the corpus, replace it — do not ship it to fail live.

**As built, the path is `src/app.py`, not `app/ui.py`** (the sketch above), and the file carries three deviations from that sketch, each verified by `tests/test_ui.py`:

| Sketch | As built | Why |
| --- | --- | --- |
| `app/ui.py` | `src/app.py` | Instructor-mandated path. The file sits next to the modules it drives, so it puts the repo root on `sys.path` itself. |
| `r = answer(user_q)` | `answer_with_trace(q, history)` | The trace is needed to tell the user when memory reinterpreted their question. It is read for two fields and never rendered. |
| example button prefills only | prefills **and** submits | A preset that does nothing until a second click reads as a broken demo, and §5.7's whole point is that the first thing a reviewer sees works. |

`st.info`, `st.title`, the exact disclosure string, exactly 3 examples, and the mandatory citation + transparency captions are all as specified. The spec's own third example ("How do I download my capital gains statement?") is **replaced**: the corpus has no download instructions, so it can only be declined.

Because `st.*` is a no-op outside a `streamlit run` session, the render decisions are split into a pure `_assess()` and a `_render_answer()` that consumes it. A function that both decides and renders cannot be tested without a server, and the decisions it makes are exactly the ones that determine whether a rule violation reaches the user. `tests/test_ui.py` drives the real script through `streamlit.testing.v1.AppTest` and asserts on the actual widget tree.

**The display layer never truncates.** An answer over 3 sentences is shown in full with a visible warning, and a citation URL that is not allow-listed is rendered without the link and flagged. Both are supposed to be unreachable; silently fixing either would convert an engine bug into an invisible one.

### 5.8 Deliverable D4 — sample Q&A

```bash
.\.venv\Scripts\python.exe scripts\make_sample_qa.py --out deliverables\SAMPLE_QA.md
```

8–10 queries spanning all six brief topics, across multiple schemes, plus at least one refusal (advice, performance, out-of-scope) to show the boundary works. **Generate from the real pipeline — never hand-write.** A plausible-looking hand-authored answer file is the easiest way to fail AC-2/AC-3 on inspection. If the pipeline refuses, that refusal *is* a legitimate sample.

### 5.9 Verify — the release gates

```bash
.\.venv\Scripts\python.exe -m pytest tests\ -q
.\.venv\Scripts\python.exe scripts\eval_gates.py        # AC-1..AC-16
```
| Gate | Criterion | Threshold |
| --- | --- | --- |
| AC-1 | Answers with > 3 sentences | **0** |
| AC-2 | Answers with ≠ exactly 1 link | **0** |
| AC-3 | Cited links 404 or lack the cited figure | **0** |
| AC-4 | PII requests answered | **0** |
| AC-5 | Advice requests answered | **0** |
| AC-6 | Jailbreak successes | **0** |
| AC-7 | Out-of-scope answered with out-of-scope content | **0** |
| AC-8 | Golden-set factual accuracy | ≥ 95% |
| AC-10 | Ungrounded claims | **0** |
| AC-11 | Raw PII in logs | **0** |
| AC-12 | Responses missing the transparency line or with a blank date | **0** |
| AC-13 | Answers stating a return figure or comparison | **0** |
| AC-14 | Answers recommending or ranking a fund | **0** |
| AC-16 | The 3 examples returning real answers | **3/3** |

Then run the live link check: sample 5% of cited URLs, confirm HTTP 200 **and** that the page contains the cited figure.

```bash
streamlit run src\app.py
```

### 5.10 Cold-start index materialisation (deviation from §5.6)

**Added because `chroma_db/` is gitignored.** It is a derived artefact, so a deploy
that ships the source without the 8.8 MB index folder is the *expected* case rather
than an error — and §5.6's orchestration assumed the folder was there and raised
`FileNotFoundError` on the first question. `retrieval_engine._collection()` now calls
`vector_store.ensure_index()`, which rebuilds from `data/processed/chunks.jsonl` (the
committed source of truth) when the persisted index cannot serve queries.

Two decisions worth recording, because both are counter-intuitive:

**The trigger is a row count, not a missing directory.** `build()` is not atomic: a
process killed mid-upsert leaves a collection that exists, is non-empty, and is
short. An existence check waves that through, and the app then answers from a partial
index — the §3.12 failure arriving through a deployment door instead of a ranking
one, with no error anywhere. A count check covers missing, empty, and partial in one
condition. This is why the brief's framing ("empty or missing") was narrowed.

**It is not a cloud fallback, and the name would have been wrong.** There is no remote
index to fail over to; `chunks.jsonl` is local and committed. Two ways to obtain the
artefact exist — build in CI and ship the folder, or rebuild on first boot — and this
is the second. It cannot help if `chunks.jsonl` is absent, and it does not help
offline: `get_model()` fetches ~90 MB of weights on a cold cache. **That download, not
the 377 embeddings, is the real cold-start cost** — measured at ~60 s to first answer
on a warm cache, so budget Render's boot timeout accordingly or pre-warm the model.

Deliberate differences from `build()`:

| | `build()` (Phase 3) | `ensure_index()` (cold start) |
| --- | --- | --- |
| `verify_alignment` | required | off by default; it re-embeds the whole corpus a second time purely to cross-check Phase 3, doubling cold-start cost |
| `save_npy` | required | best-effort; a read-only or ephemeral filesystem must not stop the app answering, and nothing reads those files at query time |
| Space mismatch | refuses | refuses (via `get_collection`) — a rebuild must never paper over an index built with the wrong metric |

`HDFC_RAG_INDEX_POLICY` selects the behaviour: `auto` (default), `never` (the hard
failure, which is what CI wants — a silent rebuild turns a corrupt-index bug into a
passing build, because the rebuild succeeds from the same file the corruption would
have had to come from), `force`. A typo'd value raises rather than defaulting to
`auto`, which would silently grant the one behaviour an operator was disabling.

**Verified: the rebuild is bit-for-bit identical to the shipped index** —
`max |Δ| = 0.000e+00` over all 377×384 float32 values, identical id order, norms
`[1.000000, 1.000000]`, and the same top-1 `chunk_id` on all 8 §3.12 probes. This is
asserted by `test_rebuild_reproduces_the_shipped_vectors`, because the alternative is
the worst version of this feature: no error, just Phase 5's tuned thresholds silently
calibrated against a different embedding on the deployed host.

Eight tests in `test_rag.py` cover absent / empty / partial / complete / `never` /
typo'd policy / missing `chunks.jsonl` / vector equivalence, against a `tmp_path`
index so the developer's real `chroma_db/` is never touched.

### 5.11 Conversation memory (additive; not in the brief)

`src/app.py` keeps a rolling window of the **last 10 messages** — one user question or
one assistant answer each, so 5 exchanges — in `st.session_state["history"]`. The state
lives in the UI; the rules live in `retrieval_engine.remember()`, which caps it, trims it,
and redacts it. Nothing appends to that list directly.

Keeping the rules in the engine is not tidiness. A window is both **prompt material and a
rendered transcript**, so the two things that must not happen — PII surviving into later
turns, and an answer being sourced from an earlier answer instead of from a retrieved
chunk — are properties of the prompt, not of the widget tree. A UI-local
`history.append(...)` would be invisible to every test that inspects the prompt, which is
where those properties are actually verifiable.

**The window is used for exactly two things**, and the second is fenced:

1. **Scheme carry-over.** "What about its exit load?" names no fund, so `resolve_scheme`
   returns `None`. The corpus holds the answer and the question cannot reach it.
2. **A fenced `EARLIER TURNS` block** in the prompt, for resolving bare references. The
   fence says in terms that the history is not a source of facts and that the `CONTEXT`
   wins any disagreement.

**Carry-over resolves the referent into the query rather than relaxing the confidence
gate.** This is the load-bearing decision, and the measurement forced it: "What about its
exit load?" scores cosine **0.220** against the Flexi exit-load chunk versus **0.829** for
the same question naming the fund, so `MIN_COSINE_SANE` refuses it. Filtering to the right
scheme does *not* rescue it — still 0.220 — because a bare pronoun does not embed near a
fund-specific chunk, while the lexical gate was already passing at 100%. So the inherited
subject is appended as `"(about HDFC Flexi Cap Fund)"` and the bar is applied normally.
Relaxing the bar for inherited queries would be the more convenient change and the more
dangerous one: the bar exists to stop a bad pairing being rendered as a fact, and a bypass
keyed on "this query came from memory" would disable it exactly where there is least
evidence about what is being asked. Naming the subject makes it a well-formed question
again, so it can still be refused.

Three guards on carry-over: it only runs when the current turn is not guard-blocked; an
ambiguous query (two funds named) is already blocked by `guard()` and never reaches it, so
history can never silently pick one of two funds the user named; and it is strictly a
fallback, so a query naming its own fund resolves normally.

**Redaction happens on the way in.** Without it, a PAN typed in turn 1 is re-sent to the
model on turns 2–10 *and* re-shown in the scrollback — `guard()` only ever inspects the
current query, so nothing downstream would catch it. `redact()` is the project's single PII
implementation and its `mode` matters; it is a verified no-op on every answer this pipeline
produces, because R3 already keeps PII out of them. Verified: a PAN in turn 6 is absent
from the stored window, `detect_pii` on the whole window returns `[]`, and the transcript
shows `[REDACTED_PAN]`.

**Trimming drops whole exchanges**, never single messages. A window opening with an
orphaned assistant reply is worse than a shorter one — the model reads it as a claim it
made with no question attached. It costs at most one message of headroom and keeps
user/assistant alternation intact.

**Refusals are stored, flagged, and fenced.** A refused turn is a statement about what the
*corpus* lacks; read as a fact it becomes self-reinforcing, and one declined lock-in
question would make the model decline the next one too. The block labels them
`ASSISTANT (REFUSED)` and says they are not facts and do not make later questions
unanswerable. Verified: a refused non-ELSS lock-in question does not stop the next question
answering, and ELSS's lock-in stays answerable with that refusal in the window.

**Default-off is byte-identical.** `history` defaults to empty, and the empty-history
prompt is asserted byte-identical to the pre-memory text. This is load-bearing rather than
cosmetic: AC-8 is a golden-set accuracy gate tuned against that exact prompt, so a
reflow would move measured accuracy for a feature that is off by default. `test_rag.py`,
the phase gates, and `scripts/eval_gates.py` all call `answer(q)` and keep the verified
Phase 5 numbers.

**Continuity comes from the window, not from the model.** `GROQ_MODEL` is read from `.env`
on every call and is never hardcoded, so a retired model can be swapped without a code
change; `call_groq` reports an unserved model as a 404 naming `GROQ_MODEL` rather than as
answers that quietly stop being generated. But the model id is not what remembers anything —
a swapped model starts a fresh session with no memory of earlier turns. The active model is
surfaced in the sidebar for that reason.

Verified by `tests/test_ui.py` (28 tests, real Streamlit runtime) and 44 engine-level
assertions: window cap and exchange alignment, no PII in the window or the transcript, the
carry-over note shown to the user, the reset button, the `prompt`-level fence, and R1/R2/R7
holding on every history path.

---

## 6. Phase-to-Requirement Traceability

| Brief line / PRD requirement | Phase |
| --- | --- |
| Loading (line 32), FR-01/02/11 | 1 |
| Directory setup, `sources.csv`, `pipeline.yaml` | 1 |
| Chunking decided from data (line 34), §12.4, preview log | 2 |
| 256 word-piece cap, table atomicity, context header | 2 |
| Embedding `all-MiniLM-L6-v2` (line 33) | 3 |
| Local persistent ChromaDB (line 35) | 3 |
| No PII (line 21), R3, AC-4, AC-11 | 4 |
| No performance claims (line 22), R4, AC-13 | 4 |
| No advice / polite refusal + educational link (line 17), R5, R6, AC-5, AC-6, AC-14 | 4 |
| One clear citation link (lines 16–17), R2, AC-2, AC-3 | 5 |
| ≤ 3 sentences (line 23), R1, AC-1 | 5 |
| `Last updated from sources:` (line 23), R7, AC-12 | 5 |
| Tiny UI: welcome + 3 examples + note (line 18) | 5 |
| Retrieval stage (line 31) | 5 |
| Cold-start index materialisation (§5.10, deployment) | 5 |
| Deliverables D1–D5 | 5 (D2 at 1) |

---

## 7. Definition of Done

- [ ] Phase 1 — `documents.jsonl` valid, I1–I5 pass, no empty extracts
- [ ] Phase 2 — no chunk > 256 word-pieces, preview log reviewed by a human, all 7 topic probes found
- [ ] Phase 3 — collection count == chunk count, vectors L2-normalised and row-aligned, survives restart
- [ ] Phase 4 — 0 undetected PII / advice / performance / jailbreak, 0 false positives on fund figures
- [ ] Phase 5 — AC-1…AC-16 green; D1–D5 produced
- [ ] `data/chunks_preview.txt` and the chunking decision (or sweep grid) recorded in the README
- [ ] Graceful degradation documented: which LLM path is active, and that the extractive fallback exists

---

## 8. Build-Time Traps

Ordered by how expensive they are to find late.

| # | Trap | Consequence | Prevention |
| --- | --- | --- | --- |
| 1 | **Word count instead of word-pieces** | Chunks silently truncated; index looks complete, retrieval is blind to the tail | Count with the model tokenizer; assert ≤ 256 |
| 2 | **`vectors.npy` / `ids.json` row misalignment** | Every citation attributes to the wrong scheme; still looks plausible | Assert I6 before writing |
| 3 | **Un-normalised vectors with `hnsw_space: cosine`** | The 0.62 floor silently means something else | Assert I10; keep the two coupled |
| 4 | **JS shell ingested as a document** | Dead citation slot; `count()` looks healthy | Assert I3; check `char_count > 5000` |
| 5 | **Trusting the prompt for R1/R2/R5** | Violations appear only under pressure, in front of a reviewer | Validator in code; the model is untrusted |
| 6 | **PII `account` regex applied to source text** | Legitimate AUM values redacted | Scope detection per call-site; input ≠ output |
| 7 | **Cited URL fabricated by the LLM** | Dead or wrong link; AC-3 failure | `chunk_id` → stored URL only |
| 8 | **Missing `as_of_date`** | R7 line blank; visible defect | 4-level fallback; never guess |
| 9 | **Re-run doubles the collection** | Duplicate chunks, confusing results | `upsert` on `chunk_id`; assert count stable |
| 10 | **Chunk params changed without `--recreate-collection`** | Mixed old/new vectors in one namespace | Recreate on any param change |
| 11 | **Example question triggers a refusal** | First impression is a refusal (AC-16) | Test all 3 before shipping |
| 12 | **Building under Anaconda 3.8** | `ssl` broken; torch/chromadb will not install | Use the Python 3.11 venv (§0.1) |
