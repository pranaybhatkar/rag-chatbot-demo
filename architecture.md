# Architecture — Mutual Fund FAQ RAG Chatbot

| Field | Value |
| --- | --- |
| Document ID | ARCH-MF-RAG-001 |
| Version | 1.0 |
| Status | Ready for implementation |
| Last updated | 2026-09-27 |
| Governs | Implementation of `PRD.md` v1.2. Where the two disagree, the PRD wins and this document is corrected. |
| Mandate | Brief lines 30–36: *"Each stage is different and when we create architecture we want to follow all the stages on RAG (Data ingestion + Data retrieval)"* |

---

## 1. Purpose and Scope

This document specifies the **ingestion half** of the RAG pipeline in implementation-level detail, plus the retrieval path that consumes it:

```
Loading → Chunking → Embedding → Store Vector Data      (brief line 32, mandated order)
```

Covered in full: §5 Loading, §6 Dynamic Chunking, §7 Embeddings, §8 Vector Store, §9 Retrieval.
Referenced but not re-specified: §10 the response validator, which is specified in PRD §4 and is **not** part of this document.

**Design principle carried from the PRD (§1.3):** the pipeline is built so that every hard constraint is verifiable *structurally* rather than by trusting a model. The most important consequence here is §7.4 — the embedding model is never allowed to invent a citation, because it only ever sees chunk IDs.

---

## 2. Pipeline at a Glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│ data/raw/                       ← the ONLY ingestion input. A directory. │
│   <scheme_id>/<doc_type>/*.{html,pdf,txt,md}                            │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  STAGE 1  LOADING      │  walk dir → decode → extract text
                    │  src/ingest.py         │  → section split → as_of_date
                    └───────────┬────────────┘  → content hash → manifest
                                │  data/processed/documents.jsonl
                    ┌───────────▼────────────┐
                    │  STAGE 2  CHUNKING     │  section-aware, table-safe
                    │  src/chunker.py        │  dynamic params (config + sweep)
                    └───────────┬────────────┘  + synthetic context header
                                │  data/processed/chunks.jsonl
                    ┌───────────▼────────────┐
                    │  STAGE 3  EMBEDDING    │  all-MiniLM-L6-v2
                    │  src/embedder.py       │  384-d, mean-pool, L2-norm
                    └───────────┬────────────┘  ⚠ hard 256 word-piece cap
                                │  data/processed/vectors.npy + ids.json
                    ┌───────────▼────────────┐
                    │  STAGE 4  VECTOR STORE │  ChromaDB PersistentClient
                    │  src/store.py          │  data/index/chroma/  (on disk)
                    └───────────┬────────────┘
                                │
              ══════════════════▼══════════════════  index built; serving begins
                                │
                    ┌───────────▼────────────┐
                    │  RETRIEVAL             │  embed query → filter → top-k
                    │  src/retriever.py      │  → MMR → confidence floor
                    └───────────┬────────────┘
                                │  ranked chunks (ids + scores + metadata)
                    ┌───────────▼────────────┐
                    │  VALIDATOR (PRD §4)    │  R1–R8 enforcement ladders
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  RENDER                │  ≤3 sentences + 1 link
                    └────────────────────────┘  + "Last updated from sources:"
```

### 2.1 Stage contracts

Each stage reads a file artifact and writes a file artifact. **No stage calls another stage in-process.** This is what makes the stages independently re-runnable, independently testable, and debuggable — and it is what lets the chunk sweep (§6.5) re-embed without re-parsing.

| Stage | Reads | Writes | Artifact contract |
| --- | --- | --- | --- |
| 1 Loading | `data/raw/**`, `config/sources.csv` | `data/processed/documents.jsonl` | one JSON object per source document |
| 2 Chunking | `data/processed/documents.jsonl`, `config/pipeline.yaml` | `data/processed/chunks.jsonl` | one JSON object per chunk, N:M with documents |
| 3 Embedding | `data/processed/chunks.jsonl` | `data/processed/vectors.npy`, `ids.json` | `float32[N,384]`, row-aligned with `ids.json` |
| 4 Store | vectors + chunks | `data/index/chroma/` | collection `hdfc_faq`, one row per `chunk_id` |

---

## 3. Directory Layout

```
RAG_chatbot_M4/
├── PRD.md                          requirements (v1.2)
├── architecture.md                 this document
├── problemstatement.txt            the milestone brief (source of truth)
├── requirements.txt                pinned deps
├── README.md                       deliverable D3
│
├── config/
│   ├── pipeline.yaml               chunking / embedding / store parameters
│   └── sources.csv                 source manifest: tier, url, doc_type, scheme_id
│
├── data/                           ← all generated. Safe to delete and rebuild.
│   ├── raw/                        ← HAND-PLACED OR FETCHED. Never generated.
│   │   ├── HDFC_LARGE_CAP/
│   │   │   ├── scheme_page/page.html
│   │   │   ├── factsheet/factsheet-2026-06.pdf
│   │   │   └── kim_sid/sid.pdf
│   │   ├── HDFC_FLEXI_CAP/…
│   │   ├── HDFC_ELSS/…
│   │   ├── HDFC_SMALL_CAP/…
│   │   ├── HDFC_BAL_ADV/…
│   │   └── _education/             cross-scheme: investor-education pages
│   ├── interim/
│   │   ├── documents.jsonl
│   │   ├── chunks.jsonl
│   │   ├── vectors.npy
│   │   └── ids.json
│   ├── index/
│   │   └── chroma/                 ← PersistentClient root. THE vector store.
│   └── manifest/
│       ├── ingest_manifest.jsonl   per-file hash, status, error
│       └── run_log.jsonl           append-only stage timings + counts
│
├── src/
│   ├── config.py                   scope allow-list, aliases, thresholds
│   ├── ingest.py                   STAGE 1
│   ├── chunker.py                  STAGE 2
│   ├── embedder.py                 STAGE 3
│   ├── store.py                    STAGE 4
│   ├── retriever.py                retrieval
│   ├── generator.py                answer proposal
│   ├── validator.py                R1–R8 ladders (PRD §4)
│   ├── guardrails.py               R4/R5 intent + lints
│   ├── pii.py                      R3 detect + redact
│   ├── textutils.py                R1 sentence segmentation
│   └── templates.py                PRD §9 response templates
│
├── prompts/system_prompt.v1.txt    versioned prompt contract (PRD §8)
├── app/{api.py, ui.py}             serving surface
├── scripts/
│   ├── build_index.py              runs stages 1→4 in order
│   ├── chunk_sweep.py              PRD §12.4 grid sweep
│   └── make_sample_qa.py           deliverable D4
├── tests/                          per-stage + contract tests
└── deliverables/                   D2, D4, D5
```

**`data/raw/` is the contract boundary of Stage 1.** The brief requires "public sources only"; making the raw directory the single explicit entry point means every ingested byte is auditable on disk, and the corpus can be inspected without running any code.

---

## 4. Stage 0 — Configuration

All tunables live in `config/pipeline.yaml`. No magic numbers in code (PRD NFR-07).

```yaml
corpus:
  raw_dir: data/raw
  allowed_scheme_ids:          # PRD §3.1 closed allow-list (R8)
    - HDFC_LARGE_CAP
    - HDFC_FLEXI_CAP
    - HDFC_ELSS
    - HDFC_SMALL_CAP
    - HDFC_BAL_ADV

chunking:                      # PRD §12.4 — values decided by the sweep, not guessed
  strategy: section_aware
  max_tokens: 224              # ⚠ see §6.2 — MUST stay under the 256 word-piece cap
  overlap_tokens: 48
  hard_ceiling_tokens: 256     # the embedder's truncation point; not negotiable
  min_chunk_tokens: 24         # below this, merge forward — orphan chunks retrieve badly
  add_context_header: true
  never_split_tables: true

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2   # mandated, brief line 33
  dimension: 384
  max_seq_length: 256          # model default; truncates silently
  batch_size: 64
  normalize: true              # L2 — required for cosine == dot product
  pooling: mean
  device: cpu

store:
  backend: chromadb            # mandated, brief line 35
  persist_dir: data/index/chroma
  collection: hdfc_faq
  hnsw_space: cosine
  hnsw_m: 16
  hnsw_ef_construction: 200
  hnsw_ef_search: 128

retrieval:
  top_k: 8
  mmr_lambda: 0.7
  confidence_floor: 0.62       # PRD §7.4
```

---

## 5. Stage 1 — Data Ingestion (Loading from a Directory)

### 5.1 Purpose

Turn a directory of raw public documents into **structured, section-segmented, dated, hashed** document records — without losing the provenance needed to cite them.

### 5.2 Input

- `data/raw/**` — the directory tree (§3)
- `config/sources.csv` — the manifest binding each URL to `scheme_id`, `doc_type`, and `tier` (PRD §7.2)

```
tier,scheme_id,doc_type,title,url
broker,HDFC_LARGE_CAP,scheme_page,HDFC Large Cap Fund (Direct Growth),https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
official,HDFC_ELSS,elss_lockin_rule,ELSS lock-in & 80C rules,https://www.hdfcmf.com/...
official,,investor_education,HDFC MF FAQs,https://www.hdfcmf.com/faqs
```

The **directory path encodes `scheme_id` and `doc_type`**, and the manifest supplies `url`, `title`, and `tier`. Neither is derivable from the other, so both are required and a mismatch is a hard error.

### 5.3 Process

**Step 1 — Walk and validate.** Recursive walk of `data/raw/`. For each file: parse the path into `(scheme_id, doc_type)`, verify `scheme_id ∈ allowed_scheme_ids`, look the file up in the manifest, and verify the manifest row's `doc_type` agrees with the directory. Reject with a logged error rather than skipping silently — a silently-dropped source is a silently-broken citation.

**Step 2 — Decode by extension.**

| Extension | Extractor | Notes |
| --- | --- | --- |
| `.html`, `.htm` | `bs4` → strip `script`/`style`/`svg`/`iframe`/`nav`/`footer`, then parse the embedded `__NEXT_DATA__` payload for the scalar facts | **MEASURED:** the prose IS server-rendered (17k–44k chars/page), but expense ratio, riskometer, min SIP and benchmark exist only in the payload. A plain strip yields the label with no value. Both layers are required |
| `.pdf` | `pypdf`, falling back to `pdfplumber` for tables | Factsheets are table-heavy |
| `.txt`, `.md` | passthrough + normalisation | |

**Step 3 — Assert non-empty extraction.** If the extracted text is under a floor (200 chars) or has no alphabetic content, **fail the file loudly** and record it in `ingest_manifest.jsonl` with `status: "empty_extract"`. This is the single most important guard in Stage 1: a JS-rendered page that yields nothing would otherwise become an empty chunk and a dead citation slot that looks healthy in the index.

**Step 4 — Segment into sections.** Split on heading elements (`h1`–`h4`) or, for PDFs, on layout-detected headings and page boundaries. Each section keeps its heading path (e.g. `Fees and charges > Exit load`). These map 1:1 to the brief's FAQ topics and are what make chunks citable and retrievable.

**Step 5 — Extract `as_of_date`.** Per the PRD Q6 resolution, in strict precedence:
1. A date in the document's own "as of" / "last updated" line.
2. The HTTP `Last-Modified` header recorded at fetch time.
3. The file mtime.
4. Otherwise `null` — which the UI must surface as a flagged unknown, never silently as today.

`as_of_date` feeds R7's `Last updated from sources:` line, so a wrong value is a visible defect, not a silent one.

**Step 6 — Hash and record.** `sha256` of the raw bytes → `source_hash`. Enables idempotent re-ingest and drift detection.

**Step 7 — Write.** Append one record per document to `data/processed/documents.jsonl`.

### 5.4 Output schema

```json
{
  "doc_id": "HDFC_SMALL_CAP__scheme_page__a1b2c3d4",
  "scheme_id": "HDFC_SMALL_CAP",
  "scheme_name": "HDFC Small Cap Fund",
  "plan": "Direct Growth",
  "doc_type": "scheme_page",
  "source_tier": "broker",
  "title": "HDFC Small Cap Fund (Direct Growth)",
  "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
  "as_of_date": "2026-07-15",
  "as_of_date_source": "document_text",
  "source_hash": "sha256:9f2c…",
  "raw_path": "data/raw/HDFC_SMALL_CAP/scheme_page/page.html",
  "sections": [
    { "heading": "Fees and charges", "level": 2, "text": "…", "order": 7 }
  ],
  "char_count": 18422,
  "ingested_at": "2026-09-27T14:02:11Z"
}
```

### 5.5 Invariants and failure modes

| # | Invariant | On violation |
| --- | --- | --- |
| I1 | Every file in `raw/` maps to exactly one manifest row | Hard fail, logged |
| I2 | `scheme_id` always in the allow-list | Hard fail — this is R8's enforcement point at ingest |
| I3 | Extracted text ≥ 200 chars with alphabetic content | Hard fail, `status: "empty_extract"` |
| I4 | `url` present and non-empty for every document | Hard fail — R2 cannot cite a document with no URL |
| I5 | `as_of_date` is `null` or a valid ISO date; never a guess | Log a warning; the UI shows "date unavailable" |

| Failure mode | Symptom | Handling |
| --- | --- | --- |
| JS-rendered page yields a shell | `char_count` ≈ 0 | Caught by I3. Re-fetch with a headless renderer and re-save into `raw/` |
| Figures absent although the page looks fine | `Expense ratio (TER)` missing from a document | Caught by the fact-completeness assertion in Stage 1. Cause is almost always `<script>` stripping, not a source gap — check the payload |
| Anti-bot / rate limiting | HTTP 429/403 | Back off, honour `Retry-After`, serialise fetches. Respect `robots.txt` and the site's terms — the corpus is for a private milestone, not redistribution |
| PDF text layer absent | Garbled or empty | Scanned PDF: OCR, or replace with the HTML source |
| Site restructure | 404 on a manifest URL | Fail that file; surface in `ingest_manifest.jsonl`; re-point the manifest row. **Never** silently substitute a different URL — that would break R2's guarantee that the citation is the document the answer came from |

### 5.6 Fetching into `raw/`

Stage 1 reads the directory; it does not fetch. Fetching is a separate, explicit, auditable step so that the corpus is reproducible:

```bash
python scripts/fetch_sources.py --config config/sources.csv --out data/raw
```

Each fetched file is written with the fetched date and HTTP headers recorded in a sidecar `.meta.json`, so `as_of_date` provenance survives to Stage 3 even for pages with no visible date.

---

## 6. Stage 2 — Dynamic Chunking

### 6.1 Purpose

Split documents into retrievable, citable units **without exceeding what the embedder can actually represent**, and choose the split parameters from evidence rather than intuition (PRD §12.4).

### 6.2 The binding constraint: 256 word-pieces

From the `all-MiniLM-L6-v2` model card:

> *"By default, input text longer than **256 word pieces** is truncated."*

Consequences, which drive this entire stage:

1. **The token unit is a word-piece, not a word.** ~1.3 word-pieces per English word, so 256 word-pieces ≈ **190 words**. A naive `len(text.split())` count overstates capacity by ~30% and will silently truncate.
2. **Truncation is silent.** No exception, no warning — the tail simply never reaches the vector. A 600-word-piece chunk is embedded as its first 256 word-pieces, so retrieval is blind to everything after that point while the stored `documents` text still shows the full chunk. The index looks complete and is not.
3. **Count word-pieces with the model's own tokenizer**, never an approximation:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
n = len(tok(chunk_text, add_special_tokens=True)["input_ids"])
```

4. **Operational ceiling is 224 word-pieces**, not 256. The 32-piece margin absorbs the synthetic context header (§6.4) and tokenizer drift between the chunker and the embedder.

> ⚠ **Conflict with PRD §12.4, flagged not silently resolved.** The §12.4 grid is `{250, 400, 600, 800}` × overlap, with an 800-token hard ceiling. Measured in the model's word-pieces, every option above 256 is truncated and therefore invalid. The grid must be re-expressed in word-pieces as `{128, 192, 224, 256}` and the ceiling set to 256. §6.5 specifies the corrected sweep. **PRD §12.4 needs this one-line amendment before the sweep is run** — recorded in §15.

### 6.3 Process

1. **Start from sections, not the document.** Chunks never span a section boundary. The brief's topics (expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, statements) map onto sections, so section-aligned chunks are both more retrievable and more precisely citable.
2. **Accumulate sections** into a chunk until adding the next would exceed `max_tokens` word-pieces.
3. **Never split a table.** A table is atomic: it travels as one chunk, with its header row repeated if it must be carried. Splitting an exit-load slab produces a chunk that looks like a fact and is nonsense (PRD §7.3).
4. **Merge orphans.** A chunk under `min_chunk_tokens` (24) is merged into its neighbour; sub-threshold chunks retrieve poorly and produce weak citations.
5. **Emit overlap** of `overlap_tokens` word-pieces between adjacent chunks from the same section, to avoid splitting a fact across a boundary.
6. **Prepend the synthetic context header** (§6.4).
7. **Assert the ceiling.** Any chunk over `hard_ceiling_tokens` is a hard error, not a warning — it is a bug, because it means the accumulator failed.

### 6.4 The synthetic context header

Every chunk is prefixed with a single self-describing line (PRD §7.3):

```
HDFC ELSS Tax Saver Fund (Direct Growth) | scheme_page | as of 2026-07-15 | Fees and charges
```

It makes chunks self-describing for both the embedder and the citation, measurably improves retrieval, and costs ~20 word-pieces — which is why §6.2 reserves headroom for it. The header is **not** counted as answer text; the generator strips it before composing a response.

### 6.5 Dynamic chunk sizing — the sweep (PRD §12.4)

The brief defers the strategy to the data. `scripts/chunk_sweep.py` executes it:

**Step 1 — Characterise the corpus.** Report the word-piece length distribution of sections, the tabular-to-prose ratio, and heading density. These pages are metadata-dense, not prose-heavy — that fact is what should drive the decision.

**Step 2 — Sweep the grid.**

| Axis | Values (word-pieces) |
| --- | --- |
| `max_tokens` | 128, 192, 224, 256 |
| `overlap_tokens` | 0, 32, 48, 64 |

16 configurations. All other variables held fixed. Each configuration is embedded and indexed into a **throwaway Chroma collection** (`hdfc_sweep_<size>_<overlap>`), so the sweep never contaminates the production index.

**Step 3 — Score** against a stratified 40-question probe set (8 per scheme, spanning all six brief topics) on exactly two metrics:
- **Citation precision** — does the top-1 retrieved chunk actually contain the answer? (Primary.)
- **Self-containment** — how often the generator must reach for a second chunk. (Tie-break.)

**Step 4 — Select** the maximum-citation-precision configuration; break ties on self-containment. **Write the full grid and the winning row into the README** (deliverable D3). A decision without evidence is not a decision.

**Escalation rule.** If the best configuration still yields citation precision < 0.90, **do not tune further** — the bottleneck is `all-MiniLM-L6-v2`'s numeric recall, not the chunk size. Escalate to PRD Q4 and add a BM25 keyword leg alongside the mandated embedder. Chunk tuning cannot fix a model that under-weights `0.55%` against `0.58%`.

### 6.6 Output schema

```json
{
  "chunk_id": "HDFC_SMALL_CAP__scheme_page__c07",
  "doc_id": "HDFC_SMALL_CAP__scheme_page__a1b2c3d4",
  "scheme_id": "HDFC_SMALL_CAP",
  "doc_type": "scheme_page",
  "source_tier": "broker",
  "as_of_date": "2026-07-15",
  "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
  "section": "Fees and charges",
  "heading_path": ["Fees and charges", "Expense ratio"],
  "ordinal": 7,
  "text": "HDFC Small Cap Fund (Direct Growth) | scheme_page | as of 2026-07-15 | Fees and charges\nThe expense ratio …",
  "header": "HDFC Small Cap Fund (Direct Growth) | scheme_page | as of 2026-07-15 | Fees and charges",
  "body": "The expense ratio …",
  "n_wordpieces": 198,
  "is_table": false,
  "chunk_params": { "max_tokens": 224, "overlap_tokens": 48, "strategy": "section_aware" }
}
```

`chunk_id` is `<scheme_id>__<doc_type>__<ordinal>` — deterministic, so re-running the pipeline produces identical IDs and Stage 4 upserts are idempotent.

---

## 7. Stage 3 — Vector Embeddings

### 7.1 Mandated model

`sentence-transformers/all-MiniLM-L6-v2` (brief line 33). Verified properties from the model card:

| Property | Value |
| --- | --- |
| Output dimension | **384** |
| Max sequence length | **256 word-pieces**, silent truncation beyond |
| Trained sequence length | 128 — quality degrades past this before hard truncation |
| Pooling | Mean pooling over the attention mask |
| Normalisation | L2 (the reference implementation applies it) |
| Similarity | Cosine — and because vectors are L2-normalised, **cosine == dot product** |
| Size / licence | 22.7M params, Apache-2.0 |

### 7.2 Why L2 normalisation is a hard requirement, not a default

`store.hnsw_space: cosine` and dot-product ANN indexing must agree, or scores are silently wrong in a way that looks plausible. With L2-normalised vectors, `cosine(a,b) = a·b`, so the HNSW index and the reported score refer to the same quantity. **If normalisation is ever disabled, `hnsw_space` must change to `l2` in the same commit** — they are a coupled pair.

### 7.3 Process

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # ~80 MB, CPU-only is fine
model.max_seq_length = 256          # make the truncation limit explicit, not inherited

def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 — required, see §7.2
        show_progress_bar=False,
    )
    assert vecs.shape[1] == 384, vecs.shape
    return vecs.astype(np.float32)
```

1. **Embed `text`, not `body`.** The synthetic context header is included — it is part of what makes the chunk retrievable. This is why §6.2 reserves headroom for it.
2. **Batch 64.** The corpus is small (hundreds of chunks); the entire set embeds in seconds on CPU. Batching mainly guards against re-invoking the model per chunk.
3. **Deterministic.** `model.eval()`, no dropout, fixed weights. The same chunk always yields the same vector — required for NFR-06 reproducibility.
4. **Truncation assertion.** Count word-pieces per chunk and **hard-fail** on any chunk over 256. This turns §6.2's silent failure into a loud one, and catches a chunker bug before it reaches the index.
5. **Persist** to `data/processed/vectors.npy` (`float32[N,384]`) plus `ids.json` (row-aligned `chunk_id` list). Row `i` of the array corresponds to `ids[i]` — Stage 4 depends on this alignment.

### 7.4 The model never sees a URL

Only `chunk_id` and chunk text are embedded. **The LLM and the embedder are both structurally incapable of producing a citation URL** — the URL is resolved from the stored `chunk_id` metadata in Stage 4 (PRD §4.2). Fabricated citations are impossible by construction, not by instruction.

### 7.5 Known limitation and mitigation

`all-MiniLM-L6-v2` is a general-purpose English sentence encoder. It is **weaker on numeric discrimination than on semantics**, and this corpus is full of near-identical numbers — five schemes each with their own expense ratio, exit-load slab, and minimum investment. Two mitigations, both in the retrieval path (§9):

1. **Hard `scheme_id` metadata filter** before search, so the five schemes never compete in one result set.
2. **Exact-match boosting** for numeric tokens from the query.

Neither replaces the model. If numeric accuracy still misses AC-8, PRD Q4 escalates to a hybrid BM25 leg.

---

## 8. Stage 4 — Vector Database Storage (Local Persistent ChromaDB)

### 8.1 Mandated backend

ChromaDB, `PersistentClient` (brief line 35). **Local and persistent** — the index lives on disk under `data/index/chroma/` and survives process restarts. No server, no Docker, no cloud dependency, satisfying PRD NFR-02 (no third-party data egress).

### 8.2 Process

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="data/index/chroma")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="hdfc_faq",
    embedding_function=ef,
    metadata={
        "hnsw:space": "cosine",       # pairs with normalised vectors (§7.2)
        "hnsw:M": 16,
        "hnsw:ef_construction": 200,
        "hnsw:ef_search": 128,
    },
)
```

**Writes are upserts keyed on `chunk_id`.** Re-running the pipeline updates in place instead of duplicating, so Stage 4 is safely re-runnable:

```python
collection.upsert(
    ids=chunk_ids,
    embeddings=vectors.tolist(),      # float32[N,384]
    documents=[c["text"] for c in chunks],
    metadatas=[chunk_metadata(c) for c in chunks],
)
```

### 8.3 Metadata stored per chunk

Chroma metadata values must be scalars or arrays of scalars — no nested objects. So `heading_path` and `chunk_params` are serialised to JSON strings rather than stored as dicts.

| Field | Type | Used by |
| --- | --- | --- |
| `scheme_id` | str | R8 hard filter, alias resolution |
| `scheme_name` | str | citation label |
| `plan` | str | Direct Growth scope guard (PRD §3.2b) |
| `doc_type` | str | R4 factsheet targeting, refusal links |
| `source_tier` | str | R2 citation preference: `official` beats `broker` |
| `title` | str | citation label |
| `url` | str | **the citation** — resolved from here, never generated |
| `as_of_date` | str | R7 `Last updated from sources:` |
| `section` | str | display + trace |
| `heading_path_json` | str | trace/debug |
| `n_wordpieces` | int | §7.3 truncation assertion |
| `is_table` | bool | chunking integrity |
| `chunk_params_json` | str | NFR-06 reproducibility |
| `content_hash` | str | drift detection |

### 8.4 Invariants

| # | Invariant | On violation |
| --- | --- | --- |
| I6 | `len(vectors) == len(ids) == len(metadatas)` | Hard fail before write — misalignment here silently mis-cites every answer |
| I7 | Every `chunk_id` unique in the batch | Hard fail |
| I8 | Every `url` non-empty and in the `sources.csv` allow-list | Hard fail — R2's citation allow-list |
| I9 | Every `scheme_id` in the corpus allow-list | Hard fail — R8 |
| I10 | Every row `‖v‖₂ == 1.0` (±1e-5) | Hard fail — §7.2 |
| I11 | Every `n_wordpieces ≤ 256` | Hard fail — §6.2 |

I6 and I10 are worth stating explicitly: both produce a *plausible-looking but wrong* system rather than an obvious crash. A one-row offset in `ids.json` would attribute every citation to the wrong scheme, and un-normalised vectors would make the confidence floor in §9.3 mean something other than it appears to.

### 8.5 Reindex and teardown

```bash
# full rebuild — safe, idempotent
python scripts/build_index.py --stages 1,2,3,4

# swap in new chunk params after the sweep
python scripts/build_index.py --stages 2,3,4 --recreate-collection

# corpus stats
python scripts/store_stats.py
```

`--recreate-collection` drops and rebuilds the collection. Required after any change to `max_tokens`, `overlap_tokens`, or `hnsw:*` parameters, because those are baked into the persisted index and a plain upsert would leave a mix of old and new vectors under one namespace.

### 8.6 Persistence notes

- The Chroma directory is **binary and version-coupled.** Never hand-edit it; rebuild instead.
- `data/index/chroma/` must be in `.gitignore` — it is large, generated, and reproducible from `data/raw/`.
- Backing up `data/raw/` + `config/sources.csv` + `config/pipeline.yaml` is sufficient to reproduce the entire index. That trio is the real source of truth.

---

## 9. Retrieval Path

Specified here because it determines what Stages 2–4 must store.

### 9.1 Process

```python
res = collection.query(
    query_embeddings=[qvec.tolist()],
    n_results=TOP_K,
    where={"scheme_id": {"$in": candidate_scheme_ids}},   # R8 hard filter
)
```

1. **Resolve scheme intent** → alias map (`hdfc equity` → `HDFC_FLEXI_CAP`, PRD §3.3). Ambiguous input ("HDFC funds") → scope question, not a search.
2. **Plan-scope guard.** Non-Direct-Growth questions route to refusal — the corpus does not cover them (PRD §3.2b).
3. **Embed the query** with the *same* model and the *same* normalisation.
4. **Hard-filter by `scheme_id`** when intent is resolved, else all five.
5. **Fetch top-k = 8.**
6. **MMR re-rank**, λ = 0.7, for diversity across sections.
7. **Numeric exact-match boost** (§7.5).
8. **Confidence floor.**

### 9.2 Fallback ladder

| Condition | Action |
| --- | --- |
| No scheme intent | search all 5 |
| Scheme resolved | filter to that scheme; if empty, widen to all 5 and note it in the trace |
| Zero results | grounded refusal |
| Top score < 0.62 | grounded refusal (R2 rung 4) |
| Top-2 chunks disagree on a number | do not answer; surface the conflict with the higher-confidence link (PRD §7.4) |

### 9.3 Score semantics

With L2-normalised vectors and `hnsw:space: cosine`, Chroma returns `(1 - cosine_distance)`, so the reported score is the cosine similarity directly comparable to `confidence_floor: 0.62`. **This equivalence is an invariant of the §7.2 pairing** — if normalisation is disabled, the floor must be recalibrated against the new score scale, or the system will refuse everything (or nothing).

---

## 10. Validator Boundary

Not detailed here — specified in PRD §4. The architectural point: **the validator is a separate process from the pipeline.** Retrieval emits ranked chunks; the validator independently re-derives sentence count, citation, and `as_of_date` and may override the generator entirely. The pipeline cannot bypass it, because nothing reaches the render path without passing through it.

---

## 11. End-to-End Run Order

```bash
# 0. one-time
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# 1. place or fetch sources
python scripts/fetch_sources.py --config config/sources.csv --out data/raw
#    → verify data/raw/ by eye; this is the audit surface

# 2. build
python scripts/build_index.py --stages 1,2,3,4

# 3. (once) decide chunking from the data — PRD §12.4
python scripts/chunk_sweep.py --grid 128,192,224,256 --overlap 0,32,48,64
#    → record the winning row in config/pipeline.yaml and in the README

# 4. rebuild with the chosen params
python scripts/build_index.py --stages 2,3,4 --recreate-collection

# 5. gates
python -m pytest tests/ -q

# 6. serve
streamlit run src/app.py
```

Stages are individually addressable, so a chunking change costs a re-run of stages 2–4 only — seconds, not a re-parse.

---

## 12. Observability

Every stage appends to `data/manifest/run_log.jsonl`:

```json
{"ts":"2026-09-27T14:06:02Z","stage":"embed","model":"all-MiniLM-L6-v2",
 "dim":384,"n_chunks":412,"max_wordpieces":231,"batches":7,"elapsed_s":11.4,
 "params":{"max_tokens":224,"overlap_tokens":48}}
```

Per-request traces record `chunk_id`s, scores, `source_tier`, `as_of_date`, prompt version, and which validator rung fired (PRD NFR-05). **All logging passes through the R3 redaction filter** — no raw user query text, no PII (AC-11).

---

## 13. Testing Strategy

| Suite | Scope | Key assertions |
| --- | --- | --- |
| `test_ingest.py` | Stage 1 | I1–I5; empty-extract detection fires on a JS shell; every manifest row yields a document |
| `test_chunker.py` | Stage 2 | No chunk > 256 word-pieces; no table split; orphans merged; chunk IDs deterministic across runs |
| `test_embedder.py` | Stage 3 | Output is `[N,384]` `float32`; `‖v‖₂ == 1`; identical input → identical vector; over-length chunk hard-fails |
| `test_store.py` | Stage 4 | I6–I11; upsert is idempotent (run twice → same count); vector/ID/metadata alignment survives a round-trip |
| `test_retriever.py` | Retrieval | `scheme_id` filter excludes out-of-scope schemes; floor refuses; MMR returns diverse sections |
| `test_contract.py` | End-to-end | AC-1, AC-2, AC-12, AC-13 = 0 violations over the probe set |

---

## 14. Design Decisions and Alternatives Rejected

| Decision | Alternative rejected | Why |
| --- | --- | --- |
| Section-aware chunking, never spanning headings | Fixed-size character windows | Keeps chunks aligned to the brief's FAQ topics; a window that straddles "Exit load" and "Riskometer" cites neither precisely |
| Hard ceiling at 256 word-pieces | 800-token ceiling from PRD §12.4 | Beyond 256 the embedder truncates silently (§6.2) |
| Operate at 224, not 256 | Use the full 256 | Leaves headroom for the context header and tokenizer drift; the model was trained at 128 anyway |
| Chunk-ID → stored-URL citation | Let the LLM emit the URL | Makes fabricated citations structurally impossible (PRD §4.2) |
| Files as stage interfaces | In-process function calls | Makes the chunk sweep re-runnable without re-parsing, and each stage testable alone |
| `raw/` as the only ingestion input | Fetch inside the loader | Every ingested byte is auditable on disk; the corpus is reproducible and inspectable |
| Explicit `n_wordpieces` per chunk | Trusting the chunker | Turns the silent 256-truncation into a hard, visible failure |
| Upsert on `chunk_id` | Delete-then-insert | Idempotent re-runs; a partial failure cannot empty the index |
| Read from `raw/`, fetch separately | Loader fetches | A fetch failure is visible as a missing file rather than as a silently shorter corpus |

---

## 15. Constraint Register

Items requiring action outside this document.

| # | Item | Impact | Action |
| --- | --- | --- | --- |
| **A-1** | **PRD §12.4 grid `{250,400,600,800}` exceeds the embedder's 256 word-piece limit** | Every option above 256 is silently truncated; the sweep would measure truncated embeddings and could select a configuration whose vector never represented most of the chunk | **Amend §12.4 to `{128,192,224,256}` and set the hard ceiling to 256** before running the sweep. Corrected procedure is in §6.5; the change is a one-line PRD edit |
| A-2 | Tier-2 official corpus not yet collected | Refusal links and riskometer/lock-in/exit-load facts have no source | Populate `data/raw/` per PRD §7.2; `scripts/fetch_sources.py` supports it once `sources.csv` is extended |
| A-3 | `all-MiniLM-L6-v2` numeric recall | Risks AC-8 on expense-ratio and exit-load questions | `scheme_id` filter + numeric boost (§9.1); escalate to PRD Q4 (BM25) if precision < 0.90 |
| A-4 | LLM availability undecided (PRD Q2) | Generation needs a model; no key or local server confirmed | Retrieval, chunking, embedding, and storage are **independent of the LLM** and can be completed now; wire the generator once Q2 is settled |
| A-5 | Environment: Anaconda 3.8 has a broken `ssl` module | `torch`/`chromadb` cannot install under it | Use the Python 3.11.9 install at `%LOCALAPPDATA%\Programs\Python\Python311` in a venv (PRD §14.6) |

---

## 16. Traceability to PRD and Brief

| Requirement | Source | Specified in |
| --- | --- | --- |
| One AMC, 5 schemes, hard allow-list | brief line 7; PRD §3.1, R8 | §3, §4 (`allowed_scheme_ids`), §5.3 I2, §8.4 I9, §9.1 |
| Loading → Chunking → Embedding → Store | brief line 32 | §2 diagram, §2.1 contracts |
| Loading raw data from a directory | user requirement | §3 (`data/raw/`), §5 |
| Dynamic chunking decided from the data | brief line 34; PRD §12.4 | §6.2, §6.5 |
| `all-MiniLM-L6-v2` embeddings | brief line 33 | §7.1–§7.3 |
| Local persistent ChromaDB | brief line 35 | §8.1–§8.2 |
| Corpus doc types (factsheets, KIM/SID, scheme FAQs, fee/charges, riskometer, statement/tax guides) | brief line 8; PRD §7.2 | §5.2 manifest `doc_type` |
| Synthetic context header | PRD §7.3 | §6.4 |
| Never split tables | PRD §7.3 | §6.3 step 3 |
| Chunk metadata schema | PRD §7.1 | §6.6, §8.3 |
| `source_tier` official/broker | PRD §7.2, C-1 | §5.4, §8.3 |
| `as_of_date` derivation | PRD Q6 resolution | §5.3 step 5 |
| Confidence floor 0.62, top-k 8, MMR 0.7 | PRD §7.4 | §4, §9.1–§9.3 |
| Citation from stored URL, never generated | PRD §4.2 | §7.4, §8.3, §8.4 I8 |
| Reproducibility (NFR-06) | PRD NFR-06 | §6.6 (`chunk_params_json`), §12 |
| No PII in logs (AC-11) | PRD R3 | §12 |
