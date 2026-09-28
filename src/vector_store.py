"""vector_store.py — Stage 3: embedding + local persistent ChromaDB.

Reads `data/processed/chunks.jsonl`, embeds every chunk with
``all-MiniLM-L6-v2``, and upserts into a local, on-disk ChromaDB collection.

Three things in here are not optional, and each guards a failure that produces a
*plausible-looking but wrong* system rather than a crash:

1. **Normalised vectors + a cosine index are a coupled pair.** With L2-normalised
   vectors, ``cosine(a,b) == a·b``, so the ANN index and the reported score are
   the same quantity — which is what makes the ``0.62`` confidence floor in
   Phase 5 mean what it appears to mean. Un-normalised vectors against a cosine
   index do not crash; they make the floor silently meaningless. Asserted, not
   assumed (I10).

2. **A one-row offset between vectors and ids attributes every citation to the
   wrong scheme** — and every answer still looks fine, because the text is
   plausible and the URL resolves. This is the most expensive bug in the project
   to find later and the cheapest to prevent here. ``verify_alignment()``
   therefore checks the *persisted* index against a **fresh re-embedding** of the
   chunk text, which is a genuinely independent derivation. Comparing the stored
   array against the in-memory array would pass even when both are wrong, which
   is the entire hazard.

3. **The model silently truncates past 256 word-pieces.** Every chunk is
   measured with the same tokenizer used in Phase 2 *before* embedding, so an
   overflow is a loud failure instead of a truncated vector.

Credentials: ``.env`` is loaded and validated here even though Phase 3 does not
use the LLM. Proving the credential path works during the build — rather than at
the moment a user is waiting on an answer — is worth the four lines. The key is
never printed, logged, or stored; only its presence and length are reported.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import count_wp, load_from_hub_cache, load_tokenizer
from src.config import ALLOWED_SCHEME_IDS, ALLOWED_URLS, EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
#: Local, on-disk. `PersistentClient` is the only Chroma client that survives a
#: restart; an in-memory client looks identical until the process exits.
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "hdfc_faq"
VECTOR_DIM = 384
#: Explicit, never inherited. The sentence-transformers default for this model is
#: 256, but relying on a default means a library upgrade can silently change the
#: truncation point of every vector already in the index.
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 64

_MODEL = None


# ── credentials ───────────────────────────────────────────────────────────

#: Where a credential was found, named so a misconfigured deploy is
#: diagnosable from the log without ever printing the value.
SRC_SECRETS = "st.secrets"      # a .streamlit/secrets.toml entry
SRC_ENV = "environment"         # a real exported process variable
SRC_DOTENV = ".env"             # the local development file
SRC_MISSING = "MISSING"

#: The variables the generator needs. Embedding uses neither of them.
CREDENTIAL_VARS = ("GROQ_API_KEY", "GROQ_MODEL")


@dataclass(frozen=True)
class Credentials:
    """Presence and shape of the required credentials. Never the values."""

    api_key: str
    model: str
    key_source: str
    model_source: str

    @property
    def ok(self) -> bool:
        return bool(self.api_key) and bool(self.model)


def _secrets_file_present() -> bool:
    """True if a ``.streamlit/secrets.toml`` exists where Streamlit would look.

    This guard exists because reading ``st.secrets`` when there is no secrets
    file is not a quiet no-op. Streamlit renders "No secrets found. Valid paths
    for a secrets.toml file or secret directories are: ..." into the page, so an
    unguarded lookup on a Community Cloud deploy - which has no secrets.toml,
    because the dashboard injects environment variables instead - puts a red
    error banner in front of every user who loads the app. Caught by
    test_refusals_render_cited_dated_and_without_advice, which asserts the app
    renders no errors.

    Both the working directory and the repository root are searched, walking
    up to the filesystem root, because Streamlit searches upward from the
    working directory and the two are not always the same folder.

    Limitation, stated rather than hidden: this does not cover Streamlit's
    ``secrets_dir`` config option, under which secrets live in an arbitrary
    directory. That is not a documented Streamlit configuration and nothing in
    this project uses it, but a deployment that adopted it would fall through
    to the environment tier - which is where Community Cloud's secrets come from
    anyway, so it would still work.
    """
    seen: list[Path] = []
    for base in (Path.cwd(), ROOT):
        try:
            base = base.resolve()
        except OSError:
            continue
        if base not in seen:
            seen.append(base)
    for base in seen:
        for directory in (base, *base.parents):
            try:
                if (directory / ".streamlit" / "secrets.toml").is_file():
                    return True
            except OSError:
                continue
    return False


def _streamlit_secret(name: str) -> str:
    """Read one name from ``st.secrets``, or return "" when it is not there.

    In Streamlit 1.42.2 this reads ``.streamlit/secrets.toml`` and nothing else.
    It does **not** fall back to the process environment: with no secrets file
    present, ``st.secrets[name]`` raises ``FileNotFoundError`` even when the
    variable is sitting in ``os.environ`` (measured, not assumed). That is the
    whole reason the environment tier in :func:`load_credentials` is not
    redundant.

    The file-existence guard runs first - see :func:`_secrets_file_present` for
    why touching ``st.secrets`` blind is not safe.

    Deliberately total, because three callers reach this and none of them can
    assume a Streamlit runtime is live:

      * ``app.py`` under ``streamlit run``, where a ScriptRunContext exists.
      * ``python -m src.vector_store`` and the pytest suite, where no runtime
        exists at all and ``st.secrets`` raises on first touch.
      * any future caller that is not a Streamlit app.

    A missing key, an absent secrets.toml, an unimportable Streamlit and a
    half-initialised context therefore all collapse to one answer: not found,
    fall through to the environment. Streamlit is imported here rather than at
    module scope so that ``vector_store`` stays importable, and cheap, without
    it - Phase 3 embedding has no business pulling in a web framework.
    """
    if not _secrets_file_present():
        return ""
    try:
        import streamlit as st
    except Exception:
        return ""
    try:
        value = st.secrets[name]
    except Exception:
        # KeyError when the name is absent, FileNotFoundError or
        # StreamlitSecretNotFoundError when the file went away between the guard
        # and the read. None is worth propagating: a missing credential is a
        # degraded mode that load_credentials reports, not a crash.
        return ""
    return str(value) if value else ""


def load_credentials(dotenv_path: Path | None = None, *,
                     secrets_reader=None) -> Credentials:
    """Resolve the generator credentials, in precedence order.

      1. ``st.secrets``   - a ``.streamlit/secrets.toml`` entry, for a checkout
                           that keeps its secrets in the Streamlit convention.
      2. the environment - a genuinely exported variable. **This is the tier
                           Community Cloud actually resolves through.** The
                           dashboard does not write a secrets.toml; it injects
                           the value as an environment variable, and Streamlit
                           1.42.2's ``st.secrets`` does not read the environment
                           (measured: FileNotFoundError with the variable set).
                           So an implementation that used only ``st.secrets``
                           would find nothing there, and one that used only
                           ``st.secrets`` would be untestable off-platform.
      3. ``.env``, which lands in the environment via ``load_dotenv`` and is
         therefore the same code path as tier 2 - which is exactly what makes
                           one implementation serve local dev and a hosted
                           deploy unchanged.
      4. nothing          - reported, never raised, so the app still starts and
                           still answers from the index without a generator.

    Reading the source *accurately* is the reason for the non-destructive read
    below. ``load_dotenv`` hands back the values it parsed whether or not it
    actually set them, so it cannot say where a variable came from. Sampling
    ``os.environ`` before and after the load also looks like it works, and does
    report correctly on the first call - but the first ``load_dotenv`` leaves its
    values in the process for the rest of the run, so every later call reports
    the same credential as coming from the environment when it came from the
    file. ``dotenv_values`` reads the file without mutating anything, so the
    answer is the same on every call.

    ``secrets_reader`` exists so this precedence is testable without a Streamlit
    runtime, and so a test can assert "no credential from any source" by
    neutralising all the sources at once rather than one at a time.
    """
    read_secrets = (secrets_reader if secrets_reader is not None
                    else _streamlit_secret)
    path = dotenv_path or (ROOT / ".env")

    from dotenv import dotenv_values, load_dotenv

    file_vals = dotenv_values(path) if path.is_file() else {}

    # Still called, so the variables stay available to anything else in the
    # process that reads os.environ directly. The label below does not depend
    # on it, which is the point.
    load_dotenv(path, override=False)

    def _get(name: str) -> tuple[str, str]:
        val = read_secrets(name)
        if val:
            return val, SRC_SECRETS
        val = os.environ.get(name, "")
        if not val:
            return "", SRC_MISSING
        return val, (SRC_DOTENV if file_vals.get(name) == val else SRC_ENV)

    key, key_src = _get("GROQ_API_KEY")
    model, model_src = _get("GROQ_MODEL")
    return Credentials(api_key=key, model=model,
                       key_source=key_src, model_source=model_src)


def report_credentials(creds: Credentials) -> list[str]:
    """Human-readable credential report. Contains no secret material."""
    def _short(src: str) -> str:
        # A 90-character absolute path makes the status line unreadable, and the
        # directory is already known from the banner.
        return src if src in (SRC_SECRETS, SRC_ENV, SRC_MISSING) else Path(src).name

    lines = ["", "CREDENTIALS  (st.secrets -> environment -> .env)", "-" * 96]
    for label, value, source in (
        ("GROQ_API_KEY", creds.api_key, creds.key_source),
        ("GROQ_MODEL", creds.model, creds.model_source),
    ):
        if not value:
            lines.append(f"  [FAIL] {label:<13} not set            <- {_short(source)}")
            continue
        if label.endswith("KEY"):
            # Length and prefix check only. Never the value, never a substring.
            shape = "gsk_ prefix ok" if value.startswith("gsk_") else "WARNING: unexpected format"
            detail = f"{len(value)} chars, {shape}"
        else:
            detail = value
        lines.append(f"  [ OK ] {label:<13} {detail:<28} <- {_short(source)}")

    if creds.ok:
        lines.append("         Phase 3 does not call this model. It is validated here so a")
        lines.append("         broken credential surfaces at build time, not mid-answer in Phase 5.")
    else:
        lines.append("         Retrieval-only mode still works; generating an answer will not.")
    lines.append("         Key value deliberately not printed. Never log it.")
    lines.append("")
    return lines


# ── embedding ─────────────────────────────────────────────────────────────

def get_model():
    """Load the model once, with the sequence limit set explicitly.

    Resolved from the local cache via ``load_from_hub_cache`` rather than
    straight from the Hub: the weights are baked into the image at build time by
    ``scripts/prefetch_model.py``, and verified bit-identical to the shipped
    vectors.npy, so there is no reason for a cold start to depend on
    huggingface.co being reachable.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = load_from_hub_cache(
            lambda **kw: SentenceTransformer(EMBEDDING_MODEL, **kw)
        )
        _MODEL.max_seq_length = MAX_SEQ_LENGTH   # explicit, not inherited
        _MODEL.eval()                            # deterministic (NFR-06)
    return _MODEL


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found - run scripts/build_index.py --stages 1 2 first"
        )
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def assert_embeddable(texts: Sequence[str], ids: Sequence[str]) -> None:
    """Fail loud on anything that would be silently truncated (I11)."""
    over = [(i, count_wp(t)) for i, t in zip(ids, texts) if count_wp(t) > MAX_SEQ_LENGTH]
    if over:
        raise ValueError(
            f"{len(over)} chunk(s) exceed the {MAX_SEQ_LENGTH} word-piece limit and "
            f"would be silently truncated: {over[:5]}"
        )


def embed(texts: Sequence[str], ids: Sequence[str] | None = None,
          batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Embed to float32 ``[N, 384]``, L2-normalised.

    Embeds ``chunk["text"]`` (header + body), not the bare body: the context
    header is part of what makes the chunk findable by scheme name.
    """
    assert_embeddable(texts, ids if ids is not None else range(len(texts)))
    vectors = get_model().encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # coupled with hnsw:space=cosine — see I10
        show_progress_bar=False,
    )
    assert vectors.shape == (len(texts), VECTOR_DIM), \
        f"expected [{len(texts)},{VECTOR_DIM}], got {vectors.shape}"
    return vectors.astype(np.float32)


# ── metadata ──────────────────────────────────────────────────────────────

def chunk_metadata(c: dict) -> dict:
    """Flatten to Chroma-legal scalars. Chroma rejects nested objects, so lists
    and dicts are serialised rather than nested — and a citation consumer must
    not have to guess which fields survived the round trip."""
    return {
        "scheme_id": c["scheme_id"],
        "scheme_name": c["scheme_name"],
        "plan": c["plan"],
        "doc_type": c["doc_type"],
        "source_tier": c["source_tier"],
        "title": c["title"],
        "url": c["url"],
        "as_of_date": c["as_of_date"] or "",
        "section": c["section"],
        "heading_path_json": json.dumps(c["heading_path"]),
        "n_wordpieces": int(c["n_wordpieces"]),
        "is_table": bool(c["is_table"]),
        "has_overlap": bool(c["has_overlap"]),
        "contains_performance": bool(c["contains_performance"]),
        "ordinal": int(c["ordinal"]),
        "n_chars": int(c["n_chars"]),
        "content_hash": c["content_hash"],
        "chunk_params_json": json.dumps(c["chunk_params"]),
        "body": c["body"],          # header is prefixed in `document`; the
                                   # generator strips it, so keep the raw body
    }


# ── invariants I6–I11 ─────────────────────────────────────────────────────

def assert_invariants(ids: list[str], vectors: np.ndarray,
                      metadatas: list[dict], chunks: Sequence[dict]) -> None:
    """The silent-failure checks, run **before** the write.

    Each of these has a failure mode that still returns plausible answers.
    """
    # I6 — row alignment. The expensive one.
    if not (len(ids) == len(vectors) == len(metadatas) == len(chunks)):
        raise ValueError(
            f"I6 row-count mismatch: ids={len(ids)} vectors={len(vectors)} "
            f"metadatas={len(metadatas)} chunks={len(chunks)}"
        )

    # I7 — unique chunk_id, which is also the upsert key.
    if len(set(ids)) != len(ids):
        dupes = [i for i in set(ids) if ids.count(i) > 1][:5]
        raise ValueError(f"I7 duplicate chunk_id (upsert would silently collapse): {dupes}")

    # I8 — citation allow-list (R2). A citation outside it is a dead/false link.
    bad_url = sorted({m["url"] for m in metadatas if m["url"] not in ALLOWED_URLS})
    if bad_url:
        raise ValueError(f"I8 url not in the citation allow-list: {bad_url[:3]}")

    # I9 — scope (R8). An out-of-scope scheme in the index is answerable but
    # must never be, and its presence is invisible until someone asks about it.
    bad_scheme = sorted({m["scheme_id"] for m in metadatas
                         if m["scheme_id"] not in ALLOWED_SCHEME_IDS})
    if bad_scheme:
        raise ValueError(f"I9 scheme_id outside the 5-scheme allow-list: {bad_scheme}")

    # I10 — L2 normalisation, the other half of the cosine-index pair.
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        worst = float(np.abs(norms - 1.0).max())
        raise ValueError(
            f"I10 vectors are not L2-normalised (max deviation {worst:.2e}). With "
            f"hnsw:space=cosine this makes the confidence floor meaningless."
        )

    # I11 — no silent truncation.
    over = [i for i, c in zip(ids, chunks) if int(c["n_wordpieces"]) > MAX_SEQ_LENGTH]
    if over:
        raise ValueError(f"I11 {len(over)} chunk(s) over {MAX_SEQ_LENGTH} wp: {over[:5]}")

    # Citation scaffolding must survive the flatten.
    for m in metadatas:
        if not m["url"] or not m["as_of_date"]:
            raise ValueError(f"I8b citation field empty in metadata: {m['chunk_id'] if 'chunk_id' in m else m['scheme_id']}")


# ── the alignment check that matters ──────────────────────────────────────

def verify_alignment(collection, chunks: Sequence[dict], ids: Sequence[str],
                     sample: int = 12, tol: float = 2e-3) -> dict:
    """Re-derive vectors for a sample and compare against what Chroma stored.

    This is the only check that can catch a vectors/ids row offset, because it
    compares the **persisted** index against a **fresh embedding** — an
    independent derivation. Comparing the stored array to the in-memory array
    would pass even when both carry the same misalignment, which is the whole
    hazard.

    Chroma returns L2-normalised copies, so the comparison is exact to float
    tolerance rather than approximate.
    """
    probe_ids = list(ids[:sample]) + list(ids[-sample:])
    texts = {c["chunk_id"]: c["text"] for c in chunks}
    expected = embed([texts[i] for i in probe_ids], probe_ids)

    got = collection.get(ids=probe_ids, include=["embeddings"])
    stored = {i: np.asarray(v, dtype=np.float32)
              for i, v in zip(got["ids"], got["embeddings"])}

    missing = [i for i in probe_ids if i not in stored]
    if missing:
        raise ValueError(f"alignment probe: {len(missing)} id(s) absent from the "
                         f"collection: {missing[:5]}")

    deltas = []
    for row, cid in enumerate(probe_ids):
        d = float(np.abs(stored[cid] - expected[row]).max())
        deltas.append((d, cid))
    deltas.sort(reverse=True)

    worst, worst_id = deltas[0]
    if worst > tol:
        # Name the swap if that is what happened — it makes the bug legible.
        hint = ""
        for d, cid in deltas[:1]:
            row = probe_ids.index(cid)
            for other, cand in enumerate(probe_ids):
                if other != row and float(np.abs(stored[cid] - expected[other]).max()) < tol:
                    hint = f" — chunk {cid} holds the vector of {cand}"
                    break
        raise ValueError(
            f"alignment check FAILED: stored vector for {worst_id} differs from a "
            f"fresh embedding by {worst:.2e} (tol {tol:.2e}){hint}. "
            f"vectors and ids are misaligned; every citation would be attributed "
            f"to the wrong scheme."
        )
    return {"probed": len(probe_ids), "max_abs_delta": worst, "tol": tol,
            "worst_id": worst_id}


# ── build ─────────────────────────────────────────────────────────────────

def get_client(path: Path = CHROMA_PATH):
    import chromadb

    path.mkdir(parents=True, exist_ok=True)
    # No embedding_function passed: this pipeline embeds explicitly so the
    # vectors are normalised and asserted *before* they reach the store. Letting
    # Chroma embed internally would hand the model a second, unasserted code path.
    return chromadb.PersistentClient(path=str(path))


#: HNSW index parameters. **Baked into the persisted index** — changing them
#: after creation does not reindex, so `get_collection` refuses to reuse a
#: collection whose space disagrees rather than silently serving a differently-
#: shaped neighbourhood.
#:
#: Chroma >= 1.5 takes these through a typed `configuration` argument; the older
#: `metadata={"hnsw:space": ...}` form now raises
#: `InvalidArgumentError: Failed to parse hnsw parameters from segment metadata`.
#: `hnsw:M` is renamed `max_neighbors`.
HNSW_CONFIG = {
    "space": "cosine",        # coupled with L2-normalised vectors (I10)
    "ef_construction": 200,
    "max_neighbors": 16,
    "ef_search": 128,
}


def get_collection(client, name: str = COLLECTION_NAME, recreate: bool = False):
    """Create or open the collection, refusing a mismatched index.

    The space is the one parameter that cannot be changed in place. If a
    collection already exists at a different space, silently opening it would
    return scores that are not cosines, and the 0.62 confidence floor calibrated
    against cosine would be meaningless while every number still looked normal.
    """
    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    try:
        existing = client.get_collection(name)
    except Exception:
        existing = None

    if existing is not None:
        space = (existing.configuration or {}).get("hnsw", {}).get("space")
        if space != HNSW_CONFIG["space"]:
            raise ValueError(
                f"collection '{name}' exists with hnsw space {space!r}, but this "
                f"build requires {HNSW_CONFIG['space']!r}. The space is baked into "
                f"the persisted index and cannot be changed in place. Re-run with "
                f"--recreate."
            )
        return existing

    # No embedding_function: this pipeline embeds explicitly so vectors are
    # normalised and asserted *before* they reach the store. Letting Chroma embed
    # internally would open a second, unasserted code path to the vectors.
    return client.create_collection(
        name=name,
        configuration={"hnsw": dict(HNSW_CONFIG)},
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "max_seq_length": MAX_SEQ_LENGTH,
            "vector_dim": VECTOR_DIM,
            "index_space": HNSW_CONFIG["space"],
        },
    )


def save_npy(ids: list[str], vectors: np.ndarray, out_dir: Path | None = None) -> Path:
    """Persist vectors + row-aligned ids as the auditable artefact.

    `ids.json` is written from the *same list object* that ordered `vectors`, so
    the two files cannot drift apart, and both are written in one operation.
    """
    out_dir = out_dir or (ROOT / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "vectors.npy"
    ids_path = out_dir / "ids.json"
    np.save(vec_path, vectors)
    ids_path.write_text(json.dumps(ids, indent=1), encoding="utf-8")
    return vec_path


def build(recreate: bool = False, path: Path = CHROMA_PATH) -> dict:
    chunks = load_chunks()
    # One ordered pass. ids, texts and metadatas are appended together, so a
    # filter that drops a chunk cannot desynchronise the three.
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for c in chunks:
        ids.append(c["chunk_id"])
        texts.append(c["text"])
        metadatas.append(chunk_metadata(c))

    print(f"  loaded {len(chunks)} chunks from {CHUNKS_PATH.name}")
    vectors = embed(texts, ids)
    print(f"  embedded -> {vectors.shape} {vectors.dtype}, "
          f"norms in [{np.linalg.norm(vectors, axis=1).min():.6f}, "
          f"{np.linalg.norm(vectors, axis=1).max():.6f}]")

    assert_invariants(ids, vectors, metadatas, chunks)
    print("  I6-I11 all pass (row alignment, unique ids, url allow-list, "
          "scope, L2 norm, word-piece cap)")

    client = get_client(path)
    collection = get_collection(client, recreate=recreate)
    space = (collection.configuration or {}).get("hnsw", {}).get("space")
    print(f"  collection '{COLLECTION_NAME}' hnsw:space={space} "
          f"(max_neighbors={HNSW_CONFIG['max_neighbors']}, "
          f"ef_search={HNSW_CONFIG['ef_search']})")

    existing = collection.count()
    collection.upsert(
        ids=ids,
        embeddings=[v.tolist() for v in vectors],
        documents=texts,
        metadatas=metadatas,
    )
    final = collection.count()
    print(f"  upserted {len(ids)} -> collection '{COLLECTION_NAME}' "
          f"({existing} before, {final} after"
          + (", unchanged as expected" if existing == final else "") + ")")

    if final != len(ids):
        raise ValueError(
            f"collection holds {final} rows but chunks.jsonl has {len(ids)}. "
            f"A count above the input means stale rows from a previous build with "
            f"different parameters; re-run with --recreate."
        )

    # The check that earns its cost.
    align = verify_alignment(collection, chunks, ids)
    print(f"  alignment verified against fresh re-embedding: "
          f"{align['probed']} probes, max delta {align['max_abs_delta']:.2e} "
          f"(tol {align['tol']:.0e})")

    save_npy(ids, vectors)
    return {"chunks": len(ids), "collection": final, "alignment": align,
            "path": str(path)}


# ═══════════════════════════════════════════════════════════════════════════
# Cold-start index materialisation
# ═══════════════════════════════════════════════════════════════════════════

#: `auto` (default) rebuilds when the persisted index is absent, empty, or
#: incomplete. `never` restores the hard failure, which is what CI wants: a
#: silent rebuild turns a corrupt-index bug into a passing build, because the
#: rebuild succeeds from the same file the corruption would have to come from.
INDEX_POLICY_ENV = "HDFC_RAG_INDEX_POLICY"


def index_policy() -> str:
    return (os.environ.get(INDEX_POLICY_ENV) or "auto").strip().lower() or "auto"


def index_status(path: Path = CHROMA_PATH) -> dict:
    """Report whether the persisted index can serve queries. Never raises.

    Separate from :func:`ensure_index` so a caller can *ask* without triggering
    a 90-second model download as a side effect of a status check.

    "Never raises" includes not raising ``SystemExit``. ``load_chunks`` exits the
    process when the chunk file is missing, which is the right behaviour for a
    build script and the wrong behaviour here: a status probe that kills the
    interpreter takes the web app down with it and prints nothing on the way.
    """
    out: dict = {"path": str(path), "expected": None, "count": 0,
                 "present": path.exists(), "usable": False, "chunks": False,
                 "problem": ""}
    if not CHUNKS_PATH.exists():
        out["problem"] = f"{CHUNKS_PATH.name} is missing, so the corpus size is unknown"
        return out
    try:
        out["expected"] = len(load_chunks())
        out["chunks"] = True
    except SystemExit:                  # load_chunks' own guard, contained
        out["problem"] = f"{CHUNKS_PATH.name} could not be read"
        return out
    if not path.exists():
        out["problem"] = f"{path} does not exist"
        return out
    try:
        count = get_collection(get_client(path)).count()
    except Exception as exc:            # noqa: BLE001 - reported, not raised
        out["problem"] = f"cannot open collection: {exc}"
        return out
    out["count"] = count
    if count == 0:
        out["problem"] = "collection is empty"
    elif count != out["expected"]:
        # The dangerous state, and the reason the trigger is a count rather
        # than an existence check. `build()` is not atomic: a process killed
        # mid-upsert leaves a collection that exists, is non-empty, and is
        # short. Answering from it would be the §3.12 failure arriving through
        # a different door.
        out["problem"] = (f"collection holds {count} vectors, "
                          f"chunks.jsonl holds {out['expected']}")
    else:
        out["usable"] = True
    return out


def ensure_index(path: Path = CHROMA_PATH, *, verbose: bool = True,
                 check_alignment: bool = False) -> dict:
    """Return a usable collection, rebuilding it from ``chunks.jsonl`` if not.

    On a normal deploy this function does almost nothing: ``chroma_db/`` is
    committed, so :func:`index_status` finds a complete collection and the first
    branch returns without touching the embedding model at all. The rebuild
    below is the *fallback*, not the expected path, and it exists for three
    cases - a local checkout that never built the index, a stale index whose row
    count disagrees with ``chunks.jsonl``, and a committed folder that turns out
    not to load on an untested platform.

    That last case is why shipping the folder is safe rather than a gamble. If
    these bytes cannot be opened anywhere, the failure mode is the old slow
    boot, not a broken app.

    The rebuild needs no network and no secret, only the model weights, and it
    cannot help if ``chunks.jsonl`` is absent: there is no remote index to fail
    over to, because ``chroma_db/`` is derived from that file and nothing else.
    So the missing-chunk-file error below is raised *before* any of this, as the
    only message that can tell an operator what is actually wrong.

    Differences from :func:`build`, all deliberate:
      * no ``verify_alignment`` by default — it re-embeds the whole corpus a
        second time purely to cross-check Phase 3, which would double rebuild
        cost for a check that already ran. Pass ``check_alignment=True`` to get it.
      * ``save_npy`` is best-effort. On a read-only or ephemeral filesystem the
        sidecar audit artefacts are a convenience, and failing to write them
        must not stop the app from answering. Phase 3's ``build()`` still treats
        them as required, because there the filesystem is known-writable.
      * :func:`get_collection` still refuses a space mismatch, so a rebuild can
        never paper over an index built with the wrong metric.
    """
    # The corpus is checked first, and cheaply. It is the source of truth, so
    # there is nothing to decide about the index until we know it can be
    # rebuilt — and on a deploy missing the chunk file, this is the only
    # message that can tell the operator what is actually wrong.
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"cannot rebuild the index: {CHUNKS_PATH} is missing, and it is the "
            f"only source of truth for the corpus. {path} is not a cache of "
            f"anything remote — ship chunks.jsonl with the deploy."
        )

    status = index_status(path)
    if status["usable"]:
        return {"rebuilt": False, "reason": "index present and complete",
                "count": status["count"], "path": str(path)}

    policy = index_policy()
    if policy == "never":
        raise RuntimeError(
            f"index unusable ({status['problem']}) and {INDEX_POLICY_ENV}=never. "
            f"Build it first: python -m src.vector_store"
        )
    if policy not in ("auto", "force"):
        raise ValueError(
            f"{INDEX_POLICY_ENV}={policy!r} is not a policy. Use 'auto' "
            f"(default), 'never', or 'force'."
        )

    say = print if verbose else (lambda *a, **k: None)
    chunks = load_chunks()
    say(f"  index cold start: {status['problem']}")
    say(f"  rebuilding {len(chunks)} chunks from {CHUNKS_PATH.name} "
        f"(model {EMBEDDING_MODEL})")

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for c in chunks:                    # one ordered pass; see build()
        ids.append(c["chunk_id"])
        texts.append(c["text"])
        metadatas.append(chunk_metadata(c))

    vectors = embed(texts, ids)         # may raise if the weights cannot be fetched
    assert_invariants(ids, vectors, metadatas, chunks)

    client = get_client(path)
    # Two workers booting at once can both miss the collection and both try to
    # create it. The loser's create_collection raises; reopening wins the race
    # and the upsert is idempotent by id, so the end state converges either way.
    try:
        collection = get_collection(client)
    except Exception:                   # noqa: BLE001 - concurrent create
        collection = get_collection(client)

    collection.upsert(ids=ids, embeddings=[v.tolist() for v in vectors],
                      documents=texts, metadatas=metadatas)
    final = collection.count()
    if final != len(ids):
        raise RuntimeError(
            f"rebuild wrote {len(ids)} chunks but the collection holds {final}. "
            f"Delete {path} and retry; a short count here means the upsert was "
            f"interrupted and the index cannot be trusted."
        )
    say(f"  rebuilt -> {final} vectors, hnsw:space={HNSW_CONFIG['space']}")

    align = None
    if check_alignment:
        align = verify_alignment(collection, chunks, ids)
        say(f"  alignment verified: {align['probed']} probes, "
            f"max delta {align['max_abs_delta']:.2e}")

    try:
        save_npy(ids, vectors)
    except OSError as exc:
        say(f"  note: could not write vectors.npy/ids.json ({exc}). The app does "
            f"not read them at query time, so this is not fatal.")

    return {"rebuilt": True, "reason": status["problem"], "count": final,
            "path": str(path), "alignment": align}


def main() -> int:
    recreate = "--recreate" in sys.argv
    print("=" * 96)
    print("STAGE 3: EMBEDDING + LOCAL PERSISTENT CHROMADB")
    print("=" * 96)
    creds = load_credentials()
    for line in report_credentials(creds):
        print(line)

    print(f"  chunks      : {CHUNKS_PATH}")
    print(f"  model       : {EMBEDDING_MODEL}  (max_seq_length={MAX_SEQ_LENGTH}, "
          f"dim={VECTOR_DIM}, mean-pooled, L2-normalised)")
    print(f"  chroma path : {CHROMA_PATH}")
    print(f"  collection  : {COLLECTION_NAME}  (hnsw:space=cosine)")
    print()

    result = build(recreate=recreate)

    # Persistence proof: a brand-new client against the same path must see the
    # data. An in-memory client would pass every check above and still lose it.
    client = get_client()
    reopened = client.get_collection(COLLECTION_NAME)
    n = reopened.count()
    print(f"\n  persistence: fresh client re-opened '{COLLECTION_NAME}' "
          f"and sees {n} rows")
    if n != result["collection"]:
        raise ValueError("index did not survive a client re-open")
    print(f"  chroma_db/  : {(CHROMA_PATH).resolve()}")

    print()
    print("=" * 96)
    print(f"STAGE 3 OK — {result['collection']} chunks, {VECTOR_DIM}-d, "
          f"cosine, persistent at ./chroma_db")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
