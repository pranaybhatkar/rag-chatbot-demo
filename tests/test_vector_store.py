"""tests/test_vector_store.py — Stage 3 invariants, and the guards themselves.

The tests that matter most here are the ones that deliberately *break* an
invariant and assert the guard fires. A guard that has never been observed to
catch anything is indistinguishable from a guard that does not work.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"

from src import vector_store as vs
from src.config import ALLOWED_SCHEME_IDS, ALLOWED_URLS

pytestmark = pytest.mark.skipif(
    not CHUNKS.exists(), reason="run scripts/build_index.py --stages 1 2 first"
)


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]


@pytest.fixture(scope="module")
def embedded(chunks):
    """Embed a small slice once — the model load dominates the runtime."""
    sample = chunks[:24]
    ids = [c["chunk_id"] for c in sample]
    return sample, ids, vs.embed([c["text"] for c in sample], ids)


@pytest.fixture
def scratch_client(tmp_path):
    """A throwaway Chroma client that releases its SQLite handle.

    Chroma's Rust bindings keep `chroma.sqlite3` open until `close()` is called.
    Without that, pytest's `tmp_path` cleanup fails on Windows with
    `PermissionError: [WinError 32] the process cannot access the file because it
    is being used by another process` — a test-harness artefact that has nothing
    to do with the code under test, but one that would mask real failures.
    """
    client = vs.get_client(tmp_path / "chroma_scratch")
    try:
        yield client
    finally:
        client.close()


# ── credentials ───────────────────────────────────────────────────────────

def test_dotenv_file_exists_and_is_ignored_by_git():
    assert (ROOT / ".env").exists(), ".env is required by Phase 3's spec"
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignore, ".env holds a live key and must be gitignored"


def test_dotenv_reads_both_required_vars():
    creds = vs.load_credentials()
    assert creds.api_key, "GROQ_API_KEY was not read from .env"
    assert creds.model, "GROQ_MODEL was not read from .env"
    assert creds.ok


#: Models this key used to be able to reach and no longer can. Measured, not
#: guessed: a request to each returned HTTP 404 ``model_not_found`` while the key
#: itself was valid (other models on the same key answered 200). Recorded here
#: because the failure is otherwise invisible — the pipeline degrades to the
#: extractive path and still produces correct-looking answers, so a dead model
#: reads as "the LLM isn't being used" rather than "this name is retired".
KNOWN_DEAD_MODELS = {"llama-3.1-8b-instant", "llama3-8b-8192", "gemma2-9b-it"}


def test_model_is_not_a_known_retired_name():
    """Replaces a test that asserted one hardcoded model string.

    The previous version asserted ``== "llama-3.1-8b-instant"``, which is why it
    went red when that model was retired: pinning a vendor's model catalogue in
    a test guarantees the test fails the first time the vendor rotates. What
    actually needs protecting is that ``.env`` names a model the key can reach,
    and that ``.env`` and ``.env.example`` do not disagree.
    """
    model = vs.load_credentials().model
    assert model, "GROQ_MODEL is empty"
    assert model not in KNOWN_DEAD_MODELS, (
        f"GROQ_MODEL={model!r} is retired; the key returns 404 model_not_found "
        f"for it. Pick a live one and update .env and .env.example together."
    )


def test_env_and_example_agree_on_the_model():
    """.env and .env.example drifting is the bug that made the change above
    necessary: the model moved in one file and the deploy template kept the old
    name, so a fresh deploy silently ran a different generator than the one the
    tests were tuned against."""
    import re

    def read(path):
        text = (ROOT / path).read_text(encoding="utf-8")
        m = re.search(r"^\s*GROQ_MODEL\s*=\s*(.+?)\s*$", text, re.M)
        assert m, f"no GROQ_MODEL in {path}"
        return m.group(1).strip().strip('"').strip("'")

    assert read(".env") == read(".env.example"), (
        "GROQ_MODEL differs between .env and .env.example; a deploy built from "
        "the template would run a different generator than the tests"
    )


@pytest.mark.skipif(
    os.environ.get("HDFC_RAG_LIVE_MODEL_CHECK") != "1",
    reason="set HDFC_RAG_LIVE_MODEL_CHECK=1 to call the Groq models endpoint",
)
def test_configured_model_is_reachable():
    """The live check behind the two offline tests. Opt-in because it needs the
    network and a funded key."""
    import requests

    creds = vs.load_credentials()
    r = requests.get("https://api.groq.com/openai/v1/models",
                     headers={"Authorization": f"Bearer {creds.api_key}"},
                     timeout=30)
    assert r.status_code == 200, f"key rejected: HTTP {r.status_code}"
    live = {m["id"] for m in r.json().get("data", [])}
    assert creds.model in live, (
        f"GROQ_MODEL={creds.model!r} is not in the key's catalogue "
        f"({len(live)} models available)"
    )


def test_credential_report_never_prints_the_key():
    """A verification routine that echoes the secret it verifies is how keys
    end up in build logs."""
    creds = vs.load_credentials()
    report = "\n".join(vs.report_credentials(creds))
    assert creds.api_key not in report
    # Nor any prefix long enough to be useful to an attacker.
    assert creds.api_key[:12] not in report
    assert "gsk_" in report, "report should still show a format sanity-check"


def test_example_env_contains_no_real_secret():
    """The committed template must be safe to push."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    real = vs.load_credentials().api_key
    assert real not in text
    assert "REPLACE_WITH_YOUR_GROQ_KEY" in text


def test_missing_credentials_are_reported_not_raised(monkeypatch):
    """A missing key must degrade to retrieval-only mode, not crash the build."""
    monkeypatch.setattr(vs.os, "getenv", lambda *a, **k: "")
    creds = vs.load_credentials()
    assert not creds.ok
    report = "\n".join(vs.report_credentials(creds))
    assert "not set" in report
    assert "Retrieval-only mode still works" in report
    # Nothing secret to leak on this path, and no stray value printed either.
    assert "FAIL" in report


# ── embedding ─────────────────────────────────────────────────────────────

def test_model_sequence_limit_is_explicit(chunks):
    model = vs.get_model()
    assert model.max_seq_length == 256


def test_embed_shape_dtype_and_norm(embedded):
    _, _, vectors = embedded
    assert vectors.shape == (24, vs.VECTOR_DIM)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embedding_is_deterministic(chunks):
    """NFR-06: the same chunk must always yield the same vector, or a reindex
    silently rewrites the index and no comparison is meaningful."""
    ids = [c["chunk_id"] for c in chunks[:4]]
    a = vs.embed([c["text"] for c in chunks[:4]], ids)
    b = vs.embed([c["text"] for c in chunks[:4]], ids)
    assert np.array_equal(a, b)


def test_embed_rejects_an_over_long_chunk():
    from src.chunker import count_wp
    long_text = "The expense ratio of the scheme is 1.03% per annum. " * 60
    assert count_wp(long_text) > 256
    with pytest.raises(ValueError, match="silently truncated"):
        vs.assert_embeddable([long_text], ["too_long"])


def test_embed_accepts_the_real_corpus(chunks):
    vs.assert_embeddable([c["text"] for c in chunks], [c["chunk_id"] for c in chunks])


# ── metadata flattening ───────────────────────────────────────────────────

def test_metadata_is_scalars_only(chunks):
    """Chroma rejects nested objects; a silently dropped field is a silently
    missing citation."""
    meta = vs.chunk_metadata(chunks[0])
    for key, value in meta.items():
        assert not isinstance(value, (list, dict, tuple, set)), f"{key} is nested"
        assert isinstance(value, (str, int, float, bool)), f"{key} has odd type"
    assert json.loads(meta["heading_path_json"])
    assert json.loads(meta["chunk_params_json"])


def test_metadata_keeps_the_citation_fields(chunks):
    meta = vs.chunk_metadata(chunks[0])
    for field in ("url", "as_of_date", "scheme_id", "section", "content_hash", "body"):
        assert meta[field] != "", field


# ── invariants: prove each guard actually fires ───────────────────────────

def _args(chunks):
    ids = [c["chunk_id"] for c in chunks[:3]]
    vectors = vs.embed([c["text"] for c in chunks[:3]], ids)
    metadatas = [vs.chunk_metadata(c) for c in chunks[:3]]
    return ids, vectors, metadatas, chunks[:3]


def test_invariants_pass_on_the_real_corpus(chunks, embedded):
    _, ids, vectors = embedded
    sample = chunks[:len(ids)]
    metadatas = [vs.chunk_metadata(c) for c in sample]
    vs.assert_invariants(ids, vectors, metadatas, sample)   # must not raise


def test_i6_catches_a_row_count_mismatch(chunks):
    ids, vectors, metadatas, sample = _args(chunks)
    with pytest.raises(ValueError, match="I6"):
        vs.assert_invariants(ids, vectors, metadatas[:-1], sample)


def test_i7_catches_duplicate_chunk_ids(chunks):
    ids, vectors, metadatas, sample = _args(chunks)
    dup = [ids[0], ids[0], ids[2]]
    with pytest.raises(ValueError, match="I7"):
        vs.assert_invariants(dup, vectors, metadatas, sample)


def test_i8_catches_a_url_outside_the_allow_list(chunks):
    ids, vectors, metadatas, sample = _args(chunks)
    metadatas[0]["url"] = "https://example.com/not-a-source"
    with pytest.raises(ValueError, match="I8"):
        vs.assert_invariants(ids, vectors, metadatas, sample)


def test_i9_catches_an_out_of_scope_scheme(chunks):
    ids, vectors, metadatas, sample = _args(chunks)
    metadatas[0]["scheme_id"] = "HDFC_MID_CAP"
    with pytest.raises(ValueError, match="I9"):
        vs.assert_invariants(ids, vectors, metadatas, sample)


def test_i10_catches_unnormalised_vectors(chunks):
    """The silent one. Un-normalised vectors plus a cosine index do not crash;
    they make the 0.62 confidence floor mean nothing."""
    ids, vectors, metadatas, sample = _args(chunks)
    unnormalised = (vectors * 7.0).astype(np.float32)
    with pytest.raises(ValueError, match="I10"):
        vs.assert_invariants(ids, unnormalised, metadatas, sample)


def test_i11_catches_a_chunk_over_the_wordpiece_cap(chunks):
    ids, vectors, metadatas, sample = _args(chunks)
    sample[0] = dict(sample[0], n_wordpieces=999)
    with pytest.raises(ValueError, match="I11"):
        vs.assert_invariants(ids, vectors, metadatas, sample)


# ── alignment: the check that must earn its cost ──────────────────────────

def test_alignment_check_detects_a_swapped_row(embedded, scratch_client):
    """A one-row offset attributes every citation to the wrong scheme, and every
    answer still looks fine because the text is plausible and the URL resolves.

    The guard compares the *persisted* index against a *fresh re-embedding*. This
    test proves that comparison can actually fail: it deliberately swaps two
    rows and asserts the guard says so.
    """
    sample, ids, vectors = embedded
    coll = scratch_client.create_collection(
        name=vs.COLLECTION_NAME,
        configuration={"hnsw": dict(vs.HNSW_CONFIG)},
    )
    coll.upsert(
        ids=ids,
        embeddings=[v.tolist() for v in vectors],
        documents=[c["text"] for c in sample],
        metadatas=[vs.chunk_metadata(c) for c in sample],
    )
    # The guard passes on an honest write.
    vs.verify_alignment(coll, sample, ids, sample=4)

    # Now corrupt exactly one row the way a desynchronised loop would.
    swapped = vectors.copy()
    swapped[3] = vectors[4]
    coll.update(ids=[ids[3]], embeddings=[swapped[3].tolist()])

    with pytest.raises(ValueError, match="alignment check FAILED"):
        vs.verify_alignment(coll, sample, ids, sample=8)


def test_alignment_check_reports_an_absent_id(embedded, scratch_client):
    sample, ids, _ = embedded
    coll = scratch_client.create_collection(
        name=vs.COLLECTION_NAME,
        configuration={"hnsw": dict(vs.HNSW_CONFIG)},
    )
    coll.add(ids=[ids[0]], embeddings=[[0.0] * vs.VECTOR_DIM],
             documents=["x"], metadatas=[vs.chunk_metadata(sample[0])])
    with pytest.raises(ValueError, match="absent from the collection"):
        vs.verify_alignment(coll, sample, ids, sample=8)


# ── index space: the coupled invariant, pinned ────────────────────────────

def test_collection_uses_cosine_space():
    """cosine index + L2-normalised vectors is a coupled pair.

    Getting it wrong does not crash. It makes the 0.62 confidence floor
    meaningless while every score still looks like a similarity, so the setting
    is asserted rather than assumed.
    """
    coll = vs.get_client().get_collection(vs.COLLECTION_NAME)
    space = (coll.configuration or {}).get("hnsw", {}).get("space")
    assert space == "cosine"
    assert vs.HNSW_CONFIG["space"] == "cosine"


def test_mismatched_space_is_refused_not_silently_opened(scratch_client):
    """A collection built at a different space must not be reused.

    The space is baked into the persisted index; silently opening a collection
    whose space disagrees returns scores that are not cosines, and the floor
    calibrated against cosine becomes meaningless with no visible symptom.
    """
    scratch_client.create_collection(
        name=vs.COLLECTION_NAME,
        configuration={"hnsw": {"space": "l2"}},
    )
    with pytest.raises(ValueError, match="baked into the persisted index"):
        vs.get_collection(scratch_client, recreate=False)


def test_cosine_distance_matches_one_minus_similarity(scratch_client):
    """Confirms the index is really doing cosine, not a metric that looks like it."""
    coll = scratch_client.create_collection(
        name="metric_probe", configuration={"hnsw": dict(vs.HNSW_CONFIG)})
    a = np.zeros(vs.VECTOR_DIM, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(vs.VECTOR_DIM, dtype=np.float32)
    b[0], b[1] = 0.9, 0.1
    coll.add(ids=["a", "b"], embeddings=[a.tolist(), b.tolist()],
             documents=["x", "y"], metadatas=[{"v": "a"}, {"v": "b"}])
    res = coll.query(query_embeddings=[a.tolist()], n_results=2)
    expected = 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert abs(res["distances"][0][0] - 0.0) < 1e-5          # self-match
    assert abs(res["distances"][0][1] - expected) < 1e-4     # 1 - cosine


# ── the persisted array ───────────────────────────────────────────────────

def test_vectors_npy_is_row_aligned_with_ids_json(chunks):
    vec_path = ROOT / "data" / "processed" / "vectors.npy"
    ids_path = ROOT / "data" / "processed" / "ids.json"
    if not vec_path.exists():
        pytest.skip("run python -m src.vector_store first")
    vectors = np.load(vec_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    assert vectors.shape == (len(ids), vs.VECTOR_DIM)
    assert len(set(ids)) == len(ids)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    # Row i of the array must be the vector for ids[i] — verified by content,
    # not by trusting the write order.
    by_id = {c["chunk_id"]: c for c in chunks}
    for row in (0, len(ids) // 2, len(ids) - 1):
        fresh = vs.embed([by_id[ids[row]]["text"]], [ids[row]])[0]
        assert np.abs(vectors[row] - fresh).max() < 2e-3, ids[row]


# ── persistence and idempotency ───────────────────────────────────────────

def test_index_survives_a_fresh_client(chunks):
    """An in-memory Chroma client passes every other check here and still loses
    everything on exit. Only a re-open proves persistence."""
    if not vs.CHROMA_PATH.exists():
        pytest.skip("run python -m src.vector_store first")
    expected = json.loads(
        (ROOT / "data" / "processed" / "ids.json").read_text(encoding="utf-8")
    )
    client = vs.get_client()
    coll = client.get_collection(vs.COLLECTION_NAME)
    assert coll.count() == len(expected)


def test_collection_holds_exactly_the_corpus(chunks):
    if not vs.CHROMA_PATH.exists():
        pytest.skip("run python -m src.vector_store first")
    coll = vs.get_client().get_collection(vs.COLLECTION_NAME)
    got = coll.get(include=["metadatas"])
    assert coll.count() == len(chunks)
    assert {m["scheme_id"] for m in got["metadatas"]} == set(ALLOWED_SCHEME_IDS)
    assert all(m["url"] in ALLOWED_URLS for m in got["metadatas"])
    assert all(m["as_of_date"] for m in got["metadatas"])


def test_upsert_is_idempotent(chunks):
    """Re-running must update in place, not append: a doubled count means stale
    rows from an earlier build are still live."""
    if not vs.CHROMA_PATH.exists():
        pytest.skip("run python -m src.vector_store first")
    coll = vs.get_client().get_collection(vs.COLLECTION_NAME)
    before = coll.count()
    result = vs.build()
    assert result["collection"] == before == len(chunks)


def test_query_returns_a_citation_for_a_brief_question():
    """Smoke test that the index is actually searchable and citable."""
    if not vs.CHROMA_PATH.exists():
        pytest.skip("run python -m src.vector_store first")
    coll = vs.get_client().get_collection(vs.COLLECTION_NAME)
    q = vs.embed(["What is the expense ratio of HDFC Large Cap Fund?"])[0]
    res = coll.query(query_embeddings=[q.tolist()], n_results=3)
    assert res["ids"][0], "no result for the brief's headline question"
    meta = res["metadatas"][0][0]
    assert meta["url"] in ALLOWED_URLS
    assert meta["as_of_date"]
