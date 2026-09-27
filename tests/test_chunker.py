"""tests/test_chunker.py — Stage 2 invariants.

Most of these guard against *silent* corruption. A chunk that overruns 256
word-pieces, a table torn mid-row, or a filter that deletes a brief topic all
produce a system that still runs, still answers, and still cites a working link
— while being wrong. So the negative tests are the point.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import (
    HARD_CAP_WP,
    MIN_BODY_CHARS,
    MIN_STANDALONE_WORDS,
    OVERLAP_WP,
    SEPARATORS,
    TARGET_WP,
    _is_fragment,
    chunk_document,
    count_wp,
    is_chrome,
    is_table,
    make_header,
    recursive_split,
    run,
    split_facts_block,
    split_table,
    tail_overlap,
)
from src.config import (
    ALLOWED_SCHEME_IDS,
    BRIEF_TOPIC_PROBES,
    EMBEDDING_MODEL,
    KNOWN_TOPIC_GAPS,
    WORDPIECE_HARD_CAP,
)

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DOCS = ROOT / "data" / "processed" / "documents.jsonl"

pytestmark = pytest.mark.skipif(
    not CHUNKS.exists(), reason="run scripts/build_index.py --stages 1 2 first"
)


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]


@pytest.fixture(scope="module")
def docs() -> list[dict]:
    return [json.loads(l) for l in DOCS.open(encoding="utf-8")]


# ── the hard cap ──────────────────────────────────────────────────────────

def test_tokenizer_reports_true_length_not_baked_in_truncation():
    """The shipped tokenizer.json truncates at 128 and pads to 128.

    Counting `len(encode(text).ids)` therefore returns exactly 128 for any
    over-long text, and a naive `<= 256` assertion passes on a chunk that loses
    its tail. load_tokenizer must disable both.
    """
    long = "The expense ratio of the scheme is 1.03% per annum. " * 40
    assert len(long) > 1000
    n = count_wp(long)
    assert n > 400, f"truncation is still active: counted only {n} word-pieces"
    assert n != 128


def test_short_string_is_not_padded_to_a_fixed_length():
    """Padding to 128 would inflate a 12-word-piece fact to 128."""
    assert count_wp("Expense ratio (TER): 1.03%") < 32


def test_no_chunk_exceeds_the_hard_cap(chunks):
    over = [(c["chunk_id"], c["n_wordpieces"]) for c in chunks
            if c["n_wordpieces"] > HARD_CAP_WP]
    assert not over, f"chunks would be silently truncated: {over[:5]}"


def test_recorded_wordpiece_count_matches_a_fresh_measurement(chunks):
    """The stored n_wordpieces must equal a fresh count, not a planning estimate."""
    for c in chunks:
        assert count_wp(c["text"]) == c["n_wordpieces"], c["chunk_id"]


def test_hard_cap_matches_the_configured_model_limit():
    assert HARD_CAP_WP == WORDPIECE_HARD_CAP == 256
    assert TARGET_WP < HARD_CAP_WP, "no room for the header or overlap"


# ── the recursive splitter ────────────────────────────────────────────────

def test_recursive_split_respects_the_budget_on_unlucky_text():
    text = " ".join(f"word{i}" for i in range(400))
    for piece in recursive_split(text, 60):
        assert count_wp(piece) <= 60


def test_recursive_split_keeps_text_and_never_empties_it():
    text = " ".join(f"word{i}" for i in range(400))
    rebuilt = " ".join(recursive_split(text, 60))
    assert rebuilt.replace(" ", "") == text.replace(" ", "")


def test_recursive_split_engages_the_char_fallback_on_unseparated_text():
    """Text with no whitespace can only be split by character.

    Note the input: BERT maps a *word longer than 100 characters* to a single
    ``[UNK]``, so 'x' * 3000 is 3 word-pieces and genuinely fits. Testing the
    fallback with it would pass for the wrong reason. A run of digits stays under
    the 100-character limit and is split one digit per word-piece, so it is the
    input that actually exercises the binary search.
    """
    digits = "9" * 99
    assert count_wp(digits) > 60
    pieces = recursive_split(digits, 60)
    assert len(pieces) > 1
    assert all(count_wp(p) <= 60 for p in pieces)


def test_recursive_split_does_not_crash_on_an_empty_separator_level():
    """Regression: SEPARATORS ended with '', which text.split('') rejects.

    The depth guard already routes to the character fallback, so the entry was
    both redundant and fatal. Every section in the real corpus has whitespace and
    never reached it.
    """
    assert "" not in SEPARATORS
    for separator in SEPARATORS:
        recursive_split(f"alpha{separator}beta", 200)


def test_a_word_longer_than_100_chars_is_counted_as_unknown_not_split():
    """Guard the assumption the test above rests on."""
    assert count_wp("x" * 3000) < 60, "BERT [UNK] behaviour changed; revisit the test"


def test_recursive_split_never_starts_a_piece_mid_word():
    text = " ".join(f"alpha{i} beta{i} gamma{i}" for i in range(120))
    for piece in recursive_split(text, 50):
        assert re.match(r"^\S+", piece)
        assert not re.match(r"^\d+[a-z]", piece), f"mid-word cut: {piece[:30]!r}"


def test_split_returns_nothing_for_blank_input():
    assert recursive_split("   \n\n  ", 60) == []


# ── tables ────────────────────────────────────────────────────────────────

def test_is_table_detects_pipe_grids():
    assert is_table("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |")
    assert not is_table("Expense ratio: 1.03%\nExit load: nil")


def test_a_table_that_fits_travels_whole():
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert split_table(table, 500) == [table]


def test_split_table_never_cuts_a_row_in_half():
    rows = "\n".join(f"| Row{i} | Value{i} | {i} |" for i in range(60))
    for piece in split_table(rows, 80):
        for line in piece.splitlines():
            if line.strip().startswith("|"):
                assert line.strip().endswith("|"), f"torn row: {line!r}"


def test_split_table_repeats_the_header_on_every_piece():
    rows = "\n".join(f"| Row{i} | Value{i} | Detail{i} |" for i in range(40))
    pieces = split_table(rows, 80)
    assert len(pieces) > 1
    first = rows.splitlines()[0]
    for piece in pieces:
        assert piece.splitlines()[0] == first


# ── the facts block: one fact per chunk ───────────────────────────────────

def test_facts_block_yields_one_fact_per_unit():
    text = ("Expense ratio (TER): 1.03%\n"
            "Exit load: Exit load of 1% if redeemed within 1 year\n"
            "Minimum SIP investment: INR 100")
    facts = split_facts_block(text)
    assert len(facts) == 3
    assert facts[0].startswith("Expense ratio")


def test_facts_block_keeps_a_wrapped_value_with_its_label():
    """A value continued on the next line must not be orphaned.

    This is the shape that produced 'Stamp duty: 0.005% (from July 1st,' and
    '2020)' in two different chunks.
    """
    facts = split_facts_block("Stamp duty: 0.005% (from July 1st,\n2020)")
    assert len(facts) == 1
    assert "2020)" in facts[0]


def test_high_value_facts_are_isolated(chunks):
    """The brief's headline question must not compete with four other figures."""
    for label in ("Expense ratio (TER)", "Riskometer level", "Benchmark:"):
        hits = [c for c in chunks if label in c["body"]]
        assert hits, f"{label} has no chunk of its own"
        for c in hits:
            others = sum(1 for other in ("Expense ratio (TER)", "Riskometer level",
                                         "Benchmark:", "Exit load:")
                         if other in c["body"])
            assert others <= 1, f"{label} shares a chunk with {others} other facts"


# ── the junk filter that ate a brief topic ────────────────────────────────

def test_a_terse_fact_is_not_mistaken_for_a_fragment():
    """Regression: MIN_BODY_CHARS=24 deleted 'ELSS • 3Y Lock-in'.

    That string is the corpus's only statement of the ELSS lock-in period, which
    is one of the brief's three example questions. The filter could not tell a
    terse fact from a broken fragment, so it removed the fact.
    """
    assert not _is_fragment("ELSS • 3Y Lock-in")
    assert not _is_fragment("Category: ELSS")


def test_but_real_fragments_still_are():
    for junk in ("5", "AA", "DM", "AB", "2020)", "18"):
        assert _is_fragment(junk), f"{junk!r} should be treated as a fragment"


def test_fragment_filter_needs_both_tests():
    """Short-but-claim-shaped survives; long-but-thin does not."""
    assert not _is_fragment("ELSS • 3Y Lock-in")   # 15 chars, 3 words
    assert _is_fragment("mm")                      # long enough, 1 word


def test_the_elss_lockin_statement_survives_chunking(chunks):
    hits = [c for c in chunks
            if re.search(r"lock[-\s]?in|lock\s+period", c["text"], re.I)]
    assert hits, "the ELSS lock-in statement was deleted by a filter"
    assert any(c["scheme_id"] == "HDFC_ELSS" for c in hits)


def test_no_chunk_body_is_a_bare_fragment(chunks):
    """Every body must be a claim. Facts-block lines are exempt from word count
    because 'ISIN: INF179K01YV8' is two words and a complete fact."""
    bad = [c["chunk_id"] for c in chunks
           if len(c["body"].strip()) < MIN_BODY_CHARS]
    assert not bad, f"fragment bodies embedded: {bad[:5]}"


# ── scope containment ─────────────────────────────────────────────────────

def test_is_chrome_flags_another_funds_full_scheme_name():
    assert is_chrome("HDFC NIFTY100 Low Volatility 30 Index Fund Direct Growth",
                     "HDFC Balanced Advantage Fund", "")
    assert is_chrome("HDFC Nifty500 Multicap 50:25:25 Index Fund Direct Growth",
                     "HDFC Balanced Advantage Fund", "")


def test_is_chrome_does_not_flag_the_documents_own_fund():
    """Regression risk: the published heading adds 'Direct Plan Growth' to the
    stored scheme name, so an equality test would delete the ELSS lock-in."""
    assert not is_chrome("HDFC ELSS Tax Saver Fund Direct Plan Growth",
                         "HDFC ELSS Tax Saver Fund", "")
    assert not is_chrome("HDFC Large Cap Fund Direct Growth",
                         "HDFC Large Cap Fund", "")


def test_no_out_of_scope_fund_is_indexed(chunks):
    other = re.compile(r"HDFC (?:NIFTY|Nifty|Mid Cap|Liquid|Debt|Banking|"
                       r"Dynamic|Value|Index|Momentum|Multicap|Corporate Bond)")
    hits = [c["chunk_id"] for c in chunks if other.search(c["text"])]
    assert not hits, f"out-of-scope funds in the index: {hits[:5]}"


def test_broker_platform_promo_is_not_indexed(chunks):
    promo = re.compile(r"trade in f&o|invest in (?:stocks|etfs|ipos)|"
                       r"watchlists?|real-time p&l", re.I)
    hits = [c["chunk_id"] for c in chunks if promo.search(c["text"])]
    assert not hits, f"broker marketing indexed: {hits[:5]}"


def test_only_the_five_in_scope_schemes_are_present(chunks):
    assert {c["scheme_id"] for c in chunks} == set(ALLOWED_SCHEME_IDS)


# ── citation scaffolding ──────────────────────────────────────────────────

def test_every_chunk_carries_the_citation_fields(chunks):
    for c in chunks:
        assert c["url"], c["chunk_id"]
        assert c["as_of_date"], c["chunk_id"]
        # C-1: Groww is Tier-1 and carries the literal value "broker";
        # HDFC/SEBI/AMFI are Tier-2 and carry "official".
        assert c["source_tier"] in {"broker", "official"}, c["chunk_id"]


def test_every_chunk_starts_with_its_own_context_header(chunks):
    for c in chunks:
        assert c["text"].startswith(c["header"]), c["chunk_id"]
        assert c["scheme_name"] in c["header"]


def test_header_is_bounded_so_it_cannot_eat_the_budget():
    header = make_header("HDFC Balanced Advantage Fund", "Direct Growth",
                         "scheme_page", "2026-09-25", "x" * 500)
    assert len(header) < 200
    assert header.endswith("...")


def test_chunks_never_span_a_section_boundary(chunks):
    """Each chunk names exactly the one section it came from."""
    for c in chunks:
        assert c["section"] and c["section"] in c["heading_path"], c["chunk_id"]


def test_chunk_ids_are_unique_and_ordered(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    for scheme in ALLOWED_SCHEME_IDS:
        ords = [c["ordinal"] for c in chunks if c["scheme_id"] == scheme]
        assert ords == sorted(ords)


def test_content_hash_matches_the_text(chunks):
    import hashlib
    for c in chunks:
        assert hashlib.sha256(c["text"].encode("utf-8")).hexdigest() \
            == c["content_hash"], c["chunk_id"]


# ── overlap ───────────────────────────────────────────────────────────────

def test_overlap_is_bounded_and_drawn_from_the_previous_chunk():
    prev = " ".join(f"alpha{i} beta{i}" for i in range(80))
    tail = tail_overlap(prev, OVERLAP_WP)
    assert tail
    assert count_wp(tail) <= OVERLAP_WP
    assert prev.endswith(tail)


def test_overlap_of_nothing_is_nothing():
    assert tail_overlap("", OVERLAP_WP) == ""
    assert tail_overlap("some text", 0) == ""


def test_overlap_never_pushes_a_chunk_over_the_cap(chunks):
    with_overlap = [c for c in chunks if c["has_overlap"]]
    assert with_overlap, "no chunk carries overlap; the feature is inert"
    for c in with_overlap:
        assert c["n_wordpieces"] <= HARD_CAP_WP, c["chunk_id"]


# ── probes must not false-pass ────────────────────────────────────────────

def test_the_lockin_probe_does_not_match_a_sip_table():
    """Regression: the probe was `lock[-\s]?in|3 years|three years`.

    It passed on the SIP calculator's '3 years' row in all five documents while
    the corpus stated no lock-in anywhere. A loose alternative buys a false pass,
    which is worse than a red check because it hides the gap.
    """
    pattern = BRIEF_TOPIC_PROBES["lock_in"]
    sip_row = "| 3 years | ₹1,80,000 | ₹1,87,909 | +4.39 % |"
    assert not re.search(pattern, sip_row, re.I)
    assert re.search(pattern, "ELSS • 3Y Lock-in", re.I)
    assert re.search(pattern, "3 year lock-in period", re.I)


def test_the_known_lockin_gap_is_declared_not_hidden():
    """The gap is real: Groww carries the lock-in only as a nav-bar fragment and
    never states the section 80C basis. It is recorded, not papered over."""
    assert "lock_in" in KNOWN_TOPIC_GAPS
    assert "80C" in KNOWN_TOPIC_GAPS["lock_in"]


def test_every_failing_probe_is_a_declared_gap(chunks):
    """A red probe that nobody has explained is a defect, not a finding."""
    corpus = "\n".join(c["text"] for c in chunks)
    for topic, pattern in BRIEF_TOPIC_PROBES.items():
        if not re.search(pattern, corpus, re.I):
            assert topic in KNOWN_TOPIC_GAPS, (
                f"topic {topic!r} is absent and is not a declared gap"
            )


# ── performance flag ──────────────────────────────────────────────────────

def test_return_figures_are_flagged_for_the_phase_5_lint(chunks):
    flagged = [c for c in chunks if c["contains_performance"]]
    assert flagged, "no chunk flagged; the returns tables were missed"
    assert all(re.search(r"%", c["body"]) for c in flagged)


def test_plain_facts_are_not_false_flagged(chunks):
    for label in ("Expense ratio (TER)", "Riskometer level"):
        for c in chunks:
            if c["body"].startswith(label):
                assert not c["contains_performance"], c["chunk_id"]


# ── determinism ───────────────────────────────────────────────────────────

def test_rechunking_is_byte_identical(docs):
    """Stage 3 upserts by chunk_id, so a drifting id or hash corrupts the index
    on every re-run without ever raising."""
    a = [chunk_document(d) for d in docs]
    b = [chunk_document(d) for d in docs]
    assert [c.to_json() for group in a for c in group] == \
           [c.to_json() for group in b for c in group]


def test_the_committed_artifact_matches_a_fresh_build(chunks, docs):
    fresh = [c.to_json() for d in docs for c in chunk_document(d)]
    assert [json.dumps(c, ensure_ascii=False) for c in chunks] == fresh
