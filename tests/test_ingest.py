"""tests/test_ingest.py — Stage 1 invariants I1-I5 plus the two failure modes
that produce a plausible-looking but wrong corpus.

The negative tests matter more than the positive ones here: an empty JS-shell
extract and an out-of-scope URL both yield a system that answers confidently
and cites a dead link.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALLOWED_SCHEME_IDS, ALLOWED_URLS, SCHEMES, SCHEME_BY_SLUG
from src.ingest import (
    EmptyExtract,
    IngestError,
    assert_non_empty,
    ingest,
    load_raw_file,
    segment_sections,
)
from src.rawrecord import RawRecord, parse_record, write_record
from src.textutils import normalise_date, resolve_as_of_date, split_sentences

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

#: A body that satisfies every assertion Stage 1 makes, so failure tests isolate
#: the rule under test rather than tripping the fact-completeness check first.
MINIMAL_FACTS = (
    "FUND FACTS (from the page's embedded data payload, verbatim)\n"
    "Fund name: HDFC Large Cap Fund\n"
    "Scheme name (as published): HDFC Large Cap Fund Direct Growth\n"
    "Expense ratio (TER): 1.03%\n"
    "Exit load: Exit load of 1% if redeemed within 1 year\n"
    "Minimum SIP investment: INR 100\n"
    "Minimum lumpsum investment: INR 100\n"
    "Riskometer level: Moderately High Riskometer\n"
    "Benchmark: NIFTY 100 TRI\n"
    "Scheme objective: The scheme seeks to provide long-term capital appreciation.\n"
    "\n"
    "The expense ratio is charged daily. Exit load applies for 1 year.\n"
    "The fund is rated Very High risk. " + "Filler text for bulk. " * 200
)


def _record(**over) -> RawRecord:
    base = dict(
        scheme_id="HDFC_LARGE_CAP", scheme_name="HDFC Large Cap Fund",
        brief_label="Large Cap", category="Equity - Large Cap", plan="Direct Growth",
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        source_tier="broker", brief_reference="problemstatement.txt line 9", slug="large_cap",
        brief_text="problemstatement.txt line 9:\nLarge Cap: https://…",
        page_text="", extracted_at="2026-09-27T00:00:00+00:00",
    )
    base.update(over)
    return RawRecord(**base)


# ── Non-empty extraction (I3) ───────────────────────────────────────────────

def test_js_shell_is_rejected():
    """A 40-char shell is a failed fetch, not a document."""
    with pytest.raises(EmptyExtract):
        assert_non_empty("d", "x" * 40, Path("page.html"))


def test_tag_soup_without_words_is_rejected():
    with pytest.raises(EmptyExtract):
        assert_non_empty("d", "1234 5678 -- | :: || " * 40, Path("page.html"))


def test_real_page_passes():
    assert_non_empty("d", "The expense ratio of the scheme is 0.55% p.a. " * 10, Path("p.txt"))


# ── Scope enforcement (I4 / I5) ────────────────────────────────────────────

def test_out_of_scope_scheme_rejected(tmp_path):
    p = write_record(tmp_path / "bad.txt", _record(
        scheme_id="HDFC_MID_CAP", page_text="x" * 5000,
        source_url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"))
    with pytest.raises(IngestError, match="not in the 5 in-scope schemes"):
        load_raw_file(p)


def test_url_outside_allowlist_rejected(tmp_path):
    """Right scheme, wrong URL — e.g. a blog scraped for facts."""
    p = write_record(tmp_path / "bad.txt", _record(
        page_text="x" * 5000, source_url="https://someblog.in/hdfc-large-cap-review"))
    with pytest.raises(IngestError, match="not in the sources allow-list"):
        load_raw_file(p)


# ── Corpus-level failures must abort the run ───────────────────────────────

def test_partial_corpus_aborts(tmp_path):
    """4 of 5 schemes must fail the build, not quietly produce 4 documents."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for s in SCHEMES[:4]:
        write_record(raw / f"{s.slug}.txt", _record(
            scheme_id=s.scheme_id, scheme_name=s.name, slug=s.slug,
            source_url=s.url, page_text=MINIMAL_FACTS))
    with pytest.raises(IngestError, match="does not cover all 5 schemes"):
        ingest(raw_dir=raw, processed_dir=tmp_path / "p", manifest_dir=tmp_path / "m")


def test_failed_source_aborts_rather_than_skipping(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for s in SCHEMES:
        write_record(raw / f"{s.slug}.txt", _record(
            scheme_id=s.scheme_id, scheme_name=s.name, slug=s.slug, source_url=s.url,
            page_text=MINIMAL_FACTS))
    write_record(raw / "broken.txt", _record(scheme_id="HDFC_ELSS", page_text="tiny"))
    with pytest.raises(IngestError, match="sources failed"):
        ingest(raw_dir=raw, processed_dir=tmp_path / "p", manifest_dir=tmp_path / "m")


def test_corpus_missing_a_required_fact_aborts(tmp_path):
    """A source without the payload facts is unusable, not merely degraded."""
    raw = tmp_path / "raw"
    manifest = tmp_path / "m"
    raw.mkdir()
    for s in SCHEMES:
        body = MINIMAL_FACTS if s.scheme_id != "HDFC_SMALL_CAP" else (
            MINIMAL_FACTS.replace("Riskometer level: Moderately High Riskometer",
                                  "Risk level: not published"))
        write_record(raw / f"{s.slug}.txt", _record(
            scheme_id=s.scheme_id, scheme_name=s.name, slug=s.slug, source_url=s.url,
            page_text=body))
    with pytest.raises(IngestError, match="sources failed"):
        ingest(raw_dir=raw, processed_dir=tmp_path / "p", manifest_dir=manifest)

    # The rejection must be recorded, not just printed. "Reject, never skip" is
    # only auditable if the reason survives the run.
    rows = [json.loads(l) for l in (manifest / "ingest_manifest.jsonl").open(encoding="utf-8")]
    failed = [r for r in rows if r["status"] == "failed"]
    assert len(failed) == 1
    assert "missing required fact 'Riskometer level'" in failed[0]["error"]
    assert "small_cap" in failed[0]["file"]


# ── Record round-trip ──────────────────────────────────────────────────────

def test_record_roundtrip(tmp_path):
    rec = _record(page_text="# Fees\n\nThe expense ratio is 0.55% p.a.",
                  page_title="T", http_status="200", content_chars="42")
    p = write_record(tmp_path / "r.txt", rec)
    back = parse_record(p)
    assert back.scheme_id == rec.scheme_id
    assert back.source_url == rec.source_url
    assert back.page_title == "T"
    assert "expense ratio" in back.page_text
    assert "problemstatement.txt line 9" in back.brief_text


def test_malformed_record_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("no provenance header here", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        parse_record(p)


# ── Section segmentation ───────────────────────────────────────────────────

def test_sections_keep_exit_load_separate():
    text = ("# Fees and charges\n\nThe expense ratio is 0.55% p.a.\n\n"
            "# Exit load\n\n | Period | Exit load |\n | 0-12 months | 1.00% |\n")
    secs = segment_sections(text)
    assert any("expense ratio" in s["text"].lower() for s in secs)
    assert any("exit load" in s["text"].lower() for s in secs)
    assert any("exit load" in s["heading"].lower() for s in secs)


def test_key_value_lines_are_never_treated_as_headings():
    """Regression: the payload facts block lost its values to heading detection.

    "Expense ratio (TER): 1.03%" must land in a section body. When the segmenter
    read it as a heading and stripped the line, the headline fact silently
    vanished while the label still appeared in the section list.
    """
    facts = ("FUND FACTS\n"
             "Fund name: HDFC Large Cap Fund\n"
             "Expense ratio (TER): 1.03%\n"
             "Riskometer level: Moderately High Riskometer\n")
    full = "\n\n".join(s["text"] for s in segment_sections(facts))
    for value in ("1.03%", "Moderately High Riskometer", "HDFC Large Cap Fund"):
        assert value in full, f"{value!r} was consumed as a heading"


def test_table_rows_are_never_treated_as_headings():
    text = "Fund facts\n\n| ICICI Bank Ltd | Financial | 10.05% |\n| HDFC Bank Ltd | Financial | 6.88% |"
    full = "\n\n".join(s["text"] for s in segment_sections(text))
    assert "ICICI Bank Ltd" in full
    assert "HDFC Bank Ltd" in full


def test_no_section_is_empty():
    secs = segment_sections("# A\n\nBody A.\n\n# B\n\nBody B.")
    assert secs and all(s["text"].strip() for s in secs)


# ── Dates (Q6) ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("2026-06-30", "2026-06-30"),
    ("30-06-2026", "2026-06-30"),
    ("30 June 2026", "2026-06-30"),
    ("June 30, 2026", "2026-06-30"),
    ("30/06/26", "2026-06-30"),
    ("not a date", None),
    ("", None),
])
def test_normalise_date(raw, expect):
    assert normalise_date(raw) == expect


def test_date_precedence_prefers_document_text(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x", encoding="utf-8")
    d, src = resolve_as_of_date("NAV as on 30 June 2026", {"last_modified": "2020-01-01"}, p)
    assert (d, src) == ("2026-06-30", "document_text")


def test_date_falls_back_to_mtime(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x", encoding="utf-8")
    d, src = resolve_as_of_date("no date here", {}, p)
    assert d is not None and src == "file_mtime"


# ── Sentence splitter (R1 binding rules, needed from Phase 1 onward) ───────

@pytest.mark.parametrize("text,expect", [
    ("The expense ratio is 0.55%. Min SIP is Rs. 500.", 2),
    ("Fees, e.g. exit load, apply. See the note.", 2),
    ("- Exit load 1% for 12m\n- No load after 12m", 2),
    ("One. Two. Three.", 3),
    ("No terminal punctuation", 1),
])
def test_split_sentences(text, expect):
    assert len(split_sentences(text)) == expect


def test_decimal_is_not_a_boundary():
    assert len(split_sentences("The ratio is 1.25% p.a. and the exit load is 1.00%.")) == 1


# ── The real corpus on disk ────────────────────────────────────────────────

@pytest.mark.skipif(not (RAW_DIR / "large_cap.txt").exists(), reason="corpus not fetched yet")
def test_real_corpus_covers_all_five():
    docs, _ = ingest()
    assert len(docs) == 5
    assert {d.scheme_id for d in docs} == ALLOWED_SCHEME_IDS
    assert len({d.content_hash for d in docs}) == 5


@pytest.mark.skipif(not (RAW_DIR / "large_cap.txt").exists(), reason="corpus not fetched yet")
def test_real_corpus_is_substantive():
    """Catches the JS-shell regression on the actual fetched pages."""
    for doc in ingest()[0]:
        assert doc.char_count > 5000, f"{doc.doc_id} only {doc.char_count} chars"
        assert doc.sections, f"{doc.doc_id} has no sections"
        assert doc.as_of_date, f"{doc.doc_id} has no as_of_date (R7 would print blank)"
        assert doc.source_url in ALLOWED_URLS
