"""ingest.py — Stage 1: Loading.

Walks ``data/raw/``, parses each raw record, and emits
``data/processed/documents.jsonl`` plus ``data/manifest/ingest_manifest.jsonl``.

Design rules (architecture.md §5):
  * The directory is the interface. Adding a source means dropping a file in.
  * Reject, never skip. A file that fails validation fails the run loudly —
    a silently-dropped source is a silently-broken citation.
  * as_of_date is resolved by the PRD Q6 chain and never guessed, because R7
    prints it to the user.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    ALLOWED_SCHEME_IDS,
    ALLOWED_URLS,
    EXPECTED_PAGE_CHARS,
    MANIFEST_DIR,
    MIN_EXTRACT_CHARS,
    PROCESSED_DIR,
    RAW_DIR,
    ROOT as PROJECT_ROOT,
    SCHEME_BY_ID,
)
from src.rawrecord import RawRecord, parse_record
from src.textutils import normalise_date, normalise_whitespace, resolve_as_of_date, split_sentences

#: Heading levels used for section segmentation. Depth is capped at 4 because
#: a deeper hierarchy fragments facts across too many chunks.
_MD_HEADING_RE = re.compile(r"^(#{1,5})\s+(.+?)\s*$")
#: A bare heading line in fetched text: short, title-ish, no punctuation.
_LOOSE_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 /&()'’,-]{2,79}$")
#: Table rows are data, never headings.
_TABLE_LINE_RE = re.compile(r"^\s*\|")
#: "Label: value" is the payload's own shape. Treating these as headings strips
#: the value out of the body and silently deletes the fact — the exact failure
#: that lost "Expense ratio (TER): 1.03%" on the first run.
_KEYVALUE_RE = re.compile(r"^[A-Z][^:]{0,70}:\s*\S")


class IngestError(Exception):
    """Raised when a source must fail the run."""


class EmptyExtract(IngestError):
    """Extraction produced too little usable text (the JS-shell failure mode)."""


@dataclass
class Document:
    doc_id: str
    scheme_id: str
    scheme_name: str
    brief_label: str
    category: str
    plan: str
    doc_type: str
    source_tier: str
    source_url: str
    title: str
    brief_reference: str
    raw_path: str
    text: str
    sections: list[dict] = field(default_factory=list)
    as_of_date: str | None = None
    as_of_date_source: str = ""
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    content_hash: str = ""
    ingested_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ── Validation ─────────────────────────────────────────────────────────────

def assert_non_empty(doc_id: str, text: str, path: Path) -> None:
    """Invariant I3 — the single most important guard in Stage 1.

    A JS-rendered page fetched without a browser yields an empty shell. Without
    this check you get an empty document, a dead citation slot, and a chunk
    count that looks perfectly healthy.
    """
    stripped = text.strip()
    if len(stripped) < MIN_EXTRACT_CHARS:
        raise EmptyExtract(f"{doc_id}: {len(stripped)} chars (< {MIN_EXTRACT_CHARS}) in {path.name}")
    if not re.search(r"[A-Za-z]{3,}", stripped):
        raise EmptyExtract(f"{doc_id}: no alphabetic content in {path.name}")


def validate_scope(doc_id: str, rec: RawRecord) -> None:
    """Invariants I4/I5 — nothing outside the 5 schemes or the URL allow-list."""
    if rec.scheme_id not in ALLOWED_SCHEME_IDS:
        raise IngestError(f"{doc_id}: scheme_id {rec.scheme_id!r} not in the 5 in-scope schemes")
    if rec.source_url not in ALLOWED_URLS:
        raise IngestError(f"{doc_id}: url {rec.source_url!r} not in the sources allow-list")


# ── Section segmentation ───────────────────────────────────────────────────

def segment_sections(text: str) -> list[dict]:
    """Split extracted text into sections so chunks never straddle a heading.

    Walks *lines* and records real offsets, so a section's body always begins
    immediately after its own heading. Two classes of line are explicitly never
    headings, because treating them as one deletes the fact they carry:

      * ``Label: value`` lines (the payload facts block)
      * table rows (``| a | b |``)

    Keeping the exit-load slab and the expense-ratio entry in their own sections
    is what makes them retrievable as self-contained facts in Phase 2.
    """
    text = normalise_whitespace(text)
    if not text:
        return []

    lines = text.split("\n")

    def is_heading(i: int) -> str | None:
        s = lines[i].strip()
        if not s:
            return None
        md = _MD_HEADING_RE.match(s)
        if md:
            return md.group(2).strip()
        if _KEYVALUE_RE.match(s):          # data, not structure
            return None
        if _TABLE_LINE_RE.match(s):         # data, not structure
            return None
        if s.endswith((".", ",", ":", ";")):
            return None
        if not _LOOSE_HEADING_RE.match(s):
            return None
        # A heading with nothing under it is a stray line, not a heading.
        if not any(l.strip() for l in lines[i + 1:i + 6]):
            return None
        return s

    heads: list[tuple[int, str]] = []
    for i in range(len(lines)):
        h = is_heading(i)
        if h:
            heads.append((i, h))

    def make(heading: str, body_lines: list[str], path: list[str]) -> dict | None:
        body = "\n".join(body_lines).strip()
        if not body:
            return None
        return {"heading": heading, "heading_path": path,
                "text": body, "char_count": len(body)}

    sections: list[dict] = []
    if not heads:
        one = make("Full document", lines, ["Full document"])
        return [one] if one else []

    for n, (idx, heading) in enumerate(heads):
        start = idx + 1
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        path = [h for _, h in heads[max(0, n - 2):n + 1]]
        sec = make(heading, lines[start:end], path)
        if sec:
            sections.append(sec)

    if not sections:
        one = make("Full document", lines, ["Full document"])
        return [one] if one else []
    return sections


# ── The loader ─────────────────────────────────────────────────────────────

def _doc_type_for(rec: RawRecord) -> str:
    if "factsheet" in rec.source_url.lower():
        return "factsheet"
    if "faq" in rec.source_url.lower():
        return "scheme_faq"
    return "scheme_page"


def _relative_to_project(path: Path) -> str:
    """Project-relative path when possible, absolute otherwise.

    ``Path.relative_to`` raises when the source lives outside the repo (a temp
    fixture, an ad-hoc corpus), and an audit field is not worth failing a load.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_raw_file(path: Path) -> Document:
    """Parse one raw record into a :class:`Document`, or raise :class:`IngestError`."""
    rec = parse_record(path)
    doc_id = f"{rec.scheme_id}__{_doc_type_for(rec)}"

    validate_scope(doc_id, rec)

    # Page text is the corpus; the brief line is provenance, not answerable
    # content, so it is recorded but not fed to the chunker.
    body = rec.page_text.strip()
    assert_non_empty(doc_id, body, path)

    sections = segment_sections(body)
    full_text = "\n\n".join(s["text"] for s in sections)

    # The fetcher already resolved a document-level date from the payload's
    # nav_date. Trust it over a regex sweep, which can latch onto a "returns as
    # on" line inside a table and date the whole document to the wrong day.
    as_of, as_of_source = None, "unresolved"
    if rec.as_of_date:
        as_of, as_of_source = normalise_date(rec.as_of_date), "payload_nav_date"
        as_of = as_of or rec.as_of_date
    if not as_of:
        as_of, as_of_source = resolve_as_of_date(
            rec.page_text, {"last_modified": rec.fetched_at}, path,
        )
    if not as_of:
        as_of, as_of_source = None, "unresolved"

    # Every source must carry the payload facts, or the brief's headline topics
    # (expense ratio, riskometer) are unanswerable no matter how good retrieval
    # is. This is a corpus-completeness assertion, not a quality nicety.
    for required in ("Expense ratio", "Exit load", "Minimum SIP investment",
                     "Riskometer level", "Benchmark"):
        if required not in full_text:
            raise IngestError(
                f"{doc_id}: corpus is missing required fact {required!r} — "
                f"re-run scripts/fetch_page_text.py"
            )

    scheme = SCHEME_BY_ID[rec.scheme_id]
    return Document(
        doc_id=doc_id,
        scheme_id=rec.scheme_id,
        scheme_name=scheme.name,
        brief_label=rec.brief_label,
        category=scheme.category,
        plan=scheme.plan,
        doc_type=_doc_type_for(rec),
        source_tier=rec.source_tier or "broker",
        source_url=rec.source_url,
        title=rec.page_title or scheme.name,
        brief_reference=rec.brief_reference,
        raw_path=_relative_to_project(path),
        text=full_text,
        sections=sections,
        as_of_date=as_of,
        as_of_date_source=as_of_source,
        char_count=len(full_text),
        word_count=len(full_text.split()),
        sentence_count=len(split_sentences(full_text)),
        content_hash=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def ingest(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR,
           manifest_dir: Path = MANIFEST_DIR) -> tuple[list[Document], list[dict]]:
    """Load every record under ``raw_dir``. Returns ``(documents, manifest_rows)``.

    Fails the run on any invalid source. A partial corpus that looks complete is
    worse than a failed build, because the failure surfaces in a demo.
    """
    if not raw_dir.exists():
        raise IngestError(f"raw directory not found: {raw_dir}")

    files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"})
    if not files:
        raise IngestError(f"no .txt/.md sources in {raw_dir} — run scripts/extract_brief_records.py")

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "documents.jsonl"

    docs: list[Document] = []
    rows: list[dict] = []
    failures: list[str] = []
    seen: set[str] = set()

    for path in files:
        rel = path.relative_to(raw_dir)
        try:
            doc = load_raw_file(path)
        except (IngestError, ValueError) as exc:
            failures.append(f"{rel}: {exc}")
            rows.append({"file": str(rel), "status": "failed", "error": str(exc)})
            continue

        if doc.doc_id in seen:                      # invariant I1
            failures.append(f"{rel}: duplicate doc_id {doc.doc_id}")
            rows.append({"file": str(rel), "status": "failed", "error": "duplicate doc_id"})
            continue
        seen.add(doc.doc_id)

        if doc.char_count < EXPECTED_PAGE_CHARS:     # soft warning, not a failure
            rows.append({"file": str(rel), "status": "warning", "doc_id": doc.doc_id,
                         "error": f"only {doc.char_count} chars — likely a partial page"})
        docs.append(doc)
        rows.append({
            "file": str(rel), "status": "ok", "doc_id": doc.doc_id,
            "scheme_id": doc.scheme_id, "doc_type": doc.doc_type,
            "url": doc.source_url, "char_count": doc.char_count,
            "section_count": len(doc.sections), "as_of_date": doc.as_of_date,
            "as_of_date_source": doc.as_of_date_source,
            "content_hash": doc.content_hash, "ingested_at": doc.ingested_at,
        })

    def flush_manifest() -> None:
        """Always persist the manifest, including on failure.

        "Reject, never skip" is only useful if the rejection is recorded. A run
        that dies with the reason only on stderr leaves no record of which
        source was rejected and why.
        """
        (manifest_dir / "ingest_manifest.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )

    if failures:
        flush_manifest()
        for f in failures:
            print(f"  FAIL  {f}", file=sys.stderr)
        raise IngestError(
            f"{len(failures)} of {len(files)} sources failed. Corpus is not trustworthy; aborting."
        )

    covered = {d.scheme_id for d in docs}          # invariant I2
    if covered != ALLOWED_SCHEME_IDS:
        missing = sorted(ALLOWED_SCHEME_IDS - covered)
        flush_manifest()
        raise IngestError(f"corpus does not cover all 5 schemes; missing {missing}")

    out_path.write_text("".join(d.to_json() + "\n" for d in docs), encoding="utf-8")
    flush_manifest()
    return docs, rows


# ── Verification report (implementation.md §1.4) ──────────────────────────

def report(docs: list[Document], rows: list[dict]) -> str:
    lines = [
        "=" * 78,
        "PHASE 1 VERIFICATION — data/processed/documents.jsonl",
        f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 78,
        "",
        f"{'doc_id':<34} {'chars':>7} {'sec':>4} {'sents':>6}  as_of",
        "-" * 78,
    ]
    for d in docs:
        lines.append(f"{d.doc_id:<34} {d.char_count:>7} {len(d.sections):>4} "
                     f"{d.sentence_count:>6}  {d.as_of_date} ({d.as_of_date_source})")
    warns = [r for r in rows if r["status"] == "warning"]
    failed = [r for r in rows if r["status"] == "failed"]
    lines += [
        "-" * 78,
        f"documents      : {len(docs)}",
        f"schemes covered: {len({d.scheme_id for d in docs})}/5",
        f"warnings       : {len(warns)}",
        f"failed         : {len(failed)}  (must be 0)",
        "",
        f"[{'PASS' if not failed else 'FAIL'}] I1  unique doc_id per source",
        f"[{'PASS' if len({d.scheme_id for d in docs}) == 5 else 'FAIL'}] I2  all 5 schemes covered",
        f"[{'PASS' if all(len(d.sections) > 1 for d in docs) else 'WARN'}] I3  section segmentation produced >1 section",
        f"[{'PASS' if all(d.scheme_id in ALLOWED_SCHEME_IDS for d in docs) else 'FAIL'}] I4  scheme_id in allow-list",
        f"[{'PASS' if all(d.source_url in ALLOWED_URLS for d in docs) else 'FAIL'}] I5  url in allow-list",
        f"[{'PASS' if all(d.as_of_date for d in docs) else 'FAIL'}] Q6  as_of_date resolved (R7 depends on it)",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    documents, manifest_rows = ingest()
    print(report(documents, manifest_rows))
