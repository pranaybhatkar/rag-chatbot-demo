"""textutils.py — text normalisation and date parsing for the ingestion stage.

Deliberately dependency-free. R1's sentence segmentation lives here too, so the
3-sentence rule is verifiable without loading a model and the same input always
produces the same sentence count.
"""
from __future__ import annotations

import re
from datetime import date, datetime

_SENT = "\x00"

ABBREVIATIONS: tuple[str, ...] = (
    "e.g.", "i.e.", "etc.", "vs.", "viz.", "cf.", "al.",
    "approx.", "resp.", "incl.", "excl.",
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.",
    "No.", "Nos.", "Rs.", "Lakh", "Lakhs", "Cr.", "Crore", "Crores",
    "p.a.", "w.r.t.", "a.m.", "p.m.",
)

_BULLET_RE = re.compile(r"^\s*(?:[-*•–—]|\d+[.)])\s+", re.MULTILINE)
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_TERMINAL_RE = re.compile(r"([.!?]+)(\s+|$)")

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalise_whitespace(text: str) -> str:
    """Collapse whitespace runs, preserving newlines that delimit list items."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise_date(value: str | None) -> str | None:
    """Parse a date into ISO ``YYYY-MM-DD``. Returns None if unparseable.

    Accepts 2026-06-30, 30-06-2026, 30/06/2026, 30 June 2026, June 30 2026.
    Never guesses: an unparseable date returns None so the caller can fall
    through the PRD Q6 precedence chain rather than inventing a value.
    """
    if not value:
        return None
    s = value.strip().rstrip(".,;")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})[-/\s]([A-Za-z]{3,9}|\d{1,2})[-/\s](\d{2,4})", s)
    if m:
        d, mid, y = m[1], m[2], m[3]
        try:
            if mid.isdigit():
                mo, dd = int(mid), int(d)
            else:
                mo, dd = _MONTHS.get(mid[:4].lower()) or _MONTHS.get(mid[:3].lower()), int(d)
                if mo is None:
                    return None
            yy = int(y) + 2000 if len(y) == 2 else int(y)
            return date(yy, mo, dd).isoformat()
        except (ValueError, TypeError):
            return None
    m = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})", s)
    if m:
        month = m[1][:4].lower()
        mo = _MONTHS.get(month) or _MONTHS.get(month[:3])
        if mo:
            try:
                return date(int(m[3]), mo, int(m[2])).isoformat()
            except ValueError:
                return None
    return None


def resolve_as_of_date(text: str, meta: dict | None = None, path=None) -> tuple[str | None, str]:
    """PRD Q6 precedence: document text → HTTP Last-Modified → file mtime.

    Returns ``(iso_date_or_None, provenance)``. A wrong date is a visible defect
    because R7 prints it, so nothing here ever guesses.
    """
    meta = meta or {}
    m = re.search(
        r"(?:as\s+of|last\s+updated\s+on|updated\s+as\s+on|as\s+on)\s*:?\s*"
        r"(\d{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})",
        text, re.I,
    )
    if m:
        d = normalise_date(m.group(1))
        if d:
            return d, "document_text"
    if meta.get("last_modified"):
        d = normalise_date(meta["last_modified"])
        if d:
            return d, "http_last_modified"
    if path is not None and path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime).date().isoformat(), "file_mtime"
    return None, "unknown"


def split_sentences(text: str) -> list[str]:
    """Split into sentences per the PRD §4.1 binding definition.

    Abbreviations (e.g., i.e., vs., Rs., No.) and decimals (0.55%, 1.25) are
    NOT boundaries. Each list item is a sentence candidate.
    """
    if not text or not text.strip():
        return []
    text = normalise_whitespace(text)

    out: list[str] = []
    for raw in _BULLET_RE.split(text):
        seg = raw.strip()
        if not seg:
            continue
        p = _DECIMAL_RE.sub(_SENT, seg)
        for abbr in ABBREVIATIONS:
            if abbr.endswith("."):
                p = re.sub(
                    rf"(?<![A-Za-z]){re.escape(abbr)}(?=\s|$)",
                    abbr[:-1] + _SENT, p, flags=re.I,
                )
        pieces: list[str] = []
        last = 0
        for m in _TERMINAL_RE.finditer(p):
            end = m.end(1)
            if end <= last:
                continue
            if p[last:end].strip():
                pieces.append(p[last:end])
            last = m.end()
        if p[last:].strip():
            pieces.append(p[last:])

        for piece in pieces:
            s = piece.replace(_SENT, ".").strip().strip(" \t\n\"'“”‘’()[]")
            if s:
                out.append(s)
    return out


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


def is_complete_sentence(sentence: str) -> bool:
    """True when the sentence terminates cleanly (not a truncation fragment)."""
    s = sentence.strip()
    if not s:
        return False
    if s[-1] in ".!?":
        return True
    if s.endswith(_SENT) or s.lower().endswith(tuple(a.rstrip(".") for a in ABBREVIATIONS)):
        return True
    return False
