"""extract_brief_records.py — turn problemstatement.txt into data/raw/*.txt.

The brief names the 5 schemes on lines 9-13, one per line, as
``<label>: <url>``. This script parses exactly that and writes one plain-text
raw record per scheme into ``data/raw/``.

What the brief does NOT contain: any scheme facts. There are no expense ratios,
exit loads, lock-in periods, or benchmarks in problemstatement.txt — only a
category label and a URL. So the record created here is provenance, not corpus.
``scripts/fetch_page_text.py`` then fills Section 2 with the real page text;
without it the corpus has nothing to retrieve.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR, ROOT, SCHEMES
from src.rawrecord import RawRecord, write_record

BRIEF = ROOT / "problemstatement.txt"
#: Brief lines 9-13 read "Large Cap: https://groww.in/...". The label must start
#: with a letter and stay short so a long prose sentence that happens to contain
#: a URL cannot masquerade as a scheme declaration.
_LINE_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z][^:]{0,60}?)\s*:\s*(?P<url>https?://\S+)\s*$"
)


def parse_brief(path: Path = BRIEF) -> dict[str, tuple[int, str, str]]:
    """-> {url: (line_number, label, verbatim_line)} for every ``label: url`` line."""
    if not path.exists():
        raise SystemExit(f"brief not found: {path}")
    found: dict[str, tuple[int, str, str]] = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = _LINE_RE.match(raw)
        if m:
            found[m["url"]] = (n, m["label"].strip(), raw.strip())
    return found


def main() -> int:
    lines = parse_brief()
    print(f"parsed problemstatement.txt: {len(lines)} 'label: url' lines\n")

    by_url = {s.url: s for s in SCHEMES}
    unmatched = sorted(set(lines) - set(by_url))
    missing = sorted(set(by_url) - set(lines))
    for u in unmatched:
        print(f"  NOTE  brief line not mapped to an in-scope scheme: {u}")
    for u in missing:
        print(f"  FAIL  in-scope scheme absent from the brief: {u}")
    if missing:
        return 1

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for scheme in SCHEMES:
        line_no, label, verbatim = lines[scheme.url]
        rec = RawRecord(
            scheme_id=scheme.scheme_id,
            scheme_name=scheme.name,
            brief_label=label,
            category=scheme.category,
            plan=scheme.plan,
            source_url=scheme.url,
            source_tier="broker",
            brief_reference=f"problemstatement.txt line {line_no}",
            slug=scheme.slug,
            extracted_at=now,
            brief_text=f"problemstatement.txt line {line_no}:\n{verbatim}",
            page_text="",  # filled by scripts/fetch_page_text.py
            extra={"brief_label_matched": "yes" if label == scheme.brief_label else "NO"},
        )
        path = write_record(RAW_DIR / f"{scheme.slug}.txt", rec)
        flag = "" if label == scheme.brief_label else f"  (brief said {label!r})"
        print(f"  wrote {path.relative_to(ROOT)}  <- line {line_no} {scheme.scheme_id}{flag}")
        written.append(path)

    print(f"\n{len(written)}/5 raw records created in {RAW_DIR.relative_to(ROOT)}")
    print("Section 2 is intentionally empty: the brief supplies no scheme facts.")
    print("Next:  .\\.venv\\Scripts\\python.exe scripts\\fetch_page_text.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
