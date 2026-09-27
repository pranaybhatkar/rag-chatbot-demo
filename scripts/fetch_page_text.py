"""fetch_page_text.py — fill Section 2 of each raw record from its source URL.

Fetch is deliberately a separate step from loading (implementation.md §1.2):
``data/raw/`` stays a real directory of files you can inspect, and the fetch is
re-runnable without touching the loader.

Writes a ``.fetch.json`` sidecar next to each record so the HTTP ``Last-Modified``
header survives for the PRD Q6 as_of_date chain even when the page shows no date.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXPECTED_PAGE_CHARS, MIN_EXTRACT_CHARS, RAW_DIR, ROOT
from src.rawrecord import parse_record, write_record
from src.structured import FACTS_HEADING, extract_facts

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-IN,en;q=0.9",
}
TIMEOUT = 45

#: Chrome/boilerplate that survives naive tag-stripping and pollutes chunks.
_DROP_TAGS = ("script", "style", "noscript", "svg", "iframe", "footer", "nav")
#: Class-name pruning is deliberately minimal. An earlier, greedier list
#: containing "header" and "nav-" silently deleted the whole "how to download
#: statements" section, which is one of the brief's own seven topics — a
#: removal that looks like an improvement and costs a required capability.
#: Verify with scripts/verify_phase1.py before widening this list.
_DROP_CLASSES = ("footer", "navbar", "breadcrumb", "cookie-banner", "popup-modal")
#: Blocks whose text is mostly a list of other scheme names.
_SIDEBAR_RE = re.compile(
    r"(HDFC [A-Z][A-Za-z ]+Fund Direct (?:Growth|Plan Growth)\s*){4,}")
_KEEP_TAGS = ("h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th", "table",
              "tr", "br", "div", "span", "section", "article")


def extract_text(html: str) -> str:
    """Render an HTML page to readable plain text, keeping table structure.

    Headings become ``#``-prefixed lines so Stage 1's section segmenter can find
    them; table rows become pipe-separated lines so exit-load slabs and
    expense-ratio grids survive as tables rather than collapsing into prose.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    # Decomposing a node nulls the attrs of its descendants, which are still in
    # the snapshot from find_all — skip those rather than crashing.
    for node in soup.find_all(class_=True):
        raw_cls = node.attrs.get("class") if getattr(node, "attrs", None) else None
        if not isinstance(raw_cls, (list, tuple)):
            continue
        cls = " ".join(raw_cls).lower()
        if any(c in cls for c in _DROP_CLASSES):
            node.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    parts: list[str] = []
    for el in root.find_all(_KEEP_TAGS):
        if el.name in ("h1", "h2", "h3", "h4", "h5"):
            t = el.get_text(" ", strip=True)
            if t:
                parts.append(f"\n{'#' * int(el.name[1])} {t}\n")
        elif el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(("td", "th"))]
            if any(cells):
                parts.append(" | " + " | ".join(cells) + " |\n")
        elif el.name in ("p", "li", "div", "section", "article"):
            if el.find(("p", "li", "table", "div", "section")):
                continue  # container; its children are handled individually
            t = el.get_text(" ", strip=True)
            if t and not _SIDEBAR_RE.search(t):
                parts.append(f"{t}\n")

    text = "".join(parts)
    lines = [ln.strip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def fetch_one(path: Path) -> tuple[bool, str]:
    rec = parse_record(path)
    try:
        r = requests.get(rec.source_url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return False, f"network error: {exc}"

    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    # Groww serves no charset or a latin-1 default; force UTF-8 or the rupee
    # sign and curly quotes turn into mojibake in the corpus.
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
        r.encoding = "utf-8"

    text = extract_text(r.text)
    if len(text) < MIN_EXTRACT_CHARS:
        return False, f"extraction produced only {len(text)} chars (JS shell?)"

    # The authoritative scalars (expense ratio, riskometer, min SIP) are only in
    # the embedded payload — the rendered DOM shows the label with no value.
    facts, values = extract_facts(r.text)
    if not facts:
        return False, ("no fund payload found (__NEXT_DATA__ missing) — expense ratio "
                       "and riskometer would be absent from the corpus")
    if "Expense ratio" not in facts:
        return False, "fund payload found but carries no expense ratio"

    # Facts first: they are the answerable scalars, and a chunker that reads
    # top-down should meet them before the prose glossary definitions.
    text = f"{facts}\n\n{text}"

    title = ""
    if r.text and "<title" in r.text:
        i = r.text.find("<title")
        j = r.text.find(">", i)
        k = r.text.find("</title>", j)
        if 0 < j < k:
            title = r.text[j + 1:k].strip()

    rec.page_text = text
    rec.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec.http_status = str(r.status_code)
    rec.page_title = title
    rec.content_chars = str(len(text))
    # nav_date is the document's own as-of date; prefer it over the NAV-date
    # regex, which can pick up a "returns as on" line from a table.
    if values.get("nav_date"):
        rec.as_of_date = str(values["nav_date"])
    rec.extra["facts_source"] = FACTS_HEADING
    rec.extra["expense_ratio"] = str(values.get("expense_ratio", ""))
    rec.extra["riskometer"] = str(values.get("nfo_risk", ""))
    rec.extra["benchmark"] = str(values.get("benchmark", ""))
    rec.extra["min_sip"] = str(values.get("min_sip_investment", ""))
    rec.extra["exit_load"] = str(values.get("exit_load", ""))

    last_mod = r.headers.get("Last-Modified", "")
    rec.extra["last_modified"] = last_mod
    write_record(path, rec)
    side = path.with_suffix(".fetch.json")
    side.write_text(json.dumps({
        "url": rec.source_url, "http_status": r.status_code, "fetched_at": rec.fetched_at,
        "last_modified": last_mod, "page_title": title, "content_chars": len(text),
        "content_type": r.headers.get("Content-Type", ""),
        "as_of_date": rec.as_of_date, "facts": values,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    note = "" if len(text) >= EXPECTED_PAGE_CHARS else "  <-- below expected 5000 chars"
    return True, f"{len(text):>6} chars  ER={values.get('expense_ratio')}{note}"


def main() -> int:
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        print("no raw records - run scripts/extract_brief_records.py first", file=sys.stderr)
        return 1

    print(f"fetching {len(files)} source pages -> {RAW_DIR.relative_to(ROOT)}\n")
    bad = 0
    for path in files:
        ok, msg = fetch_one(path)
        if ok:
            print(f"  OK    {path.name:<24} {msg}")
        else:
            bad += 1
            print(f"  FAIL  {path.name:<24} {msg}", file=sys.stderr)

    print()
    if bad:
        print(f"{bad}/{len(files)} fetches failed. Stage 1 will refuse to proceed "
              f"with a partial corpus.", file=sys.stderr)
        return 1
    print("all pages fetched. Next:  .\\.venv\\Scripts\\python.exe scripts\\build_index.py --stages 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
