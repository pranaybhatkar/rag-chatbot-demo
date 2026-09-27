"""verify_phase1.py — Phase 1 acceptance check.

Run after scripts/build_index.py --stages 1. Exits non-zero on any hard
failure so it can gate a commit or a demo.

    .\\.venv\\Scripts\\python.exe scripts\\verify_phase1.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ALLOWED_SCHEME_IDS,
    ALLOWED_URLS,
    BRIEF_TOPIC_PROBES,
    EXPECTED_PAGE_CHARS,
    PROCESSED_DIR,
    SCHEMES,
)

#: Facts the brief's own example questions depend on. If one of these is
#: missing the chatbot cannot answer, no matter how good retrieval is.
REQUIRED_FACTS = (
    "Expense ratio (TER)",
    "Exit load",
    "Minimum SIP investment",
    "Minimum lumpsum investment",
    "Riskometer level",
    "Benchmark",
    "Fund name",
    "Scheme objective",
)

#: Visible-DOM content that must survive pruning, keyed by topic.
DOM_PROBES = {
    "exit load slab":       r"[Ee]xit load of \d",
    "risk rating":          r"rated\s+\w+(?:\s+\w+)?\s+risk",
    "statement guidance":   r"capital gains|statement",
    "stamp duty":           r"stamp duty",
    "holdings table":       r"\|\s*\w+\s+(?:Ltd|Limited)\s*\|",
}

#: R4 risk indicator: a return figure visible to the generator. Presence here is
#: expected and acceptable (the PRD treats figures as reference material), but
#: every one is an opportunity for AC-13 to fail, so it is reported explicitly.
_RETURN_RE = re.compile(r"[+-]?\s*\d+\.?\d*\s*%")


def main() -> int:
    path = PROCESSED_DIR / "documents.jsonl"
    if not path.exists():
        print("FAIL  documents.jsonl not found - run --stages 1", file=sys.stderr)
        return 1

    docs = [json.loads(l) for l in path.open(encoding="utf-8")]
    failures: list[str] = []
    warns: list[str] = []

    print("=" * 96)
    print("PHASE 1 ACCEPTANCE CHECK")
    print("=" * 96)
    print(f"\n{'doc_id':<32} {'chars':>7} {'sec':>4} {'facts':>6} {'sents':>6}  as_of")
    print("-" * 96)
    for d in docs:
        facts = sum(1 for f in REQUIRED_FACTS if f in d["text"])
        print(f"{d['doc_id']:<32} {d['char_count']:>7} {len(d['sections']):>4} "
              f"{facts:>3}/{len(REQUIRED_FACTS)} {d['sentence_count']:>6}  {d['as_of_date']}")
        if facts < len(REQUIRED_FACTS):
            missing = [f for f in REQUIRED_FACTS if f not in d["text"]]
            failures.append(f"{d['doc_id']}: missing facts {missing}")
    print("-" * 96)

    # 1. Coverage
    got = {d["scheme_id"] for d in docs}
    if got != ALLOWED_SCHEME_IDS:
        failures.append(f"scheme coverage mismatch: {sorted(ALLOWED_SCHEME_IDS - got)}")
    print(f"[{'PASS' if got == ALLOWED_SCHEME_IDS else 'FAIL'}] covers all 5 schemes")

    # 2. Uniqueness / determinism
    if len({d["content_hash"] for d in docs}) != len(docs):
        failures.append("duplicate content_hash - two sources are the same page")
    if len({d["doc_id"] for d in docs}) != len(docs):
        failures.append("duplicate doc_id")
    print(f"[{'PASS' if len({d['content_hash'] for d in docs}) == len(docs) else 'FAIL'}] "
          f"unique doc_id + content_hash")

    # 3. Citations
    for d in docs:
        if d["source_url"] not in ALLOWED_URLS:
            failures.append(f"{d['doc_id']}: url outside allow-list")
        if not d["as_of_date"]:
            failures.append(f"{d['doc_id']}: no as_of_date (R7 would print blank)")
    print(f"[{'PASS' if not failures else 'FAIL'}] every url in allow-list, every as_of_date set")

    # 4. Brief topic coverage
    print("\nbrief topic coverage (probe = regex from config.BRIEF_TOPIC_PROBES)")
    corpus = "\n".join(d["text"] for d in docs)
    for topic, pat in BRIEF_TOPIC_PROBES.items():
        n = len(re.findall(pat, corpus, re.I))
        mark = "PASS" if n else "MISS"
        if not n:
            failures.append(f"brief topic never covered: {topic}")
        print(f"  [{mark}] {topic:<20} {n:>5} mentions")

    # 5. Visible-DOM survival after pruning
    print("\npruning safety: content that must survive noise removal")
    for name, pat in DOM_PROBES.items():
        n = len(re.findall(pat, corpus, re.I))
        mark = "PASS" if n else "LOST"
        if not n:
            failures.append(f"pruning removed all '{name}' content")
        print(f"  [{mark}] {name:<20} {n:>5} mentions")

    # 6. Substantiveness
    thin = [d["doc_id"] for d in docs if d["char_count"] < EXPECTED_PAGE_CHARS]
    if thin:
        warns.append(f"thin documents (<{EXPECTED_PAGE_CHARS} chars): {thin}")

    # 7. R4 exposure - informational, never a failure
    print("\nR4 exposure (return figures visible to the generator)")
    total = 0
    for d in docs:
        n = len(_RETURN_RE.findall(d["text"]))
        total += n
        print(f"  {d['scheme_id']:<18} {n:>4} percentages in corpus text")
    print(f"  -> {total} total. AC-13 defence lives in Phase 4 output lint + "
          f"Phase 5 prompt rule, not here.")

    # 8. Deliverable D2
    try:
        from src.config import write_deliverable_sources_csv
        out = write_deliverable_sources_csv()
        print(f"\n[D2] wrote {out.relative_to(path.parent.parent.parent)} ({len(SCHEMES)} sources)")
    except Exception as exc:
        warns.append(f"D2 sources csv not written: {exc}")

    print("\n" + "=" * 96)
    for w in warns:
        print(f"WARN  {w}")
    for f in failures:
        print(f"FAIL  {f}")
    print("=" * 96)
    if failures:
        print(f"PHASE 1 FAILED - {len(failures)} problem(s)")
        return 1
    print("PHASE 1 PASSED" + (f" ({len(warns)} warning(s))" if warns else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
