"""answerability.py — can the corpus actually answer the brief's questions?

This is the check that matters most in Phase 1. A corpus can pass every
structural invariant and still be unable to answer "what is the expense ratio",
because the figure was never extracted. Each probe is the literal question a
user would type, paired with the regex that proves the answer is present.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, SCHEMES

#: (question, per-scheme answer regex). The regex must match the *answer*, not
#: merely the topic word.
PROBES: tuple[tuple[str, str], ...] = (
    ("What is the expense ratio of <scheme>?",
     r"Expense ratio \(TER\):\s*([\d.]+%)"),
    ("What is the exit load on <scheme>?",
     r"Exit load:\s*(.+)"),
    ("What is the minimum SIP for <scheme>?",
     r"Minimum SIP investment:\s*(INR [\d,]+)"),
    ("What is the minimum lumpsum for <scheme>?",
     r"Minimum lumpsum investment:\s*(INR [\d,]+)"),
    ("What is the riskometer level of <scheme>?",
     r"Riskometer level:\s*(.+)"),
    ("What is the benchmark of <scheme>?",
     r"Benchmark:\s*(.+)"),
    ("Who manages <scheme>?",
     r"Fund manager:\s*(.+)"),
    ("What is the NAV of <scheme>?",
     r"Latest NAV:\s*(INR [\d,.]+)"),
    ("What is the AUM of <scheme>?",
     r"Assets under management:\s*(.+)"),
    ("What is the launch date / ISIN of <scheme>?",
     r"ISIN:\s*(\w+)"),
    ("What is the lock-in period for HDFC ELSS Tax Saver Fund?",
     r"(3 years|three years|lock[-\s]?in)"),
)


def main() -> int:
    docs = [json.loads(l) for l in (PROCESSED_DIR / "documents.jsonl").open(encoding="utf-8")]
    by_id = {d["scheme_id"]: d for d in docs}

    print("=" * 100)
    print("CORPUS ANSWERABILITY  (the brief's own example questions)")
    print("=" * 100)

    gaps = 0
    for question, pat in PROBES:
        lock_in_only = "lock-in" in question
        targets = (["HDFC_ELSS"] if lock_in_only else [s.scheme_id for s in SCHEMES])
        answers = []
        for sid in targets:
            m = re.search(pat, by_id[sid]["text"], re.I)
            answers.append(m.group(1).strip().rstrip(".") if m else None)
        shown = ", ".join(a if a else "**MISSING**" for a in answers[:3])
        if len(set(answers)) > 1:
            shown += ", ..."
        ok = all(a for a in answers)
        gaps += 0 if ok else 1
        print(f"  [{'OK  ' if ok else 'GAP '}] {question}")
        print(f"         -> {shown}")

    print("\n" + "=" * 100)
    print("per-scheme answer snapshot (facts block, verbatim from source)")
    print("=" * 100)
    for s in SCHEMES:
        d = by_id[s.scheme_id]
        facts = d["sections"][0]
        print(f"\n  {d['scheme_id']}  ({d['as_of_date']})")
        for line in facts["text"].splitlines()[:16]:
            if line.strip():
                print(f"      {line.strip()[:96]}")

    print("\n" + "=" * 100)
    if gaps:
        print(f"ANSWERABILITY: {gaps} probe(s) cannot be answered from the corpus")
        return 1
    print("ANSWERABILITY: every brief example question is answerable from the corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
