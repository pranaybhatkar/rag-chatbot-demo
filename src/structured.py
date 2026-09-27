"""structured.py — pull the authoritative fund record out of the page's payload.

Why this exists
---------------
Groww server-renders the page *shell* and the embedded ``__NEXT_DATA__`` JSON
payload, but the key/value figures (expense ratio, riskometer, minimum SIP) are
rendered client-side from that payload. A plain tag-strip therefore yields the
label "Expense ratio" with no number next to it — the single most important fact
in the corpus, silently absent.

So Stage 1 reads both:
  * the rendered text  -> prose, definitions, tables, holdings
  * the JSON payload   -> the authoritative scalars, verbatim

Nothing here is invented. Every value is copied from the page we fetched, and
the section is labelled as payload-derived so a reviewer can tell at a glance
which facts came from which layer of the same document.

A note on R4: the payload also contains a daily NAV/expense-ratio history array
and the page contains a returns table. Those are deliberately NOT imported here.
Returns are reference material that must never be stated (PRD §4.4); keeping the
number-bearing noise out of the imported facts is cheaper than arguing with a
model that can see it. The visible-DOM returns table is still captured, and is
the reason the Phase 4 output lint and Phase 5 prompt rules are mandatory.
"""
from __future__ import annotations

import json
import re
from typing import Any

#: JSON keys whose values are copied into the facts block, in output order.
#: Label wording is deliberately close to the brief's own vocabulary so the
#: chunker sees "expense ratio" and "riskometer" verbatim.
FACT_FIELDS: tuple[tuple[str, str, str], ...] = (
    # (json key,              output label,                        format)
    ("fund_name",             "Fund name",                          "text"),
    ("scheme_name",           "Scheme name (as published)",         "text"),
    ("plan_type",             "Plan type",                          "text"),
    ("scheme_type",           "Option type",                        "text"),
    ("sub_category",          "Category",                           "text"),
    ("expense_ratio",         "Expense ratio (TER)",                "pct"),
    ("base_expense_ratio",    "Basic expense ratio (excluding addl. TER)", "pct"),
    ("exit_load",             "Exit load",                          "text"),
    ("stamp_duty",            "Stamp duty",                         "text"),
    ("min_sip_investment",    "Minimum SIP investment",             "inr"),
    ("min_investment_amount", "Minimum lumpsum investment",         "inr"),
    ("min_withdrawal",        "Minimum withdrawal amount",          "inr"),
    ("nfo_risk",              "Riskometer level",                   "text"),
    ("benchmark",             "Benchmark",                          "text"),
    ("benchmark_name",        "Benchmark index name",               "text"),
    ("aum",                   "Assets under management",            "crore"),
    ("nav",                   "Latest NAV",                         "inr"),
    ("nav_date",              "NAV date",                           "text"),
    ("fund_manager",          "Fund manager",                       "text"),
    ("launch_date",           "Launch date",                        "text"),
    ("portfolio_turnover",    "Portfolio turnover ratio",           "raw"),
    ("isin",                  "ISIN",                               "text"),
    ("direct_scheme_code",    "Direct growth scheme code",          "raw"),
    ("registrar_agent",       "Registrar and transfer agent (RTA)", "text"),
    ("description",           "Scheme objective",                   "text"),
)

FACTS_HEADING = "FUND FACTS (from the page's embedded data payload, verbatim)"


def _find_payload(data: Any) -> dict | None:
    """Locate the fund record.

    Prefers the known path, then falls back to a deep search for any dict that
    carries both ``expense_ratio`` and ``scheme_name`` — the payload key names
    are not part of any public contract and have changed before.
    """
    node = data
    for key in ("props", "pageProps", "mfServerSideData"):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            node = None
            break
    if isinstance(node, dict) and "expense_ratio" in node:
        return node

    best: dict | None = None
    stack = [data]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "expense_ratio" in cur and "scheme_name" in cur:
                if best is None or len(cur) > len(best):
                    best = cur
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return best


def _fmt(value: Any, kind: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ("none", "null", "-", "n/a"):
            return None
    if kind == "pct":
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return str(value)
    if kind == "inr":
        try:
            return f"INR {float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)
    if kind == "crore":
        try:
            return f"INR {float(value):,.2f} crore"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def extract_facts(html: str) -> tuple[str, dict[str, Any]]:
    """-> (facts block text, raw field dict). Returns ("", {}) when absent."""
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return "", {}
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return "", {}

    fund = _find_payload(payload)
    if not fund:
        return "", {}

    lines: list[str] = []
    values: dict[str, Any] = {}
    for key, label, kind in FACT_FIELDS:
        formatted = _fmt(fund.get(key), kind)
        if formatted is None:
            continue
        values[key] = fund.get(key)
        text = " ".join(formatted.split())
        if kind == "text" and len(text) > 900:
            text = text[:900].rsplit(" ", 1)[0] + "..."
        lines.append(f"{label}: {text}")

    if not lines:
        return "", {}
    return "\n".join([FACTS_HEADING, *lines]), values
