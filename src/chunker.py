"""chunker.py — Stage 2: Chunking.

Turns Stage 1's section-segmented documents into retrieval-sized chunks and
writes a complete, human-readable preview of every chunk.

Why this is harder than calling a splitter
------------------------------------------
``all-MiniLM-L6-v2`` **silently truncates past 256 word-pieces**. No exception,
no warning. The tail never reaches the vector while the stored text still shows
it in full, so the index looks complete and is blind to the tail. Two traps make
that worse than it first appears, and both are handled here:

1. ``all-MiniLM-L6-v2``'s ``tokenizer.json`` ships with **truncation baked in at
   128** and padding fixed at 128. Counting ``len(encode(text).ids)`` therefore
   returns exactly 128 for *any* over-long text. Measured on real corpus text, a
   562-word-piece chunk reports as 128 — **434 word-pieces invisible**, and the
   chunk sails through a naive ``<= 256`` assertion. :func:`load_tokenizer`
   therefore calls ``no_truncation()`` and ``no_padding()`` first.
2. ``len(text.split())`` overstates capacity by roughly 30%, because ~1.3
   word-pieces ≈ 1 English word. Every budget here is measured in word-pieces
   using the model's own vocabulary.

Two structural rules from architecture.md §6, both of which prevent confidently
wrong retrieval:

* **Never span a section boundary.** The brief's topics map onto sections, so
  section-aligned chunks are both more retrievable and more precisely citable.
* **A table is atomic.** Exit-load slabs and expense-ratio grids are the
  highest-value facts in the corpus and the easiest to destroy. A split table
  yields a chunk that looks like a fact and means nothing.

Outputs:
  data/processed/chunks.jsonl   structured chunks for Stage 3
  data/chunks_preview.txt       every chunk, in full, for visual verification
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BRIEF_TOPIC_PROBES,
    EMBEDDING_MODEL,
    KNOWN_TOPIC_GAPS,
    PROCESSED_DIR,
    RAW_DIR,
    SCHEME_BY_ID,
    WORDPIECE_HARD_CAP,
)
from src.structured import FACTS_HEADING

# ── Chunking parameters (architecture.md §6) ───────────────────────────────
#: Operating point. Below the 256 cap so the context header and overlap have
#: room without pushing anything over the truncation limit.
TARGET_WP = 224
#: Not a tunable — the model's limit. Asserted as a hard error.
HARD_CAP_WP = WORDPIECE_HARD_CAP            # 256
#: Tokens of the previous chunk repeated at the head of the next, so a fact
#: split across a boundary is still retrievable from either side.
OVERLAP_WP = 48
#: A chunk below this is not worth its own vector; it merges forward.
MIN_WP = 24
#: Chars-per-word-piece used only to *plan* a split, before the exact tokenizer
#: count confirms it. Calibrated on the real corpus: min 1.68, p10 2.33,
#: median 3.74, max 5.88. The p10 is used so the first guess is conservative —
#: every emitted chunk is then measured exactly, so a bad guess costs a second
#: split, never a silent overflow.
CHARS_PER_WP = 2.33

#: Separator hierarchy for the recursive split, coarsest first.
#:
#: No empty-string entry, deliberately. An earlier version carried ``""`` to mean
#: "character level", but the ``depth >= len(SEPARATORS)`` guard below already
#: routes there, so the entry was reachable *and* fatal: ``text.split("")``
#: raises ``ValueError: empty separator``. It never fired on the real corpus,
#: because every section had whitespace, which is exactly the kind of edge that
#: waits for production input to arrive.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

#: A chunk body shorter than this is a *fragment*, not a fact: a stray rating
#: digit, a fund-manager initial ("AA", "DM"), a date split off its label
#: ("2020)"). Fragments are merged forward, or dropped when there is nothing to
#: merge into — never embedded alone, where they are near-duplicate neighbours of
#: real chunks and only dilute retrieval.
#:
#: 12, not 24, and the reason matters: at 24 this filter deleted
#: ``ELSS • 3Y Lock-in`` — 15 characters, and the corpus's *only* statement of the
#: ELSS lock-in period, which is one of the brief's three example questions. A
#: terse fact and a broken fragment are both short; only word count separates
#: them, and length alone cannot.
MIN_BODY_CHARS = 12
#: A lone unit with fewer words than this is a fragment with nothing to attach to
#: (a section that holds only "DM" or "5"). At 3, ``ELSS • 3Y Lock-in`` survives
#: while ``AA`` does not.
MIN_STANDALONE_WORDS = 3

#: A section headed by a *different* fund's full scheme name is related-funds
#: sidebar chrome, not content about the in-scope scheme. Embedding it would put
#: an out-of-scope fund in the index and invite an R8 scope violation.
_OTHER_FUND_HEADING_RE = re.compile(
    r"^HDFC\s+.+?\s+Fund\s*(?:Direct)?\s*(?:Plan\s*)?(?:Growth|Plan Growth)$", re.I)

#: A return figure inside a chunk body (PRD §4.4). Flagged, never removed — the
#: figures are legitimate reference material, but a chunk carrying one is the
#: first thing the AC-13 output lint must catch, and it is a down-rank candidate.
PERFORMANCE_RE = re.compile(
    r"\|\s*[+-]?\s*\d+\.?\d*\s*%"
    r"|[+-]?\s*\d+\.?\d*\s*%\s*(?:return|cagr|annualis|annualiz)\b"
    r"|\b(?:cagr|annualised|annualized)\b",
    re.I,
)

#: Broker-platform marketing sections. Groww's own promo blocks ("Trade in
#: Futures & Options", "Invest in Stocks", "Start SIP") are not facts about the
#: schemes, and "Start SIP" in particular competes with a real minimum-SIP
#: question at retrieval time. Matched on body, conservatively.
_BROKER_PROMO_RE = re.compile(
    r"\b(trade in f&o|futures & options|invest in (?:stocks|etfs|ipos)|"
    r"fast orders?|watchlists?\b|real-time p&l|\bterminal\b|"
    r"place (?:a )?orders?|discount brokers?\b|open (?:a )?free account)\b",
    re.I,
)

#: ``Label: value`` — the shape of a payload fact line. Used to split the facts
#: block one fact per chunk, and to keep fact lines from being mistaken for
#: headings anywhere in the pipeline.
_KEYVALUE_RE = re.compile(r"^[A-Z][A-Za-z0-9 ()/&'’.-]{2,70}:\s*\S")

_TOKENIZER = None


def load_tokenizer():
    """Load the model's tokenizer with truncation and padding **disabled**.

    The shipped ``tokenizer.json`` truncates at 128 and pads to 128. Leaving that
    in place makes every over-long text report as exactly 128 word-pieces.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer
        _TOKENIZER = Tokenizer.from_pretrained(EMBEDDING_MODEL)
        _TOKENIZER.no_truncation()
        _TOKENIZER.no_padding()
    return _TOKENIZER


def count_wp(text: str) -> int:
    """Exact word-piece count, including ``[CLS]``/``[SEP]``."""
    if not text:
        return 0
    return len(load_tokenizer().encode(text, add_special_tokens=True).ids)


def char_budget(wp_budget: int) -> int:
    return max(24, int(wp_budget * CHARS_PER_WP))


# ── The recursive splitter ─────────────────────────────────────────────────

def _split_by_chars(text: str, max_wp: int) -> list[str]:
    """Last-resort split. Binary-searches the longest prefix that fits.

    Backs off to a word boundary so a chunk never starts mid-word — a chunk
    beginning "ratio: 1.03%" instead of "Expense ratio: 1.03%" loses the token
    that makes it findable by name.
    """
    out: list[str] = []
    rest = text
    while rest:
        if count_wp(rest) <= max_wp:
            out.append(rest)
            break
        lo, hi, best = 1, len(rest), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_wp(rest[:mid]) <= max_wp:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        cut = rest.rfind(" ", 0, best)
        if cut > best * 0.6:                 # only back off if it stays close
            best = cut + 1
        out.append(rest[:best])
        rest = rest[best:]
    return out


def recursive_split(text: str, max_wp: int, depth: int = 0) -> list[str]:
    """Recursive character splitter, measured in word-pieces at every level.

    Splits on the coarsest separator that helps, then greedily re-joins the
    pieces up to the budget, then recurses into anything still oversized. The
    loop is bounded by ``len(SEPARATORS)``; the final level splits by character
    so it always terminates.
    """
    text = text.strip()
    if not text:
        return []
    if count_wp(text) <= max_wp:
        return [text]
    if depth >= len(SEPARATORS):
        return _split_by_chars(text, max_wp)

    sep = SEPARATORS[depth]
    parts = text.split(sep)
    if len(parts) == 1:
        return recursive_split(text, max_wp, depth + 1)

    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else buf + sep + part
        if buf and count_wp(candidate) > max_wp:
            out.append(buf)
            buf = part
        else:
            buf = candidate
    if buf:
        out.append(buf)

    final: list[str] = []
    for piece in out:
        if count_wp(piece) > max_wp:
            final.extend(recursive_split(piece, max_wp, depth + 1))
        else:
            final.append(piece.strip())
    return [p for p in final if p.strip()]


# ── Tables ────────────────────────────────────────────────────────────────

def is_table(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    pipes = sum(1 for l in lines if l.strip().startswith("|"))
    return pipes >= max(2, int(0.6 * len(lines)))


def split_table(text: str, max_wp: int) -> list[str]:
    """Split a table by rows, repeating the first row on every piece.

    The table never splits *mid-row*: a half-row is meaningless. When the whole
    table fits, it travels as one chunk.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    if count_wp(text) <= max_wp:
        return [text]

    header, rows = lines[0], lines[1:]
    head_wp = count_wp(header)

    out: list[str] = []
    buf: list[str] = []
    used = head_wp
    for row in rows:
        row_wp = count_wp(row)
        if used + row_wp > max_wp and buf:
            out.append("\n".join([header, *buf]))
            buf, used = [], head_wp
        # A single row wider than the budget is split by character, header kept.
        if row_wp > max_wp - head_wp:
            for piece in _split_by_chars(row, max_wp - head_wp):
                out.append("\n".join([header, piece]))
            continue
        buf.append(row)
        used += row_wp
    if buf:
        out.append("\n".join([header, *buf]))
    return out


# ── Chunk record ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    scheme_id: str
    scheme_name: str
    plan: str
    doc_type: str
    source_tier: str
    title: str
    url: str
    as_of_date: str
    section: str
    heading_path: list[str]
    ordinal: int
    text: str                    # header + body — this is what gets embedded
    header: str
    body: str
    n_wordpieces: int
    n_chars: int
    is_table: bool
    has_overlap: bool
    contains_performance: bool
    content_hash: str
    chunk_params: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def make_header(scheme_name: str, plan: str, doc_type: str, as_of: str, section: str) -> str:
    """Synthetic context header, prepended to every chunk.

    Makes each chunk self-describing for the embedder and for the citation. It
    costs ~20 word-pieces, which is why TARGET_WP is 224 rather than 256. The
    generator strips this before composing an answer — it is retrieval
    scaffolding, not answer text.
    """
    section = re.sub(r"\s+", " ", section).strip()
    if len(section) > 60:
        section = section[:57].rsplit(" ", 1)[0] + "..."
    return f"{scheme_name} ({plan}) | {doc_type} | as of {as_of} | {section}"


# ── Overlap ───────────────────────────────────────────────────────────────

def tail_overlap(prev_body: str, overlap_wp: int) -> str:
    """The last ~``overlap_wp`` word-pieces of the previous body."""
    if not prev_body or overlap_wp <= 0:
        return ""
    words = prev_body.split()
    if not words:
        return ""
    tail = ""
    for word in reversed(words):
        candidate = f"{word} {tail}".strip()
        if count_wp(candidate) > overlap_wp:
            break
        tail = candidate
    return tail


# ── Per-document chunking ─────────────────────────────────────────────────

def is_facts_block(heading: str, text: str) -> bool:
    return heading.strip().startswith(FACTS_HEADING[:20]) or text.startswith(FACTS_HEADING)


def split_facts_block(text: str) -> list[str]:
    """One chunk per ``Label: value`` fact.

    The facts block is the highest-value, most-compact content in the corpus:
    expense ratio, exit load, minimum SIP, riskometer and benchmark in one place.
    Left to the budget splitter it becomes two ~220-word-piece chunks, so a
    question about the riskometer has to compete with five unrelated figures in
    the same vector. One fact per chunk makes each of them a near-pure match.
    """
    facts: list[str] = []
    carried = ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _KEYVALUE_RE.match(s):
            facts.append(s)
        elif facts:
            facts[-1] += " " + s          # wrapped continuation of the last fact
        else:
            carried += (" " if carried else "") + s
    return [f for f in ([carried] if carried else []) + facts if f.strip()]


def _is_fragment(unit: str) -> bool:
    """True when a unit cannot stand alone as a chunk.

    Length alone killed ``ELSS • 3Y Lock-in``; word count alone would keep ``AA``
    and ``5``. A fragment is short *and* too thin to be a claim — unless it is
    ``Label: value``, which is a complete claim at any length. That exemption
    belongs here rather than at the call site so the rule is self-consistent
    wherever it is used.
    """
    text = unit.strip()
    if _KEYVALUE_RE.match(text):
        return False
    return len(text) < MIN_BODY_CHARS or len(text.split()) < MIN_STANDALONE_WORDS


def is_chrome(heading: str, own_scheme_name: str, body: str = "") -> bool:
    """True when a section is page chrome rather than in-scope content."""
    h = heading.strip()
    if _OTHER_FUND_HEADING_RE.match(h) and own_scheme_name.lower() not in h.lower():
        return True                        # another fund's full scheme name
    return bool(_BROKER_PROMO_RE.search(body or ""))


def chunk_document(doc: dict) -> list[Chunk]:
    scheme = SCHEME_BY_ID[doc["scheme_id"]]
    chunks: list[Chunk] = []
    ordinal = 0
    dropped_sections = 0

    for section in doc["sections"]:
        heading = section["heading"]
        if is_chrome(heading, doc["scheme_name"], section["text"]):
            dropped_sections += 1
            continue

        header = make_header(doc["scheme_name"], scheme.plan, doc["doc_type"],
                             doc["as_of_date"] or "undated", heading)
        header_wp = count_wp(header)
        facts_block = is_facts_block(heading, section["text"])

        if facts_block:
            units = split_facts_block(section["text"])
        elif is_table(section["text"]):
            units = split_table(section["text"], char_budget(TARGET_WP - header_wp))
        else:
            units = recursive_split(section["text"], char_budget(TARGET_WP - header_wp))

        units = [u for u in units if u.strip()]
        if not units:
            continue

        # Merge orphans forward: a chunk under MIN_WP is not worth its own vector.
        # NOT applied to the facts block. Each ``Label: value`` line is already a
        # complete, self-contained fact, and many are short by nature — merging
        # them produced one oversized blob that ``recursive_split`` then tore
        # apart on ", ", splitting "Stamp duty: 0.005% (from July 1st," from
        # "2020)". Merging must only repair fragments left by a *split*.
        merged: list[str] = []
        if facts_block:
            merged = units
        else:
            for unit in units:
                if merged and (count_wp(unit) < MIN_WP
                               or len(unit.strip()) < MIN_BODY_CHARS):
                    merged[-1] = f"{merged[-1]} {unit}".strip()
                else:
                    merged.append(unit)

        prev_body = ""
        for unit in merged:
            if not facts_block and _is_fragment(unit):
                continue                     # orphan with nowhere to merge to
            overlap = tail_overlap(prev_body, OVERLAP_WP)
            budget = TARGET_WP - header_wp - (count_wp(overlap) if overlap else 0)

            # A fact block is one fact per unit by construction; only a single
            # oversized fact (the scheme objective) needs re-splitting.
            if count_wp(unit) > budget:
                for sub in recursive_split(unit, budget):
                    _emit(chunks, doc, scheme, heading, header, sub, overlap,
                          is_table(section["text"]), ordinal)
                    ordinal += 1
                    prev_body = sub
            else:
                _emit(chunks, doc, scheme, heading, header, unit, overlap,
                      is_table(section["text"]) and not facts_block, ordinal)
                ordinal += 1
                prev_body = unit

    if dropped_sections:
        print(f"  dropped {dropped_sections} out-of-scope sidebar section(s) from "
              f"{doc['doc_id']}", file=sys.stderr)
    return chunks


def _emit(chunks: list[Chunk], doc: dict, scheme, heading: str, header: str,
          body: str, overlap: str, table: bool, ordinal: int) -> None:
    body = body.strip()
    if not body:
        return
    text = f"{header}\n{overlap + ' ' if overlap else ''}{body}".strip()

    n = count_wp(text)
    if n > HARD_CAP_WP:
        # Unreachable in practice: every budget above is measured with the same
        # tokenizer. Kept because a silent overflow here corrupts the index
        # invisibly, and an exception is the only honest failure.
        raise ValueError(
            f"{doc['doc_id']} section {heading!r}: chunk is {n} word-pieces, "
            f"over the {HARD_CAP_WP} hard cap (would be silently truncated)"
        )

    chunks.append(Chunk(
        chunk_id=f"{doc['scheme_id']}__{doc['doc_type']}__{ordinal:04d}",
        doc_id=doc["doc_id"],
        scheme_id=doc["scheme_id"],
        scheme_name=doc["scheme_name"],
        plan=scheme.plan,
        doc_type=doc["doc_type"],
        source_tier=doc["source_tier"],
        title=doc["title"],
        url=doc["source_url"],
        as_of_date=doc["as_of_date"] or "",
        section=heading,
        heading_path=section_path(doc, heading),
        ordinal=ordinal,
        text=text,
        header=header,
        body=body,
        n_wordpieces=n,
        n_chars=len(text),
        is_table=table,
        has_overlap=bool(overlap),
        contains_performance=bool(PERFORMANCE_RE.search(body)),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        chunk_params={
            "target_wp": TARGET_WP, "hard_cap_wp": HARD_CAP_WP,
            "overlap_wp": OVERLAP_WP, "min_wp": MIN_WP,
            "strategy": "section_aware+recursive_char",
            "tokenizer": EMBEDDING_MODEL,
        },
    ))


def section_path(doc: dict, heading: str) -> list[str]:
    for s in doc["sections"]:
        if s["heading"] == heading:
            return s.get("heading_path", [heading])
    return [heading]


# ── Loading ───────────────────────────────────────────────────────────────

def load_documents(input_mode: str = "processed") -> list[dict]:
    """Load Stage 1 output, or parse ``data/raw/*.txt`` directly.

    ``processed`` is the default because Stage 1 already did the work that
    matters here: section segmentation and provenance. ``raw`` re-parses the
    raw records for anyone who wants to chunk without the intermediate file.
    """
    if input_mode == "raw":
        from src.ingest import load_raw_file
        return [json.loads(load_raw_file(p).to_json())
                for p in sorted(RAW_DIR.glob("*.txt"))]
    path = PROCESSED_DIR / "documents.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found - run scripts/build_index.py --stages 1 first")
    return [json.loads(l) for l in path.open(encoding="utf-8")]


# ── Preview ───────────────────────────────────────────────────────────────

RULE = "=" * 96
THIN = "-" * 96


def write_preview(chunks: list[Chunk], docs: list[dict], path: Path) -> Path:
    """Every chunk, in full, with the metadata needed to judge it by eye.

    Deliberately not truncated: a preview that hides the tail of a chunk cannot
    show you a split fact or a stranded heading, which are the two things this
    file exists to catch.
    """
    wps = sorted(c.n_wordpieces for c in chunks)
    lines: list[str] = [
        RULE,
        "HDFC MUTUAL FUND FAQ RAG  -  CHUNK PREVIEW",
        f"generated : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"documents : {len(docs)}   chunks: {len(chunks)}",
        f"params    : target_wp={TARGET_WP} overlap_wp={OVERLAP_WP} "
        f"min_wp={MIN_WP} hard_cap_wp={HARD_CAP_WP}",
        f"strategy  : section_aware + recursive_char (separators: "
        f"{' > '.join(repr(s) for s in SEPARATORS[:-1])})",
        f"tokenizer : {EMBEDDING_MODEL}  (truncation+padding disabled, "
        f"{load_tokenizer().get_vocab_size()} vocab)",
        RULE,
        "",
        "SUMMARY",
        THIN,
        f"  chunks                    : {len(chunks)}",
        f"  word-pieces min/med/max   : {wps[0]} / {wps[len(wps) // 2]} / {wps[-1]}",
        f"  over hard cap (256)       : {sum(1 for n in wps if n > HARD_CAP_WP)}   (must be 0)",
        f"  under min (24)            : {sum(1 for n in wps if n < MIN_WP)}   (should be 0)",
        f"  table chunks              : {sum(1 for c in chunks if c.is_table)}",
        f"  chunks with overlap       : {sum(1 for c in chunks if c.has_overlap)}",
        f"  chunks w/ return figure   : {sum(1 for c in chunks if c.contains_performance)}"
        f"   (flagged for Phase 5 down-rank + AC-13 lint)",
        f"  distinct sections chunked : {len({(c.scheme_id, c.section) for c in chunks})}",
        "",
        "  by scheme:",
    ]
    for sid in sorted({c.scheme_id for c in chunks}):
        n = sum(1 for c in chunks if c.scheme_id == sid)
        print_n = f"{n:>5} chunks" if n else "      0"
        lines.append(f"    {sid:<20} {print_n}")
    lines += [
        "",
        THIN,
        f"ALL CHUNKS  ({len(chunks)} total, full text)",
        THIN,
    ]

    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    for doc in docs:
        group = by_doc.get(doc["doc_id"], [])
        lines += [
            "",
            RULE,
            f"DOCUMENT {doc['doc_id']}   {len(group)} chunks",
            f"  url        : {doc['source_url']}",
            f"  as_of_date : {doc['as_of_date']}  (R7 transparency line source)",
            f"  tier       : {doc['source_tier']}",
            RULE,
        ]
        for c in group:
            lines += [
                "",
                THIN,
                f"[{c.chunk_id}]  wp={c.n_wordpieces}  chars={c.n_chars}  "
                f"table={'yes' if c.is_table else 'no '}  "
                f"overlap={'yes' if c.has_overlap else 'no '}"
                + ("  PERF" if c.contains_performance else ""),
                f"  section : {c.section}",
                f"  path    : {' > '.join(c.heading_path)}",
                f"  url     : {c.url}",
                f"  as_of   : {c.as_of_date}",
                f"  hash    : {c.content_hash[:16]}",
                THIN,
                c.text,
            ]

    lines += _preview_checks(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _preview_checks(chunks: list[Chunk]) -> list[str]:
    """Hard checks plus topic presence, so a whole missing topic class is visible."""
    corpus = "\n".join(c.text for c in chunks)
    over = [c for c in chunks if c.n_wordpieces > HARD_CAP_WP]
    under = [c for c in chunks if c.n_wordpieces < MIN_WP]
    ids = [c.chunk_id for c in chunks]
    no_url = [c for c in chunks if not c.url]
    no_date = [c for c in chunks if not c.as_of_date]
    no_header = [c for c in chunks if not c.text.startswith(c.header)]
    schemes = {c.scheme_id for c in chunks}

    lines = ["", RULE, "HARD CHECKS", RULE]
    checks = [
        (not over, f"every chunk <= {HARD_CAP_WP} word-pieces"
                   f"{'' if not over else '  -> ' + ', '.join(c.chunk_id for c in over[:5])}"),
        (len(set(ids)) == len(ids), f"every chunk_id unique  ({len(ids)} chunks)"),
        (not no_url, f"every chunk carries a url"
                     f"{'' if not no_url else f'  -> {len(no_url)} missing'}"),
        (not no_date, f"every chunk carries as_of_date"
                      f"{'' if not no_date else f'  -> {len(no_date)} missing'}"),
        (not no_header, "every chunk starts with its context header"),
        (not under, f"no chunk under {MIN_WP} word-pieces"
                    f"{'' if not under else f'  -> {len(under)} undersized'}"),
        (len(schemes) == 5, f"all 5 schemes represented  ({len(schemes)}/5)"),
    ]
    for ok, label in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    lines += ["", "TOPIC PRESENCE  (every brief topic must appear somewhere)", THIN]
    for topic, pat in BRIEF_TOPIC_PROBES.items():
        hits = sum(1 for c in chunks if re.search(pat, c.text, re.I))
        if hits:
            lines.append(f"  [PASS] {topic:<20} {hits:>4} chunks mention it")
        elif topic in KNOWN_TOPIC_GAPS:
            lines.append(f"  [GAP ] {topic:<20} {hits:>4} chunks  "
                         f"<- KNOWN CORPUS GAP: {KNOWN_TOPIC_GAPS[topic]}")
        else:
            lines.append(f"  [FAIL] {topic:<20} {hits:>4} chunks  "
                         f"<- topic absent from the corpus")
    lines.append("")
    return lines


# ── Runner ────────────────────────────────────────────────────────────────

def run(input_mode: str = "processed") -> tuple[list[Chunk], list[dict]]:
    docs = load_documents(input_mode)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))

    # Determinism check: identical doc_ids on a re-run is what makes Stage 3's
    # upsert idempotent.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "chunks.jsonl").write_text(
        "".join(c.to_json() + "\n" for c in chunks), encoding="utf-8"
    )
    return chunks, docs


def main() -> int:
    mode = "raw" if "--from-raw" in sys.argv else "processed"
    chunks, docs = run(mode)

    preview_path = Path("data/chunks_preview.txt")
    write_preview(chunks, docs, preview_path)

    wps = sorted(c.n_wordpieces for c in chunks)
    print(RULE)
    print("STAGE 2: CHUNKING")
    print(RULE)
    print(f"  documents            : {len(docs)}   (from {mode})")
    print(f"  chunks               : {len(chunks)}")
    print(f"  word-pieces min/med/max : {wps[0]} / {wps[len(wps) // 2]} / {wps[-1]}"
          f"   (target {TARGET_WP}, cap {HARD_CAP_WP})")
    print(f"  over cap             : {sum(1 for n in wps if n > HARD_CAP_WP)}  (must be 0)")
    print(f"  under min            : {sum(1 for n in wps if n < MIN_WP)}  (should be 0)")
    print(f"  table chunks         : {sum(1 for c in chunks if c.is_table)}")
    print(f"  with overlap         : {sum(1 for c in chunks if c.has_overlap)}")
    print()
    print(f"  chunks  -> {PROCESSED_DIR / 'chunks.jsonl'}")
    print(f"  preview -> {preview_path}   <-- read this")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
