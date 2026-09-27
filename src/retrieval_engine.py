"""retrieval_engine.py — Phase 5: retrieve → generate → validate → render.

The single end-to-end entry point. ``answer(query)`` is the only supported way to
produce a user-visible response, and every hard rule (R1, R2, R3, R4, R5, R7, R8)
is enforced inside it rather than at a call site that a caller can forget.

Layered by request, in this order::

    guard()          R3 redaction, R4/R5 intent, R8 scope  → returns a decision
    retrieve()       embed → scheme filter → re-rank → MMR → confidence
    generate()       Groq, with a no-LLM extractive fallback
    validate()       R1 ladder, PII scan, R4/R5 lint ladder, R2 citation
    render()         the one render path for answers *and* refusals

Three things in here are deliberate deviations from implementation.md §5, each
recorded in §5.10 and each forced by a measurement rather than by taste.

§3.12 BLOCKER — the confidence floor and the ranking
---------------------------------------------------
implementation.md §5.1.4 specifies a 0.62 absolute cosine floor. Measured on the
live index that number is dead: top-1 cosine spans 0.846–0.903 and the 10th-best
spans 0.788–0.827, so a 0.62 floor fires on 0 of 15 probes and would never fire
at all. It cannot be recalibrated into usefulness either, because the distribution
has no gap to put a threshold in — every query, including ones the corpus cannot
answer, lands in the same band. ``CONFIDENCE_FLOOR`` is therefore **not** the
gate. The gate is lexical grounding: the top chunk must actually contain the
question's key terms. That separates, and it is the property that matters, because
a semantically near chunk that never mentions "expense ratio" cannot answer a
question about the expense ratio.

The same measurement showed pure cosine ranking the correct fact chunk first in
2 of 15 probes. Two causes, both addressed here:

1. *Cross-scheme crowding.* Other schemes' identical-looking numbers outranked
   the right one. Fixed by the ``scheme_id`` metadata filter that ``guard()``
   already resolved — applied before ranking, not after.
2. *Definition-versus-value collision.* Groww ships a glossary ("A fee payable to
   a mutual fund house for managing your mutual fund investments…") that scores
   **higher** on "what is the expense ratio" than the fact row that states it
   ("Expense ratio (TER): 1.03%"). No amount of chunk-size tuning fixes this;
   the two chunks are not the same kind of text. Fixed structurally, by three
   corpus-derived signals: a *fact-row* bonus (the ``FUND FACTS`` block that
   Phase 2 deliberately built one-fact-per-chunk), a *glossary* penalty (the
   ``Understand terms`` heading), and a *label-anchor* bonus (a chunk whose
   leading label **starts with** the question's key phrase outranks one that
   merely mentions it later). The label-anchor is what separates
   ``Expense ratio (TER): 1.03%`` from ``Basic expense ratio (excluding addl.
   TER): 0.84%`` — a distractor that is otherwise nearly identical and that a
   model handed both of will happily quote.

The 80C obligation (implementation.md §4.11)
--------------------------------------------
Phase 4's output lint treats any section 80C statement as a failure, and the
prompt half of that obligation lands here. The system prompt states outright that
the corpus carries no statutory tax basis, so the lint's two allowed
regenerations are not spent on every ELSS question.

Not built
---------
The Streamlit UI (``src/app.py``), the D4 sample-QA script and the AC gate runner
are separate deliverables. This module exposes ``answer()`` for all of them and
carries no UI dependency.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from src import vector_store as vs
from src.config import (
    ALLOWED_URLS,
    CONFIDENCE_FLOOR,
    MAX_SENTENCES,
    MMR_LAMBDA,
    ROOT,
    SCHEME_BY_ID,
    TOP_K,
)
from src.guardrails import (
    Intent,
    build_refusal,
    corpus_as_of,
    detect_pii,
    guard,
    lint_output,
    log_event,
    redact,
    render,
    resolve_scheme,
)
from src.textutils import is_complete_sentence, split_sentences

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

PROMPT_PATH = ROOT / "prompts" / "system_prompt.v1.txt"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

#: Superseded by :func:`assess_confidence`. Kept so the config constant and the
#: measured reality stay visibly linked; see the module docstring.
DEAD_ABSOLUTE_FLOOR = CONFIDENCE_FLOOR  # 0.62 — fires on 0/15 probes, see §5.10.1

#: Confidence is a *relative* test. These two bars are what actually separate.
#: The cosine bar is deliberately far below anything measured (0.55 vs a 0.846
#: observed minimum) — it exists only to catch a degenerate embedding, not to do
#: the deciding.
MIN_COSINE_SANE = 0.55
LEXICAL_GROUND_MIN = 0.50
MARGIN_MIN = 0.02
#: Two chunks whose bodies are this similar are the *same fact* under two labels,
#: so picking either cannot produce a wrong answer and they do not compete.
#: Measured need: Groww ships "Benchmark: NIFTY 50 Hybrid Composite Debt 50:50
#: Index" and "Benchmark index name: NIFTY 50 Hybrid Composite Debt 50:50
#: Index" as separate chunks, 0.001 apart in score. A margin gate that counts
#: them as rivals refused benchmark questions for 3 of the 5 schemes — a 60%
#: false-refusal rate on one of the brief's own six topics, from a gate that
#: looks principled.
NEAR_DUPLICATE_JACCARD = 0.70

#: Re-ranking weights. Not magic numbers without reasons — each is tied to a
#: failure it fixes, and ``scripts/rank_ablation.py`` reports the hit rate with
#: each one zeroed.
#:
#: ``W_*_LEXICAL`` are the three **unconditional** terms: they describe how
#: strongly the chunk answers the question, so they always apply.
W_LABEL_ANCHOR = 0.30   # body label *starts with* the key phrase → the fact row
W_LABEL_MENTION = 0.10   # key phrase in the label but not anchored
W_PHRASE_BODY = 0.08     # key phrase anywhere in the body
W_TERM_COVERAGE = 0.10   # fraction of key terms present
W_NUMERIC_MATCH = 0.40   # question supplies a figure and the chunk repeats it

#: ``W_*_LEXICAL`` are the three **conditional** terms, and the distinction is
#: load-bearing. They describe where a chunk *lives* in the corpus, which is a
#: prior about the kind of text it is, not evidence that it answers this
#: question. Applying them unconditionally lets a well-shaped chunk with no
#: lexical connection outrank one that contains the key term — measured: before
#: this gate, ``Portfolio turnover ratio: 32`` (a FUND FACTS row, +0.22) beat
#: ``ELSS • 3Y Lock-in`` (0 lexical evidence) on a lock-in question, and the
#: answer was wrong. So they are **multiplied by the chunk's lexical evidence**:
#: a prior may reorder candidates that are plausibly relevant, and may never
#: manufacture relevance that is not there.
W_FACT_ROW = 0.22        # lives in the FUND FACTS label:value block
W_GLOSSARY = -0.30       # lives under "Understand terms" — a definition
W_PERFORMANCE = -0.25    # contains a return figure — never the answer (R4)
W_NUMERIC_ANSWER = 0.06  # question wants a figure and the chunk has one

#: Section prefix for the one-fact-per-chunk block Phase 2 built. 125 of 377
#: chunks, and it contains **zero** return figures, which is why it is rewarded
#: rather than merely preferred.
FACT_ROW_PREFIX = "FUND FACTS"
#: Heading of Groww's glossary. 20 chunks, all pure definitions.
GLOSSARY_HEADING = "Understand terms"

#: Terms dropped from a query before its key phrase is derived. Deliberately
#: short: over-dropping turns a specific question into a generic one and loses
#: the anchor that fixes §3.12.
_STOPWORDS = frozenset("""
a an the of for in on to and or is are was were be been do does did what
whats what's how much many i me my we our you your it its this that these
those tell give show please can could would should about please fund funds
hdfc mutual
""".split())

#: Question phrasing that asks for a *figure* rather than a definition. Drives
#: the numeric boost and the answer shape the generator is asked for.
_MEASURE_WORDS = re.compile(
    r"\b(ratio|load|sip|lumpsum|lump\s*sum|nav|aum|charge|fee|fees|percentage|"
    r"percent|minimum|maximum|amount|tenure|period|age|benchmark|yield|"
    r"stamp\s*duty|turnover|tax)\b",
    re.I,
)

_NUM_TOKEN = re.compile(r"\d+(?:[.,]\d+)*")


# ═══════════════════════════════════════════════════════════════════════════
# Query analysis
# ═══════════════════════════════════════════════════════════════════════════

def _content_terms(query: str) -> list[str]:
    """Query tokens that carry meaning: stopwords and scheme words removed.

    Scheme words go because every chunk's header already names the scheme, so
    they are in the embedding on both sides and contribute nothing to ranking
    while diluting term-coverage.
    """
    q = query.lower()
    scheme_words: set[str] = set()
    for scheme in SCHEME_BY_ID.values():
        scheme_words |= set(re.findall(r"[a-z]+", scheme.name.lower()))
    tokens = [t for t in re.findall(r"[a-z0-9][a-z0-9\-']*", q)]
    out: list[str] = []
    for t in tokens:
        if t in _STOPWORDS or t in scheme_words:
            continue
        # A light singular/plural fold, not a stemmer: "loads" must match "load".
        out.append(t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t)
    return out


def key_phrase(query: str) -> str:
    """The question's subject as a phrase, e.g. "expense ratio".

    Used for the label-anchor and phrase bonuses. Returns a space-joined string
    rather than a list so it can be tested for containment directly.
    """
    return " ".join(_content_terms(query))


def wants_number(query: str) -> bool:
    """True when the question asks for a figure rather than an explanation."""
    return bool(_MEASURE_WORDS.search(query))


def question_kind(query: str) -> str:
    """``numeric`` | ``definition`` — the answer *shape* to ask the generator for.

    Not a classifier for refusal; the intent classifiers in Phase 4 own that.
    This only shapes how the draft is written, which is enough to stop the model
    answering "what is the expense ratio" with a paragraph about what expense
    ratios are.
    """
    if re.search(r"\bhow\s+(do|can|would|to)\b", query, re.I):
        return "procedure"
    return "numeric" if wants_number(query) else "definition"


# ═══════════════════════════════════════════════════════════════════════════
# Chunk view
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Retrieved:
    """One retrieved chunk with its provenance and its scores.

    ``cosine`` is the raw vector score; ``score`` is the re-ranked score that
    ordering actually uses. Keeping both is what makes the §3.12 measurements
    reproducible after the fact.

    Deliberately **not** frozen: re-ranking assigns ``score`` and ``features``
    in place after construction. Copying the record per feature would be 40
    copies per query to hold an invariant nothing reads, and the risk this
    guards against — a caller mutating a returned chunk — is covered by
    ``retrieval.by_id`` returning the same objects consistently.
    """

    chunk_id: str
    body: str
    header: str
    scheme_id: str
    scheme_name: str
    plan: str
    doc_type: str
    source_tier: str
    url: str
    as_of_date: str
    section: str
    heading_path: tuple[str, ...]
    contains_performance: bool
    cosine: float
    score: float
    features: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """The leading ``Label:`` of a fact row, else the body's first words.

        ``Expense ratio (TER): 1.03%`` → ``expense ratio (ter)``. A body with no
        colon yields its first four words, so the anchor test degrades to
        "does the body open with the phrase" instead of failing outright.
        """
        b = self.body.strip()
        head, sep, _ = b.partition(":")
        if sep and len(head) <= 60:
            return head.strip().lower()
        return " ".join(b.lower().split()[:4])

    @property
    def is_fact_row(self) -> bool:
        return self.section.startswith(FACT_ROW_PREFIX)

    @property
    def is_glossary(self) -> bool:
        return bool(self.heading_path) and self.heading_path[0] == GLOSSARY_HEADING

    @property
    def has_number(self) -> bool:
        return bool(_NUM_TOKEN.search(self.body))

    @property
    def text(self) -> str:
        return self.header + "\n" + self.body


def _term_coverage(phrase: str, body: str) -> float:
    """Fraction of the key phrase's terms that appear in the body."""
    terms = [t for t in phrase.split() if t]
    if not terms:
        return 0.0
    low = body.lower()
    return sum(1 for t in terms if t in low) / len(terms)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _term_coverage(phrase: str, body: str) -> float:
    """Fraction of the key phrase's terms that appear in the body."""
    terms = [t for t in phrase.split() if t]
    if not terms:
        return 0.0
    low = body.lower()
    return sum(1 for t in terms if t in low) / len(terms)


#: Finance abbreviations that appear contracted in one chunk and expanded in
#: another. Groww ships both forms of every benchmark as separate chunks:
#: ``Benchmark: NIFTY 100 TRI`` and ``Benchmark index name: NIFTY 100 Total
#: Return Index``. Same index; counted as rival answers, they refuse a correct
#: benchmark on 3 of the 5 schemes.
_ABBREVIATIONS: dict[str, str] = {
    "tri": "total return index",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _value_of(body: str) -> str:
    """The part of a fact row after its label, lowercased and abbreviated out.

    Only a *short* leading segment counts as a label, for the same reason
    :attr:`Retrieved.label` does: ``Benchmark: NIFTY 50 Hybrid Composite Debt
    50:50 Index`` contains a colon that is not a label separator.
    """
    head, sep, tail = body.partition(":")
    text = tail if (sep and len(head.strip()) <= 60) else body
    text = text.lower()
    for short, long in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(short)}\b", long, text)
    return " ".join(text.split())


def same_answer(a: str, b: str) -> bool:
    """True when two chunk bodies state the same fact, so either may be cited.

    Used only by the margin gate, to answer one question: could choosing the
    rival instead of the winner produce a *different* answer? If not, it is not
    a rival, and it must not be allowed to fail a question.

    Two routes, because this corpus states the same fact two ways: exact match
    on the abbreviation-expanded value (the TRI case), and token overlap for
    genuine paraphrase (the "Benchmark: X" / "Benchmark index name: X" case).
    """
    va, vb = _value_of(a), _value_of(b)
    if va and vb and va == vb:
        return True
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= NEAR_DUPLICATE_JACCARD


def near_duplicate(a: str, b: str) -> bool:
    """Alias for :func:`same_answer`, kept for the weight-table commentary."""
    return same_answer(a, b)


def score_chunk(chunk: Retrieved, phrase: str, numerics: Sequence[str],
                kind: str) -> float:
    """Re-rank one chunk. See the module docstring and the weight table for why
    each term exists, and why three of the six are gated on lexical evidence."""
    body_low = chunk.body.lower()
    label = chunk.label
    phrase_hit = phrase in body_low
    anchored = bool(phrase) and label.startswith(phrase)
    label_mentions = bool(phrase) and phrase in label and not anchored
    coverage = _term_coverage(phrase, body_low)

    # How much lexical evidence this chunk carries, 0..1. Gates the structural
    # priors. A chunk that shares no terms with the question cannot be rescued
    # by living in a well-behaved section.
    lexical = max(1.0 if phrase_hit else 0.0, coverage)

    exact = bool(numerics) and all(n in body_low for n in numerics)

    chunk.features = {
        "label_anchor": anchored,
        "label_mention": label_mentions,
        "phrase_in_body": phrase_hit,
        "term_coverage": coverage,
        "lexical": lexical,
        "fact_row": chunk.is_fact_row,
        "glossary": chunk.is_glossary,
        "performance": chunk.contains_performance,
        "numeric_exact": exact,
    }
    f = chunk.features
    return (
        chunk.cosine
        + W_LABEL_ANCHOR * f["label_anchor"]
        + W_LABEL_MENTION * f["label_mention"]
        + W_PHRASE_BODY * f["phrase_in_body"]
        + W_TERM_COVERAGE * coverage
        + W_NUMERIC_MATCH * f["numeric_exact"]
        + lexical * (
            W_FACT_ROW * f["fact_row"]
            + W_GLOSSARY * f["glossary"]
            + W_PERFORMANCE * f["performance"]
            + (W_NUMERIC_ANSWER if kind == "numeric" and chunk.has_number else 0.0)
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# Retriever
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Confidence:
    """Why retrieval is or is not trusted. Every field is inspectable.

    ``grounded`` is deliberately NOT ``cosine > CONFIDENCE_FLOOR``. That test is
    vacuous on this corpus (see the module docstring); lexical grounding is the
    test that separates.
    """

    cosine: float
    grounding: float
    margin: float
    grounded: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "cosine": round(self.cosine, 4),
            "grounding": round(self.grounding, 4),
            "margin": round(self.margin, 4),
            "grounded": self.grounded,
            "reason": self.reason,
        }


@dataclass
class RetrievalResult:
    chunks: list[Retrieved]
    confidence: Confidence
    scheme_id: str | None
    key_phrase: str
    kind: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def by_id(self) -> dict[str, Retrieved]:
        return {c.chunk_id: c for c in self.chunks}

    @property
    def top(self) -> Retrieved | None:
        return self.chunks[0] if self.chunks else None


#: chunks.jsonl, indexed once. The Phase-2 record is authoritative over the
#: Chroma metadata because it is the artefact the index was built *from*; a
#: disagreement means the index is stale, and ``_assert_index_current`` says so
#: rather than answering from whichever is convenient.
_CHUNKS: dict[str, dict] | None = None
_COLLECTION = None
_CLIENT = None

#: How the index came to be available on this process, for the trace and the
#: UI banner. A rebuild is a deployment event, not a per-request event, and it
#: should be visible rather than silently changing the latency of one answer.
_INDEX_STATE: dict = {"rebuilt": False, "reason": "", "count": 0}


def index_state() -> dict:
    """Report whether this process rebuilt the index. Safe to call any time."""
    return dict(_INDEX_STATE)


def _chunks() -> dict[str, dict]:
    global _CHUNKS
    if _CHUNKS is None:
        _CHUNKS = {c["chunk_id"]: c for c in vs.load_chunks()}
    return _CHUNKS


def _open_index():
    """Open ./chroma_db, materialising it from chunks.jsonl if it is unusable.

    ``chroma_db/`` is gitignored — it is a derived artefact, rebuilt from
    ``data/processed/chunks.jsonl``, which *is* committed. So a deploy that ships
    the source without the 8.8 MB index folder is the expected case, not an
    error, and the engine rebuilds rather than raising
    ``FileNotFoundError`` at the first question.

    The trigger is a **row count**, not a missing directory. A rebuild killed
    part-way leaves a collection that exists and is non-empty but short, and an
    existence check would wave that through — the app would then answer from a
    partial index, which fails silently and in the direction that looks like a
    retrieval-quality bug rather than a deployment one.

    The rebuild is not a *cloud fallback* and is worth not calling it one: there
    is no remote index to fail over to. Its real cold-start cost is the ~90 MB
    embedding-model download on an empty cache, which happens on first request
    either way. Set ``HDFC_RAG_INDEX_POLICY=never`` to get the old hard failure
    back (what CI should use, so a corrupt index cannot rebuild itself away).
    """
    result = vs.ensure_index(verbose=True)
    _INDEX_STATE.update(rebuilt=result["rebuilt"], reason=result["reason"],
                        count=result["count"])
    # One client handle, opened after the rebuild so it points at the folder as
    # it is now on disk. `get_client` memoises nothing, but a handle taken
    # before the upsert would be reading a directory mid-write.
    client = vs.get_client()
    return client, vs.get_collection(client)


def _collection():
    """Open the persistent index at ./chroma_db (Phase 3), asserting its shape."""
    global _CLIENT, _COLLECTION
    if _COLLECTION is None:
        _CLIENT, _COLLECTION = _open_index()
    return _COLLECTION


def _assert_index_current(count: int) -> None:
    """Refuse to answer from an index that no longer matches the chunk file.

    The failure this catches is silent: a stale index still returns plausible
    neighbours, and a plausible wrong neighbour is exactly the §3.12 failure
    arriving through a different door.
    """
    have = _chunks()
    if count != len(have):
        raise RuntimeError(
            f"index/corpus drift: {vs.CHROMA_PATH.name} holds {count} vectors but "
            f"chunks.jsonl holds {len(have)}. This is the state a rebuild killed "
            f"part-way leaves behind; delete {vs.CHROMA_PATH.name} and the next "
            f"request rebuilds it, or set HDFC_RAG_INDEX_POLICY=never and run "
            f"`python -m src.vector_store` to see the failure at build time."
        )


def assess_confidence(chunks: list[Retrieved], phrase: str) -> Confidence:
    """Decide whether the top chunk may answer the question.

    Three components, and the reasons for the structure:

    * ``cosine`` — a sanity bar only. Set far below anything measured so it
      never fires on a real query; it catches a degenerate embedding.
    * ``grounding`` — the real test. How much of the question's key phrase the
      top chunk actually contains. A chunk that never says "expense ratio"
      cannot answer a question about the expense ratio, however close its vector
      is. This is the replacement for the dead 0.62 floor.
    * ``margin`` — how far the winner is ahead of the strongest **rival answer**.
      A rival is a chunk that both grounds the question and could give a
      *different* answer. Chunks that merely repeat the winner under another
      label are not rivals: choosing between them cannot be wrong, and counting
      them as competition is what produced a 60% false-refusal rate on
      benchmark questions. When no rival survives, the margin does not apply and
      the reason string says so rather than reporting a meaningless number.

    A question with **no** key terms ("Tell me about HDFC Small Cap Fund") has
    nothing to ground against, so the lexical test is skipped rather than
    defaulted to a pass. Such a question still has to clear the cosine and
    margin bars, and the reason string says which test actually ran — a gate
    that silently did nothing is worse than one that admits it.
    """
    if not chunks:
        return Confidence(0.0, 0.0, 0.0, False, "no candidates returned")

    top = chunks[0]
    body_low = top.body.lower()
    grounding = max(
        1.0 if phrase and phrase in body_low else 0.0,
        _term_coverage(phrase, body_low),
    )
    top_lexical = top.features.get("lexical", 0.0) if top.features else 0.0

    # The **head term** of the key phrase must be present, in addition to the
    # coverage bar. In these questions the head is the subject and the tail is
    # a generic head noun: "lock-in | period", "expense | ratio", "exit | load".
    # Coverage alone is a hole — measured, Groww's exit-load glossary contains
    # the word "period", so a *lock-in* question scored 0.50 coverage on a
    # definition of exit load and would have been answered with it. Requiring
    # the head term closes that without touching the coverage bar.
    terms = [t for t in phrase.split() if t]
    head = terms[0] if terms else ""
    head_present = bool(head) and head in body_low

    rivals = [
        c for c in chunks[1:]
        if (c.features.get("lexical", 0.0) if c.features else 0.0)
        >= max(0.5, 0.9 * top_lexical)
        and not near_duplicate(top.body, c.body)
    ]
    margin = top.score - (max(c.score for c in rivals) if rivals else 0.0)

    if top.cosine < MIN_COSINE_SANE:
        return Confidence(top.cosine, grounding, margin, False,
                          f"cosine {top.cosine:.3f} below sanity bar {MIN_COSINE_SANE}")
    if not phrase:
        # No key terms to test against; fall through to the margin bar only.
        if rivals and margin < MARGIN_MIN:
            return Confidence(top.cosine, grounding, margin, False,
                              f"margin {margin:.3f} below {MARGIN_MIN} "
                              f"(no key phrase to ground against)")
        return Confidence(top.cosine, grounding, margin, True,
                          f"no key phrase in the question; {len(rivals)} rival(s)")
    if grounding < LEXICAL_GROUND_MIN:
        return Confidence(top.cosine, grounding, margin, False,
                          f"top chunk grounds only {grounding:.0%} of {phrase!r}")
    if not head_present:
        return Confidence(top.cosine, grounding, margin, False,
                          f"head term {head!r} of {phrase!r} is absent from the "
                          f"top chunk")
    if not rivals:
        return Confidence(top.cosine, grounding, margin, True,
                          f"grounded {grounding:.0%} on {phrase!r}; no rival answer")
    if margin < MARGIN_MIN:
        return Confidence(top.cosine, grounding, margin, False,
                          f"margin {margin:.3f} below {MARGIN_MIN} over "
                          f"{len(rivals)} rival answer(s) - ranking is a tie")
    return Confidence(top.cosine, grounding, margin, True,
                      f"grounded {grounding:.0%} on {phrase!r}, margin {margin:.3f} "
                      f"over {len(rivals)} rival(s)")


def _mmr(candidates: list[Retrieved], vecs: np.ndarray, k: int,
         lambda_: float = MMR_LAMBDA) -> list[int]:
    """Greedy Maximal Marginal Relevance over the re-ranked candidates.

    Trades relevance against redundancy, so the eight chunks handed to the
    generator are eight different facts rather than eight overlapping windows
    onto the same paragraph. The corpus has 48-word-piece overlaps, so without
    this the top of the context is frequently the same sentence three times.
    """
    if not candidates:
        return []
    n = len(candidates)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.where(norms == 0, 1.0, norms)
    relevance = np.array([c.score for c in candidates], dtype=np.float32)

    selected: list[int] = [int(np.argmax(relevance))]
    while len(selected) < min(k, n):
        best_i, best_v = -1, -np.inf
        for i in range(n):
            if i in selected:
                continue
            redundancy = float(np.max(unit[selected] @ unit[i]))
            v = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy
            if v > best_v:
                best_i, best_v = i, v
        if best_i < 0:
            break
        selected.append(best_i)
    return selected


def retrieve(query: str, scheme_id: str | None, k: int = TOP_K,
             pool_size: int = 40) -> RetrievalResult:
    """Query the persistent index in ./chroma_db and return the answer set.

    The ``scheme_id`` filter is applied as a Chroma ``where`` clause, before
    ranking. That is the §3.12 fix for cross-scheme crowding and it is nearly
    free: the filter is a metadata predicate, not a second search.
    """
    col = _collection()
    _assert_index_current(col.count())

    phrase = key_phrase(query)
    kind = question_kind(query)
    numerics = [n for n in _NUM_TOKEN.findall(query) if len(n) >= 2]

    vec = vs.embed([query])[0]

    where = {"scheme_id": scheme_id} if scheme_id else None
    n = min(pool_size, col.count())
    res = col.query(
        query_embeddings=[vec.tolist()],
        n_results=n,
        where=where,
        include=["metadatas", "distances", "embeddings"],
    )
    ids = res["ids"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    # Chroma nests embeddings per query — [[v1, v2, ...]] — so this is [0].
    # Indexing [0] off the outer list instead yields shape (1, n), which never
    # equals the candidate count, which silently disabled MMR and handed the
    # model the entire pool instead of k chunks. Hence the loud fallback below.
    emb = res.get("embeddings") or []
    vectors = np.asarray(emb[0], dtype=np.float32) if len(emb) else np.zeros((0, 0), np.float32)

    corpus = _chunks()
    candidates: list[Retrieved] = []
    for cid, meta, dist in zip(ids, metadatas, distances):
        rec = corpus.get(cid)
        if rec is None:
            # An id in the index with no record in chunks.jsonl. Skipping is
            # right; quoting it would invent provenance.
            continue
        candidates.append(Retrieved(
            chunk_id=cid,
            body=meta.get("body") or rec["body"],
            header=rec["header"],
            scheme_id=rec["scheme_id"],
            scheme_name=rec["scheme_name"],
            plan=rec["plan"],
            doc_type=rec["doc_type"],
            source_tier=rec["source_tier"],
            url=rec["url"],
            as_of_date=rec["as_of_date"] or "",
            section=rec["section"],
            heading_path=tuple(rec["heading_path"]),
            contains_performance=bool(rec["contains_performance"]),
            cosine=1.0 - float(dist),   # cosine space → distance is 1 - cos
            score=0.0,
        ))

    if not candidates:
        return RetrievalResult(
            [], Confidence(0.0, 0.0, 0.0, False, "no candidates for filter"),
            scheme_id, phrase, kind,
        )

    for c in candidates:
        c.score = score_chunk(c, phrase, numerics, kind)
    candidates.sort(key=lambda c: c.score, reverse=True)

    if vectors.shape[0] != len(candidates):
        # MMR unavailable. Degrade to the re-ranked order, but still cap at k:
        # handing the generator the whole pool is 4x the context budget, and a
        # silent overflow like that degrades answers without failing anything.
        chosen = list(range(min(k, len(candidates))))
        mmr_degraded = True
    else:
        # `candidates` was built by skipping unknowns, so the vector rows and the
        # candidate rows must be re-aligned before MMR indexes them by position.
        id_to_row = {cid: i for i, cid in enumerate(ids)}
        vecs = np.stack([vectors[id_to_row[c.chunk_id]] for c in candidates])
        chosen = _mmr(candidates, vecs, k)
        mmr_degraded = False

    selected = [candidates[i] for i in chosen]

    # R4 context hygiene. A return figure is never the right answer to a factual
    # question — Phase 4 already refuses performance questions outright — so
    # putting one in the context is pure downside: it is the single most likely
    # thing for the model to quote, and the reason the lint ladder exists.
    #
    # Drop them, but only when something clean survives. A context of nothing but
    # performance chunks is not a reason to fail the question; it is a reason to
    # let the model see them and refuse properly, rather than to answer from a
    # less-relevant clean chunk and cite the wrong source.
    clean = [c for c in selected if not c.contains_performance]
    if clean:
        selected = clean

    # Confidence is judged on the *re-ranked* leader, before MMR spends slots on
    # diversity — MMR must not be able to demote the best answer and have that
    # read as low confidence.
    return RetrievalResult(
        chunks=selected,
        confidence=assess_confidence(candidates[:max(k, 8)], phrase),
        scheme_id=scheme_id,
        key_phrase=phrase,
        kind=kind,
        diagnostics={
            "pool": len(candidates),
            "scheme_filter": scheme_id,
            "cosine_top": round(candidates[0].cosine, 4),
            "score_top": round(candidates[0].score, 4),
            "features_top": candidates[0].features,
            "dead_absolute_floor": DEAD_ABSOLUTE_FLOOR,
            "pool_size": pool_size,
            "k": k,
            "mmr_degraded": mmr_degraded,
            "dropped_performance": len(selected) - len(clean),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Generator — Groq LLM path + no-LLM extractive fallback
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = "You are a facts-only FAQ assistant."

#: Fact-only framing, added on a lint retry. Not a stronger threat, a narrower
#: instruction: state only what CONTEXT literally says.
FACT_ONLY_SUFFIX = (
    "\n\nSTRICT FACT-ONLY MODE (retry after a rule violation). State only what "
    "the CONTEXT literally says, word for word where it contains figures. No "
    "inference, no comparison, no restatement, no percentage or period that is "
    "not verbatim in CONTEXT, and no reference to returns, performance, tax "
    "sections, or the suitability of the fund."
)

#: The one JSON shape both generator paths return, so the validator cannot tell
#: which one produced a draft. A validator with a special case per generator is a
#: validator that quietly stops covering one of them.
DRAFT_SCHEMA = (
    '{"intent":"FACTUAL","sentences":["..."],"chunk_id":"<one id from CONTEXT>"}'
)


@dataclass
class Draft:
    """A generator proposal, before any rule has been applied to it."""

    sentences: list[str]
    chunk_id: str | None
    intent: str = "FACTUAL"
    source: str = "llm"          # "llm" | "extractive"
    note: str = ""

    @property
    def text(self) -> str:
        return " ".join(s for s in self.sentences if s.strip())


def system_prompt() -> str:
    """The versioned prompt artefact. Falls back to a stub only if it is gone.

    A missing prompt file must not silently change behaviour, so ``main()``
    asserts the file is present and ``tests/test_rag.py`` fails if it is not.
    """
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_SYSTEM_PROMPT


def _credentials() -> tuple[str | None, str | None]:
    """The API key and model id, both from ``.env`` via ``vector_store``.

    The model is never hardcoded here or anywhere else in the pipeline. It is
    read once per call so that a rotated key or a swapped model id takes effect
    without a code change — and, more usefully, so that a model the provider has
    retired surfaces as a `GROQReply(status="http_error", http=404)` naming
    `GROQ_MODEL`, rather than as answers that quietly stop being generated and
    every question falls through to extraction.
    """
    creds = vs.load_credentials()
    return (creds.api_key or None), (creds.model or None)


# ═══════════════════════════════════════════════════════════════════════════
# Conversation memory
# ═══════════════════════════════════════════════════════════════════════════

#: The rolling context window, counted in **messages**. One message is one user
#: question or one assistant answer, so 10 messages is 5 exchanges.
MEMORY_MESSAGES = 10


def remember(history: list[dict] | None, role: str, text: str, *,
             refused: bool = False) -> list[dict]:
    """Append one message to the rolling window; return the trimmed copy.

    A new list is returned rather than mutating in place. The caller is
    `st.session_state`, and assigning a fresh value is what tells Streamlit the
    state changed; an in-place `append` can leave the widget tree holding the
    previous list.

    **Redaction happens here, on the way in**, for the same reason `guard()`
    redacts: a window is both prompt material and a rendered transcript. Without
    it, a PAN typed in turn 1 is re-sent to the model on turns 2-10 and re-shown
    in the chat history — the guard only ever inspects the *current* query, so
    nothing downstream would catch it. `redact` is the project's single PII
    implementation and its `mode` matters: a 16-digit run is a card number in
    user text and a folio number in a scraped document, and the two have different
    false-positive rates. Verified against real answers: redaction is a no-op on
    every response this pipeline produces, because R3 already keeps PII out of
    them.

    **Trimming drops whole exchanges**, never single messages. A window that
    opens with an orphaned assistant reply is worse than a slightly shorter one,
    because the model reads it as a claim it made with no question attached. It
    costs at most one message of headroom and keeps user/assistant alternation
    intact, which is the property that makes the block legible to the model.
    """
    clean, _labels = redact(text or "", mode="user")
    clean = clean.strip()
    if not clean:
        return list(history or [])

    messages = list(history or [])
    messages.append({"role": role, "text": clean, "refused": bool(refused)})
    while len(messages) > MEMORY_MESSAGES:
        del messages[0:2]
    return messages


def _remembered_scheme(history: list[dict] | None) -> str | None:
    """The most recent scheme named in the window, or ``None``.

    Scans backwards for the newest **user** turn that names a scheme, and skips
    assistant turns. An assistant turn names a scheme only because it was
    answering about one, which is the same signal one turn late and would let a
    refusal's text ("I can't advise on HDFC Small Cap Fund…") donate a subject
    the user never asked about.

    This is what makes a follow-up answerable at all. "What about its exit
    load?" and "And the benchmark?" name no fund, so `resolve_scheme` returns
    ``None`` and retrieval has nothing to filter on — the corpus holds the fact
    but the question cannot reach it. Inheriting the subject is the entire
    legitimate job of memory here.

    Redaction already happened in `remember`, so this is not a path by which PII
    reaches `resolve_scheme`.
    """
    for message in reversed(history or []):
        if message.get("role") != "user":
            continue
        scheme_id, ambiguous = resolve_scheme(message.get("text", "") or "")
        if scheme_id and not ambiguous:
            return scheme_id
    return None


def _memory_block(history: list[dict] | None) -> str:
    """Render the window as a fenced, explicitly non-evidentiary block.

    The fence lives next to the history rather than in the system prompt, on
    purpose. An instruction is most reliable when it is adjacent to the content
    it governs, and it keeps `prompts/system_prompt.v1.txt` — a versioned
    artefact with a drift test against `DEFAULT_SYSTEM_PROMPT` — unchanged for a
    feature that is off by default.

    Two failure modes this wording is written against, both of which a bare
    transcript invites:

    * **History as a second corpus.** Without "not evidence / CONTEXT wins", the
      model will happily answer "the exit load is 1%" because it said so two
      turns ago, even if this turn's chunks are about a different fund. That
      silently voids R4, because the figure is unsourced by the time it renders.
    * **Refusal as a premise.** A refused turn is a statement about what the
      *corpus* lacks. Read as a fact it becomes self-reinforcing: one declined
      lock-in question makes the model decline the next one too, which is a
      correct-looking answer to a question the corpus can answer.
    """
    messages = [m for m in (history or []) if (m.get("text") or "").strip()]
    if not messages:
        return ""

    lines = [
        "EARLIER TURNS IN THIS CONVERSATION",
        "Read these ONLY to work out what a bare reference points at — \"it\",",
        "\"that fund\", \"the same scheme\", \"the other one\". They are NOT a",
        "source of facts. Every figure and every claim in your answer must come",
        "from the CONTEXT above.",
        "If an earlier turn disagrees with the CONTEXT, the CONTEXT is correct:",
        "do not repeat the earlier turn and do not mention it.",
        "A turn marked REFUSED means the published pages could not answer it.",
        "That is not a fact about the fund, it must not be treated as one, and",
        "it does not make a later question unanswerable.",
        "",
    ]
    for message in messages:
        if message.get("role") == "user":
            lines.append(f"  USER: {message['text']}")
        else:
            tag = "ASSISTANT (REFUSED)" if message.get("refused") else "ASSISTANT"
            lines.append(f"  {tag}: {message['text']}")
    return "\n".join(lines)


def build_context(chunks: Sequence[Retrieved]) -> str:
    """Render the numbered CONTEXT block the prompt refers to.

    Numbers are 1-based and positional; the model is told to copy the id
    verbatim rather than the number, because a positional reference is one more
    thing that can be misread.
    """
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] chunk_id: {c.chunk_id}\n"
            f"    scheme: {c.scheme_name} ({c.plan})\n"
            f"    section: {c.section}\n"
            f"    text: {c.body}"
        )
    return "\n".join(lines)


def _user_prompt(query: str, chunks: Sequence[Retrieved], kind: str,
                 fact_only: bool, history: list[dict] | None = None) -> str:
    hint = {
        "numeric": "The question asks for a FIGURE. Give it in the first sentence.",
        "definition": "The question asks what something MEANS. Answer in one or two "
                      "sentences, using only the CONTEXT.",
        "procedure": "The question asks HOW to do something. Give the steps the "
                     "CONTEXT states, in order.",
    }[kind]
    # The empty-history case is byte-identical to the pre-memory prompt, and that
    # is load-bearing rather than cosmetic: AC-8 is a golden-set accuracy gate
    # tuned against this exact text, so a cosmetic reflow here would move
    # measured accuracy for a feature that is off by default. The history block
    # is therefore *inserted*, not woven in.
    context = f"CONTEXT\n{build_context(chunks)}"
    memory = _memory_block(history)
    if memory:
        # CONTEXT, then the fenced history, then the question last: the current
        # question is the thing being answered and is the last thing read.
        context = f"{context}\n\n{memory}"
    return (
        f"{context}\n\n"
        f"QUESTION\n{query}\n\n"
        f"{hint}\n"
        f"Reply with one JSON object only, no prose and no code fence: {DRAFT_SCHEMA}"
    )


@dataclass(frozen=True)
class GroqReply:
    """One API call's outcome, including **why** it failed.

    The failure reason is not decoration. Before this existed a rate-limited
    call returned a bare ``None``, indistinguishable from "no key" and from "the
    model declined" — so a run that had silently fallen back to the extractive
    path for every query looked exactly like a working system. The reason string
    is what lets the pipeline say *which* degradation happened.
    """

    text: str | None
    status: str          # ok | no_key | network | rate_limited | http_error | empty
    http: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.text is not None

    def __str__(self) -> str:
        return f"{self.status}({self.http})" if self.http else self.status


#: Bounded retry for 429 only. Two attempts, honouring ``Retry-After``.
#:
#: Measured on this key: the first few calls in a burst succeed and the rest
#: return 429, so one retry after the server's own backoff is the difference
#: between a generated answer and a silent downgrade to extraction.
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_SLEEP = 2.0


def call_groq(messages: list[dict], *, temperature: float = 0.0,
              max_tokens: int = 400, timeout: int = 45) -> GroqReply:
    """One Groq chat completion. Never raises.

    The extractive fallback is only reachable if this comes back not-ok,
    and a raised exception would turn a degraded answer into no answer at
    all. A missing key, a dead network, a model that 404s (the configured
    GROQ_MODEL may not be served any more) and a JSON body that is not JSON
    all land in the same place, deliberately, but they are *reported*
    differently.

    For the gpt-oss family, reasoning_effort and temperature are both sent.
    The reasoning trace comes out of the same token budget as the answer, so
    without reasoning_effort the model can return an **empty** content and
    the answer is lost (measured: 4/4 empty). Dropping temperature instead
    costs reproducibility (measured: 2/4 distinct across four identical
    calls). AC-8 is a golden-set accuracy gate, and a pipeline that re-rolls
    the dice on every run cannot be signed off.
    """
    import requests

    key, model = _credentials()
    if not key or not model:
        return GroqReply(None, 'no_key',
                         detail='GROQ_API_KEY' if not key else 'GROQ_MODEL')

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # gpt-oss models take reasoning_effort and reject some sampling params.
    if "gpt-oss" in model:
        # Both of these, not one. See the docstring for the measurements.
        payload["reasoning_effort"] = "low"

    headers = {'Authorization': f'Bearer {key}',
               'Content-Type': 'application/json'}
    last = GroqReply(None, "rate_limited", 429)

    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload,
                                 timeout=timeout)
        except Exception as exc:
            return GroqReply(None, "network", detail=type(exc).__name__)

        if resp.status_code == 200:
            try:
                message = resp.json()["choices"][0]["message"]
            except Exception:
                return GroqReply(None, "empty", 200, "unreadable body")
            content = (message.get("content") or "").strip()
            # gpt-oss may wrap its answer in harmony channel tags.
            content = re.sub(r'<\|channel\|>[^<]*<\|message\|>', '', content)
            content = re.sub(r'<[^>]{1,40}>', '', content).strip()
            if not content:
                return GroqReply(None, "empty", 200, "empty content")
            return GroqReply(content, "ok", 200)

        if resp.status_code == 429 and attempt < RATE_LIMIT_RETRIES:
            wait = RATE_LIMIT_SLEEP * (attempt + 1)
            try:
                hinted = float(resp.headers.get("retry-after") or 0)
                wait = max(wait, min(hinted, 20.0))
            except (TypeError, ValueError):
                pass
            last = GroqReply(None, "rate_limited", 429, f"waited {wait:.0f}s")
            time.sleep(wait)
            continue

        if resp.status_code == 404:
            # The configured model is not served to this key any more. The
            # most common cause of "the app answers nothing", and the one
            # that actually happened here, so it is named explicitly rather
            # than folded into a generic HTTP error: a named error is a
            # five-second fix, a bare 404 is a debugging session.
            return GroqReply(
                None, "http_error", 404,
                f'model {model!r} not found or not accessible to this key; '
                'see .env.example for currently served models')
        try:
            body = resp.json().get('error', {}).get('message', '')[:140]
        except Exception:
            body = resp.text[:140]
        return GroqReply(None, "http_error", resp.status_code, body)

    return last


def parse_draft(raw: str | None) -> Draft | None:
    """Parse the model's JSON, tolerantly but not credulously.

    A fenced block, surrounding prose and single quotes are all accepted.

    A valid-JSON **decline** is also accepted, and this distinction is the whole
    point of the function. ``{"sentences": [], "chunk_id": "null"}`` means the
    model read the context and found the question unanswerable — which is
    correct behaviour on a corpus gap, and it must become an UNGROUNDED
    refusal. Returning ``None`` for it would route the request into the
    extractive fallback, which quotes the *nearest* chunk regardless of whether
    it answers anything. Measured: "How do I download my capital gains
    statement?" drew a correct decline, and the fallback then answered it with
    a definition of capital-gains taxation.

    ``None`` is returned only when the payload is not usable as JSON at all,
    which is the one case where falling back is the right move.
    """
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        try:
            obj = json.loads(text[start:end + 1].replace("'", '"'))
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None

    sents = obj.get("sentences")
    if isinstance(sents, str):
        sents = [sents]
    if sents is None:
        sents = []
    if not isinstance(sents, list):
        return None
    cleaned = [str(s).strip() for s in sents if str(s).strip()]

    cid = obj.get("chunk_id")
    if isinstance(cid, str) and cid.strip().lower() in ("null", "none", ""):
        cid = None
    cid = str(cid).strip() if isinstance(cid, (str, int)) and str(cid).strip() else None
    intent = str(obj.get("intent") or "FACTUAL").strip().upper()

    if not cleaned or cid is None:
        # A decline. Preserved as a Draft so the caller can refuse on it.
        return Draft([], None, intent="UNGROUNDED", source="llm",
                     note="model declined: no sentences and no chunk_id")

    return Draft(sentences=cleaned, chunk_id=cid, intent=intent, source="llm")


def extractive_draft(query: str, chunks: Sequence[Retrieved]) -> Draft:
    """Answer with no LLM at all: quote the best chunk, verbatim.

    Zero hallucination surface — every word came out of the corpus — and it
    satisfies R1, R2, R4, R5 and R7 by construction. Less fluent, and it does
    not reword a fragment into a sentence, but it cannot be wrong about a
    figure. This is the path that runs when the key is absent, the model 404s,
    or the network is down; PRD Q2 is still open, so it is a shipping path and
    not a test fixture.
    """
    if not chunks:
        return Draft([], None, intent="UNGROUNDED", source="extractive",
                     note="no chunks")

    top = chunks[0]
    terms = set(_content_terms(query))
    sentences = split_sentences(top.body)

    def overlap(s: str) -> int:
        toks = set(re.findall(r"[a-z0-9]+", s.lower()))
        return len(toks & terms)

    ranked = sorted(enumerate(sentences), key=lambda p: (-overlap(p[1]), p[0]))
    keep = [s for _, s in ranked[:MAX_SENTENCES] if s.strip()]
    if not keep:
        keep = [top.body.strip()[:240]]
    keep.sort()  # restore reading order; the ranking was only for selection

    # A truncated fragment must still terminate, or R1's own splitter counts the
    # next fragment as part of this sentence.
    if keep and not is_complete_sentence(keep[-1]):
        keep[-1] = keep[-1].rstrip(" ,;:-") + "."

    return Draft(sentences=keep, chunk_id=top.chunk_id, source="extractive",
                 note=f"no LLM; quoted {top.chunk_id}")


def generate(query: str, chunks: Sequence[Retrieved], *,
             kind: str = "definition", fact_only: bool = False,
             history: list[dict] | None = None) -> Draft:
    """LLM path, falling back to extraction. Both return a :class:`Draft`.

    ``kind`` and ``chunks`` must come from the same retrieval result, or the
    hint and the context disagree and the model resolves that disagreement by
    guessing.

    Two things are deliberately *not* treated as failures, because treating them
    as failures produces a worse answer:

    * **A decline.** The model read the context and found nothing. That becomes
      an UNGROUNDED refusal. Substituting the nearest chunk instead would
      answer a question the model just established the corpus cannot answer.
    * **Malformed output.** The model replied but not as JSON. Extractive, and
      not a second live call, so a formatting wobble cannot become a retry loop
      that spends the rate limit.
    """
    if not chunks:
        return Draft([], None, intent="UNGROUNDED", source="extractive",
                     note="no context")

    system = system_prompt() + (FACT_ONLY_SUFFIX if fact_only else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _user_prompt(query, chunks, kind, fact_only,
                                                 history)},
    ]
    reply = call_groq(messages)
    draft = parse_draft(reply.text)
    if draft is not None:
        draft.note = draft.note or f"groq={reply.status}"
        return draft

    fallback = extractive_draft(query, chunks)
    if not reply.ok:
        # Record *why* the LLM path was abandoned. Without this, a run that
        # silently answered every question from extraction looks identical to a
        # healthy one, which is the failure mode the GroqReply reason exists to
        # prevent.
        fallback.note = f"groq={reply.status} {reply.detail}".strip()
    else:
        fallback.note = f"groq ok but unparseable: {reply.text[:80]!r}"
    return fallback


# ═══════════════════════════════════════════════════════════════════════════
# Validator — R1, R2, R3-out, R4, R5
# ═══════════════════════════════════════════════════════════════════════════

#: R1's ladder, rung 4. A templated sentence from the top chunk, used when
#: truncation would leave a dangling clause — which AC-1 calls a release blocker.
def _rung4(query: str, chunks: Sequence[Retrieved]) -> tuple[list[str], str | None]:
    if not chunks:
        return [], None
    top = chunks[0]
    body = " ".join(split_sentences(top.body)[:MAX_SENTENCES]).strip()
    if not body:
        body = top.body.strip()[:200]
    if body and not is_complete_sentence(body):
        body = body.rstrip(" ,;:-") + "."
    return [body], top.chunk_id


def enforce_r1(text: str, query: str, chunks: Sequence[Retrieved],
               compress: bool = True) -> tuple[list[str], str | None, str]:
    """implementation.md §5.3, exactly.

    Returns ``(sentences, forced_chunk_id, rung)``. ``forced_chunk_id`` is set
    only by rung 4, where the text is no longer the model's and the citation has
    to follow it.

    Rung 3 cuts at the end of a whole sentence — no ellipsis, no partial clause.
    A dangling clause reads as a rendering bug, which is what AC-1 is about.
    """
    sents = split_sentences(text)
    if len(sents) <= MAX_SENTENCES:
        return sents, None, "rung1_ok"

    if compress:
        sents = _compress_via_llm(sents, query)
    if len(sents) > MAX_SENTENCES:
        sents = sents[:MAX_SENTENCES]
    if not all(is_complete_sentence(s) for s in sents):
        body, cid = _rung4(query, chunks)
        return body, cid, "rung4_template"
    return sents, None, "rung3_truncated"


def _compress_via_llm(sents: list[str], query: str) -> list[str]:
    """Rung 2. One live call, and a hard no-op on any failure.

    Compression is asked for rather than mechanical, because the mechanical
    alternative is deleting whole sentences, which is rung 3 wearing a hat. If
    the call fails, rung 3 is perfectly acceptable — it truncates at a
    sentence boundary and never breaches R1.
    """
    if not _credentials()[0]:
        return sents
    raw = call_groq(
        [
            {"role": "system", "content":
                "Rewrite the sentences as at most 3 complete sentences, keeping "
                "every figure exactly as written. Output plain sentences separated "
                "by a single space. No preamble."},
            {"role": "user", "content": " ".join(sents)},
        ],
        max_tokens=200,
    ).text
    if not raw:
        return sents
    out = [s for s in split_sentences(raw) if s.strip()]
    return out or sents


#: R2. The LLM emits a ``chunk_id``; the URL is resolved here. **The model never
#: produces a URL**, which is what makes a fabricated citation structurally
#: impossible rather than merely unlikely.
def resolve_citation(chunk_id: str | None, result: RetrievalResult) -> dict | None:
    """``chunk_id`` → citation, or None. Membership is checked against the
    retrieved set, so an id the model invented resolves to nothing."""
    if not chunk_id:
        return None
    chunk = result.by_id.get(chunk_id)
    if chunk is None:
        return None
    if chunk.url not in ALLOWED_URLS:
        return None
    return {"label": f"{chunk.scheme_name} — {chunk.doc_type}", "url": chunk.url}


#: Which refusal a lint violation escalates to. An answer that stated a return
#: figure has failed R4, so the honest response is the R4 refusal, not a generic
#: one: the user learns why there is no number.
_LINT_TO_REFUSAL = {
    "r4_return_figure": "PERFORMANCE_REQUEST",
    "r5_directive": "ADVICE_REQUEST",
    "statutory_80c_unsourced": "UNGROUNDED",
}

#: PRD §4.5: two consecutive lint failures escalate to a refusal.
MAX_LINT_ATTEMPTS = 2


@dataclass
class Validated:
    """A draft that has passed every rule, or a refusal that replaced it."""

    sentences: list[str]
    citation: dict
    as_of: str
    is_refusal: bool
    chunk_id: str | None = None
    r1_rung: str = "rung1_ok"
    lint_attempts: int = 0
    generator: str = "llm"
    escalated: str = ""
    #: Why the generator behaved as it did — "ok", "rate_limited", "declined",
    #: "unparseable". Carried to the trace so a degraded run is visible rather
    #: than merely slower.
    llm_status: str = ""


def validate(query: str, result: RetrievalResult,
             history: list[dict] | None = None) -> Validated:
    """Apply every output-side rule to a draft, regenerating as needed.

    Order matters and is not arbitrary:

    1. **PII on output** (PRD §6.1 f) before anything else. A draft that echoed a
       phone number must never reach a regenerate loop that might log it.
    2. **R1** before the lint, so a lint decision is made on the text that will
       actually be shown rather than on a longer pre-truncation draft.
    3. **R4/R5 lint**, twice, then the matching refusal.
    4. **R2**, repairing an uncited answer by re-asking for one id from the
       enumerated list. An uncited factual answer is never rendered (AC-2).
    """
    chunks = result.chunks
    kind = result.kind
    draft = generate(query, chunks, kind=kind, history=history)

    # 0 — the model declined. This is a *refusal*, not a failure: it read the
    # context and found the question unanswerable, which on a corpus gap is the
    # correct and most useful thing to do. Reaching the extractive fallback here
    # would quote the nearest chunk regardless of whether it answers anything.
    if draft.intent == "UNGROUNDED" and not draft.sentences:
        log_event("block.ungrounded", query=query, stage="model_declined",
                  note=draft.note)
        return _as_refusal(build_refusal("UNGROUNDED"),
                           escalated=f"model_declined: {draft.note}")

    # 1 — PII in the model's own output.
    hits = detect_pii(draft.text, mode="user")
    if hits:
        log_event("block.pii_output", query=query,
                  pii_labels=[h.label for h in hits])
        return _as_refusal(build_refusal("PII_REQUEST"), escalated="pii_in_output")

    # 2 + 3 — R1, then the lint ladder.
    sentences, forced_cid, rung = enforce_r1(draft.text, query, chunks)
    chunk_id = forced_cid or draft.chunk_id
    attempts = 0
    lint_violations: list[str] = []

    for attempt in range(MAX_LINT_ATTEMPTS + 1):
        candidate = " ".join(sentences)
        if not candidate.strip():
            return _as_refusal(build_refusal("UNGROUNDED"), escalated="empty_draft")

        pii = detect_pii(candidate, mode="user")
        if pii:
            log_event("block.pii_output", query=query,
                      pii_labels=[h.label for h in pii])
            return _as_refusal(build_refusal("PII_REQUEST"),
                               escalated="pii_after_r1")

        lint = lint_output(candidate)
        if lint.ok:
            break
        lint_violations = lint.violations
        if attempt == MAX_LINT_ATTEMPTS:
            key = next((_LINT_TO_REFUSAL[v] for v in lint_violations
                        if v in _LINT_TO_REFUSAL), "UNGROUNDED")
            log_event("block.lint", query=query, violations=lint_violations,
                      evidence=lint.evidence)
            return _as_refusal(build_refusal(key),
                               escalated="lint:" + ",".join(lint_violations))
        # Regenerate in fact-only framing, then re-lint. Never strip the clause:
        # deleting it leaves broken grammar and a confident sentence asserting
        # something adjacent to the violation.
        attempts += 1
        log_event("retry.lint", query=query, attempt=attempts,
                  violations=lint_violations, evidence=lint.evidence)
        # History carries into the retry. It is the same question about the same
        # fund, so the referents are still in scope; dropping it here would make
        # the retry answer a different question than the one that was asked.
        draft = generate(query, chunks, kind=kind, fact_only=True, history=history)
        sentences, forced_cid, rung = enforce_r1(draft.text, query, chunks)
        chunk_id = forced_cid or draft.chunk_id

    # 4 — R2, with a repair pass.
    citation = resolve_citation(chunk_id, result)
    if citation is None and attempts < MAX_LINT_ATTEMPTS:
        repaired = _repair_citation(query, chunks, kind, result.by_id)
        if repaired is not None:
            sentences, forced_cid, rung = repaired[0], None, rung
            chunk_id = repaired[1]
            citation = resolve_citation(chunk_id, result)

    if citation is None:
        log_event("block.uncited", query=query, chunk_id=chunk_id)
        return _as_refusal(build_refusal("UNGROUNDED"), escalated="no_valid_citation")

    # R7 — the date comes from the *cited* chunk, so the transparency line
    # describes the source of the sentence shown. Never blank (AC-12).
    cited = result.by_id.get(chunk_id)
    as_of = (cited.as_of_date if cited else "") or corpus_as_of()

    return Validated(
        sentences=sentences,
        citation=citation,
        as_of=as_of,
        is_refusal=False,
        chunk_id=chunk_id,
        r1_rung=rung,
        lint_attempts=attempts,
        generator=draft.source,
        llm_status=draft.note or "ok",
    )


def _repair_citation(query: str, chunks: Sequence[Retrieved], kind: str,
                     by_id: dict[str, Retrieved]) -> tuple[list[str], str] | None:
    """R2 rung 2: re-ask for one id from the enumerated list.

    Only the ids that were actually retrieved are offered, so the repair cannot
    be talked into citing a plausible id from outside the context.
    """
    allowed = [c.chunk_id for c in chunks]
    if not allowed:
        return None
    raw = call_groq(
        [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": (
                f"CONTEXT chunk_ids: {', '.join(allowed)}\n"
                f"QUESTION\n{query}\n\n"
                f"Copy the one chunk_id from the list above that most directly "
                f"answers the question, and write the answer in at most "
                f"{MAX_SENTENCES} sentences using only that chunk. Reply with one "
                f"JSON object only: {DRAFT_SCHEMA}"
            )},
        ],
        max_tokens=200,
    ).text
    draft = parse_draft(raw)
    if draft is None or draft.chunk_id not in by_id:
        return None
    return draft.sentences, draft.chunk_id


def _as_refusal(response: dict, escalated: str = "") -> Validated:
    """Adapt a rendered refusal dict to the same shape an answer takes."""
    return Validated(
        sentences=[response["text"]],
        citation=response["citation"],
        as_of=response["as_of_date"],
        is_refusal=True,
        escalated=escalated,
        r1_rung="refusal",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def answer_with_trace(query: str,
                      history: list[dict] | None = None) -> tuple[dict, dict]:
    """``answer`` plus a diagnostic trace. The trace is never rendered.

    Kept separate rather than as a key on the response because a trace field on
    the response dict is a field some caller will eventually put in the chat
    bubble.

    ``history`` is the rolling conversation window and defaults to empty, which
    makes this function's behaviour with no argument byte-identical to its
    behaviour before memory existed. Every caller that is a single-shot Q&A tool
    — `test_rag.py`, the phase gates, the eval harness — keeps the verified
    Phase 5 numbers rather than silently moving onto a different prompt.
    """
    started = time.time()
    decision = guard(query)

    trace: dict = {
        "query": query,
        "intent": decision.intent.value,
        "blocked": decision.blocked,
        "reason": decision.reason,
    }
    if decision.blocked:
        # `intent` alone is not enough to identify the rule that fired: the PII
        # block reuses Intent.ADVICE, so a trace showing "advice" for a question
        # that was actually blocked on a PAN sends you hunting a classifier bug
        # that does not exist. The template is the honest label.
        trace["template"] = decision.template_key
        if decision.pii_labels:
            trace["pii_labels"] = decision.pii_labels
        # guard() already rendered this through the one render path, with its
        # educational link and the corpus date. Nothing to add.
        trace["stage"] = "guard"
        trace["ms"] = int((time.time() - started) * 1000)
        return decision.response, trace

    q = decision.redacted_query or query
    scheme_id = decision.scheme_id

    # Scheme carry-over. A follow-up that names no fund — "What about its exit
    # load?", "And the benchmark?" — cannot reach the answer even though the
    # corpus holds it, and the reason is the embedding rather than the lexical
    # gate. Measured: "What about its exit load?" scores cosine 0.220 against the
    # Flexi exit-load chunk versus 0.829 for the same question naming the fund,
    # so `MIN_COSINE_SANE` refuses it. Filtering to the right scheme does *not*
    # rescue it (still 0.220) — a bare pronoun simply does not embed near a
    # fund-specific chunk, while the lexical gate was already passing at 100%.
    #
    # So the referent is resolved into the query rather than the confidence gate
    # being relaxed for inherited queries. Relaxing it would be the more
    # convenient change and the more dangerous one: the sanity bar exists to stop
    # a bad pairing being rendered as a fact, and a bypass keyed on "this query
    # came from memory" would disable it exactly where there is least evidence
    # about what is being asked. Naming the subject makes it a well-formed
    # question again, so the bar is applied normally and can still refuse.
    #
    # Three guards, each load-bearing:
    #   * `decision.blocked` is False here, so the *current* turn is a real
    #     question. A blocked turn returns before this point, which is what stops
    #     a PII question from being answered off its own redacted text.
    #   * An ambiguous query (two funds named) is already blocked by `guard()`,
    #     so there is no need to re-derive the ambiguity flag here — and a
    #     question that names two funds never silently picks one of them.
    #   * Inheritance is strictly a fallback: a query naming its own fund
    #     resolves normally and this never runs.
    inherited = None
    if history and scheme_id is None:
        inherited = _remembered_scheme(history)
        if inherited:
            scheme_id = inherited
            # The parenthetical names a subject; it asserts no fact. Facts still
            # come only from CONTEXT, and the model is told to reply in JSON, so
            # this is not echoed back to the user.
            q = f"{q} (about {SCHEME_BY_ID[scheme_id].name})"

    result = retrieve(q, scheme_id)
    trace["retrieval"] = result.diagnostics
    trace["confidence"] = result.confidence.as_dict()
    trace["key_phrase"] = result.key_phrase
    trace["kind"] = result.kind
    trace["top_chunk"] = result.top.chunk_id if result.top else None
    if inherited:
        trace["scheme_inherited"] = inherited
        trace["resolved_query"] = q
    trace["memory_messages"] = len(history or [])

    if not result.chunks or not result.confidence.grounded:
        log_event("block.ungrounded", query=query,
                  reason=result.confidence.reason)
        trace["stage"] = "ungrounded"
        trace["ms"] = int((time.time() - started) * 1000)
        return build_refusal("UNGROUNDED"), trace

    validated = validate(q, result, history)
    trace["r1_rung"] = validated.r1_rung
    trace["lint_attempts"] = validated.lint_attempts
    trace["generator"] = validated.generator
    trace["llm_status"] = validated.llm_status
    trace["escalated"] = validated.escalated
    trace["chunk_id"] = validated.chunk_id
    trace["stage"] = "answered"
    trace["ms"] = int((time.time() - started) * 1000)

    if validated.is_refusal:
        log_event("refuse.validator", query=query, escalated=validated.escalated)
        response = render(
            validated.sentences,
            link_label=validated.citation["label"],
            link_url=validated.citation["url"],
            as_of=validated.as_of,
            is_refusal=True,
        )
    else:
        log_event("answer", query=query, chunk_id=validated.chunk_id,
                  r1_rung=validated.r1_rung, generator=validated.generator,
                  lint_attempts=validated.lint_attempts)
        response = render(
            validated.sentences,
            link_label=validated.citation["label"],
            link_url=validated.citation["url"],
            as_of=validated.as_of,
            is_refusal=False,
        )

    return response, trace


def answer(query: str, history: list[dict] | None = None) -> dict:
    """The end-to-end entry point. Returns the rendered response dict.

    ``history`` is an optional rolling conversation window (see
    :func:`remember`): a list of ``{"role", "text", "refused"}`` messages, at
    most :data:`MEMORY_MESSAGES` of them. It is used for exactly two things —
    resolving which fund a bare follow-up refers to, and telling the model which
    references to resolve — and never as a source of facts.

    Contract, enforced inside and not by this function's callers:

    * at most :data:`MAX_SENTENCES` sentences  (R1)
    * exactly one citation, url in ``ALLOWED_URLS`` or the educational page  (R2)
    * no PII in, no PII out  (R3)
    * no return figure, no directive, no unsourced statutory claim  (R4, R5)
    * one sentence of refusal text, never a dead end  (R6)
    * a non-blank transparency line on **every** response  (R7)
    * only the 5 in-scope schemes, Direct Growth only  (R8)
    """
    return answer_with_trace(query, history)[0]


# ═══════════════════════════════════════════════════════════════════════════
# Self-check
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    """Offline checks. No network: the LLM path is exercised by test_rag.py."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.textutils import split_sentences

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    print("PROMPT ARTEFACT")
    check("prompts/system_prompt.v1.txt present", PROMPT_PATH.exists())
    sp = system_prompt()
    check("prompt states the 3-sentence cap", "MAXIMUM 3 SENTENCES" in sp)
    check("prompt covers the 80C obligation", "80C" in sp)

    print("\nQUERY ANALYSIS")
    for q, want in [
        ("What is the expense ratio of HDFC Large Cap Fund?", "expense ratio"),
        ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", "lock-in period"),
        ("What is the exit load of HDFC Small Cap Fund?", "exit load"),
        ("How do I download my capital gains statement?", "download capital gain statement"),
    ]:
        got = key_phrase(q)
        check(f"key_phrase({q[:34]!r})", got == want, f"-> {got!r}")
    check("a contentless question yields an empty phrase",
          key_phrase("Tell me about HDFC Small Cap Fund") == "")
    check("that case skips grounding, not the gate",
          assess_confidence(
              retrieve("Tell me about HDFC Small Cap Fund", "HDFC_SMALL_CAP").chunks,
              "").reason.startswith("no key phrase")
          or not assess_confidence(
              retrieve("Tell me about HDFC Small Cap Fund", "HDFC_SMALL_CAP").chunks,
              "").grounded,
          "defined, not defaulted")

    print("\nR1 SPLITTER (implementation.md 5.2 table)")
    cases = [
        ("The expense ratio is 0.55%. Min SIP is Rs. 500.", 2),
        ("Fees, e.g. exit load, apply. See note.", 2),
        ("- Exit load 1%\n- No load after 12m", 2),
        ("A. B. C.", 3),
    ]
    for text, want in cases:
        got = len(split_sentences(text))
        check(f"{text[:38]!r} -> {want}", got == want, f"got {got}")

    print("\nINDEX")
    col = _collection()
    check("collection reachable", col.count() > 0, f"{col.count()} vectors")
    _assert_index_current(col.count())
    check("index matches chunks.jsonl", True, f"{len(_chunks())} chunks")

    print("\nRETRIEVAL — the 3.12 ranking")
    probes = [
        ("What is the expense ratio of HDFC Large Cap Fund?", "HDFC_LARGE_CAP__scheme_page__0005"),
        ("What is the expense ratio of HDFC ELSS Tax Saver Fund?", "HDFC_ELSS__scheme_page__0005"),
        ("What is the expense ratio of HDFC Balanced Advantage Fund?", "HDFC_BAL_ADV__scheme_page__0005"),
        ("What is the minimum SIP for HDFC ELSS Tax Saver Fund?", "HDFC_ELSS__scheme_page__0009"),
        ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", "HDFC_ELSS__scheme_page__0026"),
        ("What is the benchmark of HDFC Small Cap Fund?", "HDFC_SMALL_CAP__scheme_page__0013"),
        ("What is the exit load of HDFC Flexi Cap Fund?", "HDFC_FLEXI_CAP__scheme_page__0007"),
        ("What is the riskometer level of HDFC Large Cap Fund?", "HDFC_LARGE_CAP__scheme_page__0012"),
    ]
    hits = 0
    for q, want in probes:
        sid, _ = resolve_scheme(q)
        r = retrieve(q, sid)
        got = r.top.chunk_id if r.top else None
        hits += got == want
        print(f"    {q[:46]:46s} -> {str(got)[-6:]:6s} "
              f"[{'ok' if got == want else 'MISS want ' + want[-6:]}]"
              f"  cos={r.confidence.cosine:.3f} gr={r.confidence.grounding:.2f}"
              f" mg={r.confidence.margin:+.3f}")
    check("top-1 correct on every probe", hits == len(probes), f"{hits}/{len(probes)}")

    # The failure §3.12 was about, stated as an assertion rather than a hope:
    # the glossary definition must never outrank the fact row that states the
    # figure, and no performance chunk may reach the top.
    r = retrieve("What is the expense ratio of HDFC Large Cap Fund?", "HDFC_LARGE_CAP")
    check("top-1 is a FUND FACTS row, not the glossary", r.top.is_fact_row)
    check("top-1 states the figure", "1.03" in r.top.body, r.top.body)
    check("no performance chunk in the context (R4)",
          not any(c.contains_performance for c in r.chunks),
          f"dropped {r.diagnostics['dropped_performance']}")
    check("no performance chunk is the answer",
          not r.top.contains_performance)
    check("every context chunk is in-scheme",
          all(c.scheme_id == "HDFC_LARGE_CAP" for c in r.chunks))
    check("context is non-empty and within k", 0 < len(r.chunks) <= TOP_K,
          f"{len(r.chunks)} chunks")
    check("MMR ran (not degraded to plain order)", not r.diagnostics["mmr_degraded"])
    check("context is not the whole pool", len(r.chunks) < r.diagnostics["pool"],
          f"{len(r.chunks)} of {r.diagnostics['pool']}")
    check("MMR diversified the context (no duplicate chunk_ids)",
          len({c.chunk_id for c in r.chunks}) == len(r.chunks))

    print("\nSAME-ANSWER TEST (margin gate)")
    # Groww ships these as separate chunks, 0.001 apart in score. Counting them
    # as rival answers refused benchmark questions on 3 of the 5 schemes.
    check("TRI == Total Return Index",
          same_answer("Benchmark: NIFTY 100 TRI",
                      "Benchmark index name: NIFTY 100 Total Return Index"))
    check("BSE SmallCap TRI == expanded",
          same_answer("Benchmark: BSE 250 SmallCap TRI",
                      "Benchmark index name: BSE 250 SmallCap Total Return Index"))
    check("a colon inside a value is not a label separator",
          same_answer("Benchmark: NIFTY 50 Hybrid Composite Debt 50:50 Index",
                      "Benchmark index name: NIFTY 50 Hybrid Composite Debt 50:50 Index"))
    check("TER vs basic expense ratio are NOT the same answer",
          not same_answer("Expense ratio (TER): 1.03%",
                          "Basic expense ratio (excluding addl. TER): 0.84%"))
    check("two schemes' figures are NOT the same answer",
          not same_answer("Expense ratio (TER): 1.03%",
                          "Expense ratio (TER): 1.21%"))
    check("a different benchmark is NOT the same answer",
          not same_answer("Benchmark: NIFTY 100 TRI", "Benchmark: NIFTY 500 TRI"))

    print("\nCONFIDENCE")
    check("absolute floor documented as dead", DEAD_ABSOLUTE_FLOOR == 0.62)
    r = retrieve("What is the expense ratio of HDFC Large Cap Fund?", "HDFC_LARGE_CAP")
    check("lexical grounding on a real question", r.confidence.grounded,
          r.confidence.reason)
    r2 = retrieve("What is the colour of the fund manager's car?", "HDFC_LARGE_CAP")
    check("ungroundable question refused", not r2.confidence.grounded,
          r2.confidence.reason)

    # The head-term requirement. Coverage alone let a *lock-in* question reach
    # 0.50 coverage on Groww's exit-load glossary, which contains the word
    # "period" but says nothing about a lock-in.
    r3 = retrieve("What is the lock-in period of HDFC Large Cap Fund?", "HDFC_LARGE_CAP")
    check("half-covered key phrase is not grounding",
          not r3.confidence.grounded, r3.confidence.reason)
    r4 = retrieve("What is the lock-in period of HDFC ELSS Tax Saver Fund?",
                  "HDFC_ELSS")
    check("the same question passes when the corpus has the answer",
          r4.confidence.grounded, r4.confidence.reason)

    # The 6 brief topics x 5 schemes matrix. 26/30 grounded, and all 4 misses
    # are the same correct refusal: only ELSS has a lock-in in the corpus, so
    # the other four must decline rather than answer from a near neighbour.
    print("\nBRIEF TOPIC MATRIX (5 schemes x 6 topics)")
    grounded = total = 0
    misses: list[str] = []
    for name in ("HDFC Large Cap Fund", "HDFC Flexi Cap Fund",
                 "HDFC ELSS Tax Saver Fund", "HDFC Small Cap Fund",
                 "HDFC Balanced Advantage Fund"):
        sid, _ = resolve_scheme(name)
        for topic in ("expense ratio", "exit load", "minimum SIP",
                      "lock-in period", "riskometer level", "benchmark"):
            res = retrieve(f"What is the {topic} of {name}?", sid)
            total += 1
            if res.confidence.grounded:
                grounded += 1
            else:
                misses.append(f"{name[5:17]}/{topic}")
    check("every answerable brief question is grounded", grounded >= 26,
          f"{grounded}/{total}")
    check("the only refusals are lock-in on non-ELSS schemes",
          bool(misses) and all(m.endswith("/lock-in period") for m in misses),
          ", ".join(misses) or "none")

    print("\nCONTEXT BLOCK")
    ctx = build_context(r.chunks)
    check("one numbered block per chunk",
          all(f"[{i}]" in ctx for i in range(1, len(r.chunks) + 1)))
    check("every chunk_id is enumerated verbatim",
          all(c.chunk_id in ctx for c in r.chunks))
    check("plan is stated (scope at the prompt level too)",
          "Direct Growth" in ctx)
    check("no chunk_id appears that was not retrieved",
          ctx.count("chunk_id:") == len(r.chunks))

    print("\nDRAFT PARSING")
    ok = parse_draft('{"intent":"FACTUAL","sentences":["A fact."],'
                     '"chunk_id":"HDFC_LARGE_CAP__scheme_page__0005"}')
    check("valid draft parses", ok is not None and ok.chunk_id is not None)
    check("fenced JSON parses", parse_draft(
        '```json\n{"sentences":["A fact."],'
        '"chunk_id":"X"}\n```') is not None)
    # The decline case. Returning None here routed a correct refusal into the
    # extractive fallback, which answered a capital-gains *download* question
    # with a definition of capital-gains taxation.
    declined = parse_draft('{"intent":"FACTUAL","sentences":[],"chunk_id":"null"}')
    check("a decline is preserved, not discarded", declined is not None)
    check("a decline is marked UNGROUNDED",
          declined is not None and declined.intent == "UNGROUNDED")
    check("a decline carries no sentences",
          declined is not None and declined.sentences == [])
    check("a decline does NOT become an extractive answer",
          declined is not None and declined.source == "llm")
    check("the string 'null' is read as absent, not as an id",
          parse_draft('{"sentences":[],"chunk_id":"null"}').chunk_id is None)
    check("unparseable text returns None", parse_draft("not json at all") is None)
    check("no-braces text returns None", parse_draft("") is None)

    print("\nGROQ REPLY REPORTING")
    rep = GroqReply(None, "rate_limited", 429)
    check("a failure reports a reason, not a bare None",
          not rep.ok and str(rep) == "rate_limited(429)", str(rep))
    check("ok reports as ok", GroqReply("x", "ok", 200).ok)
    check("rate limiting is retried, not given up on",
          RATE_LIMIT_RETRIES >= 1, f"{RATE_LIMIT_RETRIES} retries")

    print("\nEXTRACTIVE FALLBACK (no LLM)")
    d = extractive_draft("What is the expense ratio of HDFC Large Cap Fund?", r.chunks)
    check("quotes the corpus verbatim", "1.03" in d.text, d.text[:60])
    check("cites a retrieved chunk", d.chunk_id in {c.chunk_id for c in r.chunks})
    check("within R1", len(split_sentences(d.text)) <= MAX_SENTENCES)
    check("output lint clean", lint_output(d.text).ok)

    print("\nR2 RESOLVER")
    check("known id resolves", resolve_citation(r.top.chunk_id, r) is not None)
    check("invented id resolves to None",
          resolve_citation("HDFC_LARGE_CAP__scheme_page__9999", r) is None)
    check("null id resolves to None", resolve_citation(None, r) is None)
    check("url is in the allow-list",
          resolve_citation(r.top.chunk_id, r)["url"] in ALLOWED_URLS)

    print("\nR1 LADDER")
    four = " ".join(["Fact one here.", "Fact two here.", "Fact three here.", "Fact four here."])
    sents, cid, rung = enforce_r1(four, "q", r.chunks, compress=False)
    check("4 sentences -> 3", len(sents) == MAX_SENTENCES, rung)
    check("no dangling clause", all(is_complete_sentence(s) for s in sents))
    s2, _, rung2 = enforce_r1("One fact.", "q", r.chunks)
    check("1 sentence passes", len(s2) == 1, rung2)

    print("\nRENDER")
    resp = build_refusal("ADVICE_REQUEST")
    check("refusal carries the transparency line",
          resp["transparency_line"].startswith("Last updated from sources: "))
    check("refusal date non-blank", bool(resp["as_of_date"]), resp["as_of_date"])
    check("refusal has exactly one link", bool(resp["citation"]["url"]))
    check("refusal is <= 3 sentences", resp["sentences"] <= MAX_SENTENCES)

    print("\nENV")
    creds = vs.load_credentials()
    check("GROQ_API_KEY readable via python-dotenv", creds.api_key != "",
          f"from {Path(creds.key_source).name if creds.key_source != 'MISSING' else creds.key_source}")
    check("GROQ_MODEL set", creds.model != "", creds.model)
    if creds.model and creds.model.startswith("llama-"):
        print("    WARNING: llama-* models are no longer served by Groq (HTTP 404).")

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
