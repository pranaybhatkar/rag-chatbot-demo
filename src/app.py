"""src/app.py — the Streamlit front end.

    .\\.venv\\Scripts\\streamlit.exe run src\\app.py

What this file is responsible for, and what it deliberately is not
----------------------------------------------------------------
It renders. Every rule — the 3-sentence cap, the single allow-listed link, the
PII and performance and advice refusals, the transparency line — is enforced in
`src/retrieval_engine.py` and `src/guardrails.py`, and was verified there before
any of it reached a screen. This file does **not** re-implement those rules,
because a second implementation of a safety rule is a second place for it to be
wrong, and a truncation applied here would be invisible: the user would see a
short answer and conclude the engine had produced a short answer.

So where the UI does check, it checks *loudly*:
  * a response over 3 sentences is shown in full, plus a visible warning. The
    engine's R1 ladder is supposed to make that impossible; if it ever happens,
    the reviewer sees the evidence instead of a silently shortened sentence.
  * a response whose citation URL is not on the allow-list is rendered without
    the link and flagged, rather than linked.

Conversation memory
-------------------
This file owns the *state*; `src/retrieval_engine.py` owns the *rules*. The
window is held in `st.session_state["history"]` and is written only through
`retrieval_engine.remember()`, which caps it, trims it in whole exchanges, and
redacts it. Nothing here appends to that list directly.

Keeping the rules in the engine rather than here is not tidiness. A window is
both prompt material and a rendered transcript, so the two things that must not
happen — PII surviving into later turns, and an answer being sourced from an
earlier answer instead of from a retrieved chunk — are properties of the engine's
prompt, not of the widget tree. A UI-local `history.append(...)` would be
invisible to every test that checks the prompt, which is where those properties
are actually verifiable.

The model id is read from `.env` (`GROQ_MODEL`) on every call, never hardcoded
here. Note that continuity comes from the window, not from the model: swapping
`GROQ_MODEL` changes how answers are phrased, and a new model starts a fresh
session with no memory of earlier turns. The model is configurable so a retired
model can be swapped without a code change; it is not what remembers anything.

Two operational notes, both of which this file is shaped around:

**Caching is not optional here.** Streamlit re-executes the entire script on
every keystroke, button press, and rerun. Without `st.cache_resource` each one of
those would re-embed 377 chunks (~20 s warm, ~60 s cold) and re-read the index.
`_engine()` is decorated so the model and the index are built once per server
process, not once per interaction.

**The first load is slow and that is expected.** If `chroma_db/` is missing —
it is gitignored, so a fresh deploy has none — the engine materialises it from
`data/processed/chunks.jsonl`, and the embedding model downloads ~90 MB on a cold
cache. That is tens of seconds before the first question is answerable. The
loading notice below says so, because a blank screen for a minute reads as a
crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

#: The repository root, not this file's directory. This file lives in ``src/``
#: next to the modules it drives, and Streamlit runs it with ``src/`` as the
#: working directory, so the project root has to be put on the path explicitly
#: before ``from src import ...`` can resolve.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import retrieval_engine as E          # noqa: E402
from src.config import (                       # noqa: E402
    ALLOWED_URLS,
    MAX_SENTENCES,
    SCHEME_BY_ID,
    SCHEMES,
)
from src.guardrails import EDUCATIONAL_URL     # noqa: E402
from src.textutils import split_sentences      # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Content
# ═══════════════════════════════════════════════════════════════════════════

TITLE = "HDFC Mutual Fund — Facts-only FAQ"

WELCOME = (
    "Hi — I'm a facts-only assistant for 5 HDFC Mutual Fund schemes: Large Cap, "
    "Flexi Cap, ELSS Tax Saver, Small Cap, and Balanced Advantage. Ask me about "
    "expense ratio, exit load, minimum SIP, lock-in, riskometer, or benchmark."
)

#: PRD §4.6 / §5.7 — the literal disclosure string. Not paraphrased anywhere.
DISCLOSURE = "Facts-only. No investment advice."

#: Exactly three, and **no two share a scheme** (PRD §10.1). Chosen so the first
#: three things a reviewer sees each exercise a different failure mode rather
#: than three easy lookups:
#:
#: 1. Large Cap / expense ratio — the §3.12 blocker. The corpus holds
#:    `Expense ratio (TER): 1.03%` *and* `Basic expense ratio (excluding addl.
#:    TER): 0.84%` plus a glossary definition that outscores both. Answering
#:    0.84% means the distractor won.
#: 2. ELSS / lock-in — the 80C obligation. The corpus states the 3-year lock-in
#:    and says nothing about the statutory basis, so this is where a model
#:    invents a tax claim; the answer must be 3 years and nothing about 80C.
#: 3. Small Cap / benchmark — the answer contains an abbreviation (`TRI`), which
#:    the generator has to expand correctly rather than echo as `Label: value`.
#:
#: All three are verified answerable. The spec's own third example, "How do I
#: download my capital gains statement?", is **deliberately not used**: the
#: corpus has no download instructions (the ingest strips Groww's nav block), so
#: it can only be declined. §5.7 is explicit — replace an unanswerable example
#: rather than ship it to fail live, because AC-16 requires 3/3 real answers and
#: the first thing a reviewer sees should be a working one.
EXAMPLES: tuple[str, ...] = (
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the lock-in period of HDFC ELSS Tax Saver Fund?",
    "What is the benchmark of HDFC Small Cap Fund?",
)

#: Sidebar topics, phrased as what the corpus can actually answer.
TOPICS = ("objective", "expense ratio", "exit load", "minimum SIP or lumpsum",
          "benchmark", "riskometer", "NAV", "AUM", "fund manager")

#: Shown next to the memory counter so the window size is not a mystery.
MEMORY_NOTE = (
    f"I remember the last {E.MEMORY_MESSAGES} messages of this conversation, so "
    f"you can follow up with “what about its exit load?” instead of repeating "
    f"the fund's name. I use that memory only to work out which fund you mean — "
    f"every figure I give still comes from the published pages I cite, never "
    f"from something I said earlier."
)

_STYLE = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 54rem; }
  /* st.title renders <h1><a>…</a></h1>; the anchor is the real title text. */
  h1, h1 a { letter-spacing: -0.02em; }
  .answer-card {
      border: 1px solid rgba(128,128,128,0.28);
      border-left: 4px solid rgba(31,119,180,0.85);
      border-radius: 0.5rem;
      padding: 0.9rem 1.1rem;
      margin: 0.2rem 0 0.6rem 0;
  }
  .answer-card.refusal { border-left-color: rgba(214,139,0,0.9); }
  .answer-text { font-size: 1.06rem; line-height: 1.55; }
  .transcript-turn {
      border-left: 2px solid rgba(128,128,128,0.3);
      padding: 0.15rem 0 0.15rem 0.8rem;
      margin: 0 0 0.85rem 0;
      font-size: 0.95rem;
  }
  .transcript-q { font-weight: 600; }
  .transcript-a { color: rgba(60,60,60,1); }
  .transcript-meta { font-size: 0.76rem; color: rgba(125,125,125,1); }
  .examples-label {
      font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.09em;
      color: rgba(120,120,120,1); margin: 0.9rem 0 0.35rem 0;
  }
  button[kind="primary"] { font-weight: 600; }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════
# Conversation memory — state lives here, rules live in the engine
# ═══════════════════════════════════════════════════════════════════════════

#: The one session key that holds the window. Named so it is greppable; every
#: read and write goes through the helpers below.
HISTORY_KEY = "history"


def get_history() -> list[dict]:
    """The rolling window, or an empty one.

    ``st.session_state.get`` rather than indexing, so this is safe to call before
    anything has been asked — which it is, on every rerun that is not a submit.
    """
    return list(st.session_state.get(HISTORY_KEY) or [])


def set_history(history: list[dict]) -> None:
    """Store the window. Assignment, not mutation.

    Streamlit only re-renders when a session-state key is *rebound*. An in-place
    ``list.append`` would leave the transcript on screen one turn behind, which
    looks like a dropped message.
    """
    st.session_state[HISTORY_KEY] = list(history)


def record_turn(question: str, response: dict) -> None:
    """Append one question and its answer to the window.

    The question is stored as typed. `remember` redacts it on the way in, so what
    lands in the transcript is the redacted form — a user who pasted a PAN is
    already being told to retype, and echoing their PAN back at them in the
    scrollback would be strictly worse than the leak it replaces.
    """
    history = get_history()
    history = E.remember(history, "user", question)
    history = E.remember(history, "assistant", response.get("text", ""),
                         refused=bool(response.get("is_refusal")))
    set_history(history)


# ═══════════════════════════════════════════════════════════════════════════
# Engine access — cached once per server process
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _engine():
    """Load the index and the embedding model once, not per interaction.

    Returns the engine module itself. There is no state to thread through: the
    module memoises the collection and the model, and both are read-only from
    here. The Groq model id is *not* cached — it is read from `.env` per call, so
    rotating the key or swapping the model takes effect without a restart.
    """
    return E


@st.cache_resource(show_spinner=False)
def _index_note() -> str:
    """One-line description of how the index became available.

    Worth surfacing in the UI: a rebuild is a deployment event that changes the
    latency of the first answer by a minute, and an operator debugging a slow
    cold start should not have to guess why.
    """
    state = E.index_state()
    if state.get("rebuilt"):
        return (f"Index rebuilt on startup from `chunks.jsonl` "
                f"({state.get('count', 0)} chunks).")
    return f"Index ready — {E._collection().count()} chunks."


@st.cache_data(show_spinner=False)
def _model_note() -> str:
    """Which model is answering, and where that name came from.

    Surfaced because the model id lives in `.env` and is otherwise invisible. A
    reviewer who sees a flat or oddly-worded answer should be able to find out
    which model produced it without reading the code. The name is reported, never
    asserted as good: `call_groq` reports a retired model as a 404 naming
    GROQ_MODEL, and the answer path then falls back to extraction — so "which
    model" is a real operational question, not a cosmetic one.
    """
    _key, model = E._credentials()
    if not model:
        return "⚠️ `GROQ_MODEL` is not set in `.env` — answers fall back to "\
               "extraction from the retrieved text, not the language model."
    return f"Answers generated by `{model}` (from `.env`; retrieved facts are "\
           "always from the cited pages)."


def _use_example(question: str) -> None:
    """Prefill the input box from an example button *and* submit it.

    An `on_click` callback rather than an assignment in the script body: by the
    time the body runs, the text_input widget already exists for this rerun, and
    assigning to a widget's session_state key after it has been instantiated
    raises `StreamlitAPIException`. Callbacks run *before* the body, so this is
    the supported way to do it.

    Submitting as well as prefilling is deliberate. §5.7 requires that the first
    thing a reviewer sees is a *working answer*, and a preset question that
    silently does nothing until a second click is pressed is the kind of small
    dead end that reads as a broken demo. The text lands in the box too, so the
    question that produced the answer stays visible and editable.
    """
    st.session_state["question"] = question
    st.session_state["submit"] = True


def _take_submit_flag() -> bool:
    """Read-and-clear the one-shot submit flag.

    `del st.session_state[key]` is the documented way to drop a non-widget key;
    `.pop()` is only probed for above, not relied on, because the proxy is not
    part of the stable public API. The flag must be cleared or the answer would
    re-render on every subsequent rerun — which, given the answer costs a
    retrieval and a generation, would make the page feel stuck.
    """
    if "submit" in st.session_state:
        flag = bool(st.session_state["submit"])
        del st.session_state["submit"]
        return flag
    return False


def _new_conversation() -> None:
    """Clear the window and the input.

    Needed because the window is session state: without it, "what about its exit
    load?" on a fresh page would silently resolve against whatever fund the last
    person asked about, which is a confident answer to a question that had no
    subject.
    """
    st.session_state[HISTORY_KEY] = []
    st.session_state.pop("question", None)


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

def _assess(response: dict) -> dict:
    """Derive everything the renderer needs to decide, and nothing else.

    Split out from :func:`_render_answer` so the decisions are testable without a
    Streamlit server. `st.*` calls are no-ops outside `streamlit run`, so a
    function that both decides and renders cannot be verified headlessly — and
    these are the decisions that determine whether a rule violation is visible
    to the user or hidden by the display layer.
    """
    text = (response.get("text") or "").strip()
    citation = response.get("citation") or {}
    url = (citation.get("url") or "").strip()
    # Re-count rather than trusting `response["sentences"]`. That field is a
    # convenience the engine also supplies; measuring the string that is about
    # to be displayed is the only check that describes what the user sees.
    shown = len(split_sentences(text))
    return {
        "text": text,
        "url": url,
        "label": (citation.get("label") or "source").strip(),
        "is_refusal": bool(response.get("is_refusal")),
        "shown": shown,
        "transparency": response.get("transparency_line") or "",
        "over_cap": shown > MAX_SENTENCES,
        # R2: a link is only rendered if it is on the allow-list. Refusals cite
        # the educational page, which is deliberately *not* in ALLOWED_URLS —
        # it is a separate set, so a refusal still gets exactly one link.
        "link_ok": bool(url) and (url == EDUCATIONAL_URL or url in ALLOWED_URLS),
    }


def _render_answer(response: dict) -> None:
    """Render one response, with the citation and transparency line mandatory."""
    a = _assess(response)

    with st.container():
        st.markdown(
            f'<div class="answer-card{" refusal" if a["is_refusal"] else ""}">'
            f'<div class="answer-text">{_escape(a["text"])}</div></div>',
            unsafe_allow_html=True,
        )

        if a["over_cap"]:
            # Shown in full, not truncated. The R1 ladder is supposed to make
            # this unreachable; hiding the overflow would turn an engine bug
            # into an invisible one.
            st.error(
                f"⚠️ This answer is {a['shown']} sentences, over the "
                f"{MAX_SENTENCES}-sentence cap. It is shown in full rather than "
                f"silently trimmed — this is an engine bug, not a display choice."
            )
        if a["url"] and not a["link_ok"]:
            st.error(
                f"⚠️ Citation URL is not on the allow-list, so it is not being "
                f"linked: `{a['url']}`"
            )

        # R2: exactly one link, always.
        if a["link_ok"]:
            st.caption(f"**Source:** [{a['label']}]({a['url']})")
        else:
            st.caption(f"**Source:** {a['label']} (link withheld — not allow-listed)")
        # R7: the transparency line, on answers and refusals alike.
        st.caption(a["transparency"])
        st.caption(
            f"{a['shown']} sentence{'s' if a['shown'] != 1 else ''} of max "
            f"{MAX_SENTENCES} · facts only, no advice"
        )
        if a["is_refusal"]:
            st.caption("No advice given — published facts only.")


def _escape(text: str) -> str:
    """HTML-escape before injecting into the card.

    The answer text is model output over a fixed prompt. It is not expected to
    contain markup, but it is untrusted input into `unsafe_allow_html=True`, and
    the cost of escaping is zero.
    """
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))


def _render_transcript(history: list[dict]) -> None:
    """Show the remembered turns above the input.

    The window is invisible otherwise, and an invisible memory is worse than none:
    the user cannot tell whether "what about its exit load?" was understood
    because the bot remembered the fund or because it guessed. Showing the
    question the answer was actually about makes that legible, and makes the
    10-message cap observable instead of surprising.

    Every string here came out of `remember`, which redacts, so the transcript
    cannot reintroduce PII that the window was built to keep out.
    """
    if not history:
        return
    st.markdown('<div class="examples-label">This conversation</div>',
                unsafe_allow_html=True)
    for message in history:
        if message.get("role") == "user":
            st.markdown(
                f'<div class="transcript-turn"><div class="transcript-q">'
                f'You asked: {_escape(message.get("text", ""))}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            refused = message.get("refused")
            st.markdown(
                f'<div class="transcript-turn"><div class="transcript-a">'
                f'{_escape(message.get("text", ""))}</div>'
                f'<div class="transcript-meta">'
                f'{"declined — not a fact I can reuse" if refused else "answered"}'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Page
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="HDFC Mutual Fund — Facts-only FAQ",
        page_icon="📊",
        layout="centered",
    )
    st.markdown(_STYLE, unsafe_allow_html=True)

    # ── warm the engine before anything interactive ─────────────────────
    # Done here rather than lazily on first submit so the one-off cost is paid
    # while the user is reading the page, not while they are waiting on a
    # question they already asked. `chroma_db/` is committed, so the usual path
    # is an index open costing milliseconds and the model is not loaded until
    # the first question actually needs it. The spinner therefore only runs
    # long when the committed index was unusable and had to be rebuilt, which is
    # worth surfacing rather than hiding — see `_index_note`.
    try:
        with st.spinner("Preparing the search index (first run only)…"):
            engine = _engine()
            index_note = _index_note()
            model_note = _model_note()
    except Exception as exc:                               # noqa: BLE001
        st.error(
            "The search index could not be loaded, so no question can be "
            f"answered yet. Details: `{type(exc).__name__}: {exc}`"
        )
        st.caption(
            "This is a startup problem, not a question problem — the corpus or "
            "the embedding model is unavailable. See the README's cold-start "
            "notes."
        )
        return

    # ── header ──────────────────────────────────────────────────────────
    # st.title, not a hand-rolled <h1>: it is the element the spec names, and it
    # gives the page a real heading for screen readers and the browser tab.
    st.title(TITLE)
    st.caption(WELCOME)
    st.info(DISCLOSURE)          # the exact required string (PRD §4.6)
    if index_note.startswith("Index rebuilt"):
        st.caption(index_note)

    # ── examples ────────────────────────────────────────────────────────
    if not get_history():
        # Only on a fresh conversation. Once there is a window, three more
        # buttons above a transcript is four ways to ask something and no clear
        # "ask about this fund" affordance.
        st.markdown('<div class="examples-label">Try one of these</div>',
                    unsafe_allow_html=True)
        for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            # key= is the example text, so each button is its own stable widget.
            col.button(example, key=f"ex::{example}", use_container_width=True,
                       on_click=_use_example, args=(example,))

    # ── input ───────────────────────────────────────────────────────────
    question = st.text_input(
        "Ask a question about any of the 5 schemes",
        key="question",
        placeholder="e.g. What is the exit load of HDFC Flexi Cap Fund?",
        label_visibility="collapsed",
    )
    ask_col, new_col = st.columns([1, 6])
    asked = ask_col.button("Ask", type="primary", use_container_width=True)
    if new_col.button("New conversation", use_container_width=False,
                      on_click=_new_conversation):
        st.rerun()

    # ── answer ──────────────────────────────────────────────────────────
    # Either the Ask button, or an example button that already set the flag.
    submit = asked or _take_submit_flag()
    if submit and (question or "").strip():
        asked_text = question.strip()
        try:
            with st.spinner("Looking this up in the published scheme pages…"):
                response, trace = engine.answer_with_trace(
                    asked_text, get_history()
                )
        except Exception as exc:                           # noqa: BLE001
            st.error(f"Something went wrong answering that: `{exc}`")
            return
        # Recorded *after* the answer returns, so a turn that raised is not
        # stored as though it had been answered.
        record_turn(asked_text, response)
        _render_answer(response)
        if trace.get("scheme_inherited"):
            # Say so. Memory silently reinterpreting a question is the one
            # behaviour here that could surprise someone, so it is stated rather
            # than left for the user to infer from the answer.
            st.caption(
                f"Read “{asked_text}” as being about "
                f"{SCHEME_BY_ID[trace['scheme_inherited']].name} — from earlier "
                f"in this conversation. Name a different fund to switch."
            )
        _render_transcript(get_history())

    # ── sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### In scope")
        st.caption("5 schemes · all Direct Growth")
        for scheme in SCHEMES:
            st.markdown(f"- **{scheme.name}**  \n  <small>{scheme.category}</small>",
                        unsafe_allow_html=True)
        st.divider()
        st.markdown("### I can answer")
        st.caption(" · ".join(TOPICS))
        st.divider()
        history = get_history()
        st.markdown("### Memory")
        st.caption(
            f"Remembering {len(history)} of {E.MEMORY_MESSAGES} messages this "
            f"session."
        )
        st.caption(MEMORY_NOTE)
        st.divider()
        st.caption(model_note)
        st.divider()
        st.caption(
            "I do not give investment advice, compare returns, or recommend a "
            "fund. I only restate what the published scheme pages say, and I "
            "cite the page for every figure."
        )
        st.caption(
            "Questions outside these 5 schemes, or about other plan variants, "
            "get a scope note rather than an answer."
        )


if __name__ == "__main__":
    main()
