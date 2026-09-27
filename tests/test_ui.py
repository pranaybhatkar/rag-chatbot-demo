"""tests/test_ui.py — the Streamlit front end, driven through a real runtime.

Why this file exists at all
--------------------------
Every other phase in this project is verified by calling the code. A Streamlit
app cannot be verified that way: outside a `streamlit run` session every `st.*`
call is a no-op that writes to a dummy sink, so importing src/app.py and calling
`app._render_answer(...)` proves nothing about what the user sees. A function
that both decides and renders is therefore untestable without a server.

`streamlit.testing.v1.AppTest` closes that gap. It executes the real script
against a real Streamlit runtime and hands back the widget tree, so these tests
assert on the actual title, the actual buttons, the actual markdown, and the
actual links — the same things a reviewer would see.

The split in src/app.py between `_assess` (pure decisions) and `_render_answer`
(presentation) exists because of this. `_assess` is cheap to unit-test directly;
the interaction tests below cover the wiring that pure functions cannot.

Scoping note: the engine is memoised with `st.cache_resource`, which is
process-global, so the ~60 s cold start is paid once for the whole module and
the interaction tests run against a warm index.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "src" / "app.py"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX = ROOT / "chroma_db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

streamlit = pytest.importorskip("streamlit", reason="pip install streamlit==1.42.2")
from streamlit.testing.v1 import AppTest                      # noqa: E402

from src import app                                          # noqa: E402
from src import retrieval_engine                              # noqa: E402
from src.config import ALLOWED_URLS, MAX_SENTENCES, SCHEMES   # noqa: E402
from src.guardrails import EDUCATIONAL_URL                    # noqa: E402

#: Read the examples from the app rather than restating them. A copy in this
#: file would drift, and a drifted copy is worse than no check: it would keep
#: passing while the shipped buttons changed. Importing app is safe — its
#: `st.*` calls all live under main(), which the `__main__` guard keeps out of
#: import.
EXAMPLE_QUESTIONS = app.EXAMPLES

#: The engine needs a corpus, and it needs an index it can build or open. Both
#: are absent on a bare checkout, and the UI cannot be exercised without them.
pytestmark = pytest.mark.skipif(
    not CHUNKS.exists(),
    reason="run scripts/build_index.py --stages 1 2 first",
)

#: A cold start is minutes; a stalled one is a hang. 300 s is well clear of the
#: measured ~60 s while still failing rather than blocking forever.
TIMEOUT = 300

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _boot() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, f"app raised on boot: {at.exception}"
    return at


def _example_buttons(at: AppTest):
    """Every preset button, identified by its label.

    Selected against `app.EXAMPLES` rather than by "all buttons except Ask",
    because the page also carries a "New conversation" button and that shortcut
    silently starts counting it as a fourth example the moment one is added.
    """
    hits = [b for b in at.button if b.label in set(EXAMPLE_QUESTIONS)]
    return hits


def _ask_button(at: AppTest):
    """The Ask button, located by label.

    `at.button(key=...)` matches Streamlit's internal widget id, not the visible
    label. The Ask button is auto-keyed from its label plus a per-run suffix, so
    its id changes every session and matching on it is a silent no-op waiting to
    happen. Matching on the label is the only stable handle.
    """
    hits = [b for b in at.button if b.label == "Ask"]
    assert len(hits) == 1, f"expected exactly 1 Ask button, found {len(hits)}"
    return hits[0]


def _button(at: AppTest, label: str):
    """Any button, located by label. Returns None if absent."""
    hits = [b for b in at.button if b.label == label]
    assert len(hits) <= 1, f"expected at most 1 {label!r} button, found {len(hits)}"
    return hits[0] if hits else None


def _example_button(at: AppTest, question: str):
    """The preset button for `question`, located by label.

    src/app.py gives these an explicit `key=f"ex::{question}"`, so the key is
    stable — but matching on the label keeps this test independent of that choice.
    """
    hits = [b for b in at.button if b.label == question]
    assert len(hits) == 1, f"expected 1 example button for {question!r}, got {len(hits)}"
    return hits[0]


def _all_links(at: AppTest) -> list[tuple[str, str]]:
    """Every markdown link the app emitted, in the order a reader would meet it."""
    found: list[tuple[str, str]] = []
    for el in list(at.markdown) + list(at.caption):
        found += LINK_RE.findall(el.value)
    return found


def _visible_text(at: AppTest) -> str:
    """All rendered prose, for substring assertions on what is on screen."""
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [i.value for i in at.info] + [w.value for w in at.warning]
    parts += [e.value for e in at.error]
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Module-scoped boot
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def booted() -> AppTest:
    return _boot()


# ═══════════════════════════════════════════════════════════════════════════
# The five elements the brief names
# ═══════════════════════════════════════════════════════════════════════════

def test_welcome_title_header(booted: AppTest) -> None:
    """A welcome title header, via st.title so the page gets a real <h1>."""
    assert [t.value for t in booted.title] == ["HDFC Mutual Fund — Facts-only FAQ"]


def test_disclosure_label_is_exact(booted: AppTest) -> None:
    """The disclosure label states exactly the required sentence.

    Compared against the literal, not a substring: a paraphrase such as
    "This is facts-only, no investment advice" reads as compliant at a glance
    and is not.
    """
    assert "Facts-only. No investment advice." in [i.value for i in booted.info]


def test_exactly_three_example_questions(booted: AppTest) -> None:
    """Exactly 3 preset questions, as buttons."""
    labels = [b.label for b in _example_buttons(booted)]
    assert len(labels) == 3, f"expected 3 example buttons, found {labels}"


def test_custom_question_input_bar(booted: AppTest) -> None:
    """One text input for custom questions, plus the Ask button."""
    assert len(booted.text_input) == 1
    assert booted.text_input[0].placeholder
    assert _ask_button(booted).label == "Ask"
    assert _button(booted, "New conversation") is not None, "no way to clear memory"


def test_answer_area_renders_with_exactly_one_link(booted: AppTest) -> None:
    """The answer, its citation link, and its transparency line all render.

    Driven through a preset click rather than asserted on the boot screen,
    because the answer area is empty until a question is asked.
    """
    question = "What is the expense ratio of HDFC Large Cap Fund?"
    at = _boot()
    _example_button(at, question).click().run()
    assert not at.exception, f"app raised while answering: {at.exception}"

    links = _all_links(at)
    assert len(links) == 1, f"R2 violated: {len(links)} links rendered: {links}"
    assert links[0][1] in ALLOWED_URLS, f"link is not allow-listed: {links[0][1]}"

    text = _visible_text(at)
    assert "1.03" in text, "the answer itself is not on screen"
    assert "Last updated from sources: " in text, "R7 transparency line missing"
    assert f"of max {MAX_SENTENCES}" in text, "sentence count not shown"


# ═══════════════════════════════════════════════════════════════════════════
# PRD §10.1 — the examples must not all be about the same fund
# ═══════════════════════════════════════════════════════════════════════════

def test_examples_span_three_distinct_schemes(booted: AppTest) -> None:
    """No two preset questions share a scheme.

    Asserted against the guardrails' own resolver rather than a string match,
    so an alias or a renamed scheme cannot quietly make two examples collide.
    """
    from src.guardrails import resolve_scheme

    ids = []
    for b in _example_buttons(booted):
        scheme_id, ambiguous = resolve_scheme(b.label)
        assert not ambiguous, f"example is ambiguous: {b.label!r}"
        ids.append(scheme_id)
    assert len(set(ids)) == 3, f"examples collapse onto {len(set(ids))} scheme(s): {ids}"


# ═══════════════════════════════════════════════════════════════════════════
# AC-16 — all three examples must answer, live, in the real UI
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("question", EXAMPLE_QUESTIONS, ids=lambda q: q[:34])
def test_every_example_answers_live(question: str) -> None:
    """Each preset produces a real answer inside the 3-sentence cap.

    §5.7 is explicit that no example may trigger a refusal, because the first
    thing a reviewer sees must be a working answer. Asserted end-to-end through
    the UI so an example cannot be left in place after the corpus changes.
    """
    at = _boot()
    _example_button(at, question).click().run()
    assert not at.exception

    assessed = app._assess(retrieval_engine.answer(question))

    assert not assessed["is_refusal"], f"example triggers a refusal: {question!r}"
    assert 1 <= assessed["shown"] <= MAX_SENTENCES, (
        f"{assessed['shown']} sentences for {question!r}"
    )
    assert not assessed["over_cap"]
    assert assessed["link_ok"], f"no allow-listed citation for {question!r}"
    assert assessed["transparency"].startswith("Last updated from sources: ")
    assert not at.error, f"rendered an error box: {[e.value for e in at.error]}"


# ═══════════════════════════════════════════════════════════════════════════
# Interaction
# ═══════════════════════════════════════════════════════════════════════════

def test_typing_a_custom_question_answers_it() -> None:
    """The custom input path works end to end, not just the preset path."""
    at = _boot()
    at.text_input[0].set_value("What is the exit load of HDFC Flexi Cap Fund?").run()
    _ask_button(at).click().run()

    assert not at.exception, f"app raised: {at.exception}"
    text = _visible_text(at).lower()
    assert "exit load" in text
    links = _all_links(at)
    assert len(links) == 1, f"R2 violated on the typed path: {links}"


def test_preset_click_submits_without_a_second_click() -> None:
    """Clicking a preset answers immediately and leaves the question in the box.

    A preset that only prefills is a dead end the reviewer has to guess their way
    out of. The text is still set, so the question behind the answer stays
    visible and editable.
    """
    question = "What is the lock-in period of HDFC ELSS Tax Saver Fund?"
    at = _boot()
    _example_button(at, question).click().run()

    assert at.text_input[0].value == question
    assert "Last updated from sources: " in _visible_text(at), (
        "a preset click prefilled the box but produced no answer"
    )


def test_the_submit_flag_does_not_stick() -> None:
    """A preset answer does not re-render on later reruns.

    Without clearing the one-shot flag, every subsequent interaction would
    re-answer the old question, which looks like the app is stuck.
    """
    at = _boot()
    _example_button(at, "What is the benchmark of HDFC Small Cap Fund?").click().run()
    first = _visible_text(at)
    assert "Last updated from sources: " in first

    at.text_input[0].set_value("").run()
    after = _visible_text(at)
    assert "Last updated from sources: " not in after, (
        "the previous answer persisted after the input was cleared"
    )


def test_asking_with_an_empty_box_does_not_fabricate() -> None:
    """Ask with no text must not crash and must not invent an answer."""
    at = _boot()
    _ask_button(at).click().run()
    assert not at.exception
    assert "Last updated from sources: " not in _visible_text(at), (
        "an answer was rendered for an empty question"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Refusals must still look and cite like answers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "question",
    [
        "Should I invest in HDFC Small Cap Fund?",
        "What is the 1 year return of HDFC Large Cap Fund?",
        "My PAN is ABCDE1234F, what is the exit load of HDFC Large Cap Fund?",
        "How do I download my capital gains statement?",
    ],
    ids=["advice", "performance", "pii", "out-of-corpus"],
)
def test_refusals_render_cited_dated_and_without_advice(question: str) -> None:
    """A refusal is still a full response: one link, dated, and no advice.

    The refusal templates are user-visible text, so a template bug is a product
    bug. Each case below is a different guard in src/guardrails.py.
    """
    at = _boot()
    at.text_input[0].set_value(question).run()
    _ask_button(at).click().run()

    assert not at.exception, f"app raised on a refusal: {at.exception}"
    links = _all_links(at)
    assert len(links) == 1, f"a refusal must still cite exactly one source: {links}"
    assert links[0][1] == EDUCATIONAL_URL, (
        f"a refusal must cite the educational page, not a scheme page: {links[0][1]}"
    )
    text = _visible_text(at)
    assert "Last updated from sources: " in text, "R7 must hold on refusals too"
    assert "No advice given" in text
    assert not at.error, f"a refusal is not an error: {[e.value for e in at.error]}"


# ═══════════════════════════════════════════════════════════════════════════
# The UI's own guards
# ═══════════════════════════════════════════════════════════════════════════

def test_over_cap_answers_warn_instead_of_being_truncated() -> None:
    """The display layer never silently shortens an answer.

    The engine's R1 ladder should make a 4-sentence answer unreachable. If it
    ever happens, the reviewer must see all four sentences plus a warning —
    truncating would convert an engine bug into an invisible one.
    """
    good = retrieval_engine.answer(app.EXAMPLES[0])
    over = dict(good, text="One fact here. Two fact here. Three fact here. Four fact here.")
    assessed = app._assess(over)

    assert assessed["shown"] == 4
    assert assessed["over_cap"] is True
    assert "Four fact here." in assessed["text"], "the text must survive intact"


def test_a_non_allow_listed_citation_is_never_linked() -> None:
    """A citation the engine should never produce still cannot become a link.

    Defence in depth: `_assess` is the last gate before a URL is clickable, so it
    is tested even though nothing upstream can currently feed it a bad value.
    """
    good = retrieval_engine.answer(app.EXAMPLES[0])
    assessed = app._assess(
        dict(good, citation={"label": "rogue", "url": "https://evil.example.com/x"})
    )
    assert assessed["link_ok"] is False
    assert assessed["label"] == "rogue", "the label is still shown, just not linked"
    assert "evil.example.com" in assessed["url"], "the bad URL is kept for the warning"


def test_answer_text_is_html_escaped() -> None:
    """Model output is injected with unsafe_allow_html, so it is escaped.

    The generator is prompted not to emit markup, but "the prompt says so" is not
    a control — this is the control.
    """
    assert app._escape("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert app._escape("A & B") == "A &amp; B"


# ═══════════════════════════════════════════════════════════════════════════
# Scope
# ═══════════════════════════════════════════════════════════════════════════

def test_sidebar_lists_the_five_schemes(booted: AppTest) -> None:
    """The sidebar names all 5 in-scope schemes."""
    sidebar = "\n".join(m.value for m in booted.sidebar.markdown)
    assert "In scope" in sidebar
    for scheme in SCHEMES:
        assert scheme.name in sidebar, f"{scheme.name} missing from the sidebar"


# ═══════════════════════════════════════════════════════════════════════════
# Conversation memory
# ═══════════════════════════════════════════════════════════════════════════

def test_the_window_is_visible_in_the_sidebar(booted: AppTest) -> None:
    """The memory state is reported, not hidden.

    An invisible window is worse than none: the user cannot tell whether a
    follow-up was understood or guessed. The counter also makes the 10-message
    cap observable instead of surprising.
    """
    sidebar = "\n".join(m.value for m in booted.sidebar.markdown) + \
        "\n".join(c.value for c in booted.sidebar.caption)
    assert "Memory" in sidebar
    assert "Remembering 0 of 10 messages" in sidebar
    assert "never from something I said earlier" in sidebar, (
        "the sidebar must say memory is not a source of facts"
    )


def test_a_second_turn_lands_in_the_window() -> None:
    """Asking twice fills the window, and the transcript shows both turns."""
    at = _boot()
    at.text_input[0].set_value("What is the exit load of HDFC Flexi Cap Fund?").run()
    _ask_button(at).click().run()
    assert not at.exception

    page = _visible_text(at)
    assert "You asked: What is the exit load of HDFC Flexi Cap Fund?" in page, (
        "the question was not recorded in the transcript"
    )
    assert "answered" in page, "the answer turn was not marked as answered"

    sidebar = "\n".join(c.value for c in at.sidebar.caption)
    assert "Remembering 2 of 10 messages" in sidebar, sidebar

    at.text_input[0].set_value("What is the minimum SIP for HDFC Large Cap Fund?").run()
    _ask_button(at).click().run()
    assert "Remembering 4 of 10 messages" in \
        "\n".join(c.value for c in at.sidebar.caption)


def test_a_follow_up_resolves_against_the_window() -> None:
    """A bare follow-up is answered about the fund named a turn earlier.

    This is the whole point of the window. "What about its exit load?" names no
    fund, so without memory the corpus holds the answer and the question cannot
    reach it.
    """
    at = _boot()
    at.text_input[0].set_value("What is the benchmark of HDFC Small Cap Fund?").run()
    _ask_button(at).click().run()
    at.text_input[0].set_value("What about its exit load?").run()
    _ask_button(at).click().run()
    assert not at.exception

    page = _visible_text(at)
    assert "Last updated from sources: " in page
    assert "declined" not in page.lower().split("you asked:")[-1], (
        "the follow-up was declined rather than answered from memory"
    )
    # The UI states that it reinterpreted the question, so the behaviour is
    # visible instead of the user having to infer it.
    assert "from earlier in this conversation" in page, (
        "scheme carry-over happened but was not disclosed to the user"
    )


def test_new_conversation_clears_the_window() -> None:
    """The reset button empties the window, so a follow-up cannot inherit."""
    at = _boot()
    at.text_input[0].set_value("What is the exit load of HDFC Flexi Cap Fund?").run()
    _ask_button(at).click().run()
    assert "Remembering 2 of 10" in "\n".join(c.value for c in at.sidebar.caption)

    reset = [b for b in at.button if b.label == "New conversation"]
    assert len(reset) == 1, "the reset button is missing"
    reset[0].click().run()
    assert not at.exception

    sidebar = "\n".join(c.value for c in at.sidebar.caption)
    assert "Remembering 0 of 10" in sidebar, sidebar
    assert "You asked:" not in _visible_text(at), "the transcript survived a reset"


def test_the_window_never_shows_pii_back_to_the_user() -> None:
    """A PAN typed in one turn is not redisplayed in the transcript.

    The guard redacts the question it is answering, but the *stored* copy is a
    separate surface: it is re-sent to the model on later turns and re-rendered in
    the scrollback. Echoing a user's PAN back at them would be worse than the
    leak it replaces.
    """
    at = _boot()
    at.text_input[0].set_value(
        "My PAN is ABCDE1234F, what is the exit load of HDFC Large Cap Fund?"
    ).run()
    _ask_button(at).click().run()
    assert not at.exception

    everything = _visible_text(at) + "\n" + \
        "\n".join(m.value for m in at.markdown)
    assert "ABCDE1234F" not in everything, "the PAN was redisplayed"
    assert "[REDACTED_PAN]" in everything, "the redacted form should be what is shown"


def test_the_model_comes_from_dotenv(booted: AppTest) -> None:
    """The sidebar names the model, and the name came from `.env`.

    Asserted as a shape, not a value. Pinning `openai/gpt-oss-20b` in a test is
    exactly the rot this project has already been bitten by twice — a retired
    model id fails the test instead of the app, which is backwards.
    """
    _key, model = retrieval_engine._credentials()
    assert model, "GROQ_MODEL is not set in .env"
    sidebar = "\n".join(c.value for c in booted.sidebar.caption)
    assert model in sidebar, "the active model is not surfaced in the UI"
    assert "from `.env`" in sidebar


def test_the_window_holds_only_redacted_text() -> None:
    """`remember` is the only writer, and it redacts.

    Asserted at the engine boundary rather than through the UI, because this is
    the property the prompt depends on and a widget-level test cannot see it.
    """
    from src.guardrails import detect_pii

    history = []
    for i in range(1, 4):
        history = retrieval_engine.remember(
            history, "user", f"My PAN is ABCDE1234F, question {i}?"
        )
        history = retrieval_engine.remember(
            history, "assistant", f"Answer {i}."
        )
    blob = " ".join(m["text"] for m in history)
    assert "ABCDE1234F" not in blob
    assert detect_pii(blob, mode="user") == []
    assert len(history) == 6
