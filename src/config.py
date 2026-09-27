"""config.py — scope, sources, and hard limits.

Every hard limit in PRD.md lives here as a named constant, so no magic number
is buried in business logic and tests can assert on the configuration itself.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Directories ────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # hand-placed / fetched. Never generated.
PROCESSED_DIR = DATA_DIR / "processed"  # stage outputs (documents.jsonl, chunks…)
MANIFEST_DIR = DATA_DIR / "manifest"    # ingest manifest + run log
INDEX_DIR = DATA_DIR / "index" / "chroma"
CONFIG_DIR = ROOT / "config"
DELIVERABLES_DIR = ROOT / "deliverables"
SOURCES_CSV = CONFIG_DIR / "sources.csv"

# ── R1: strict 3-sentence response limit (PRD §4.1) ───────────────────────
MAX_SENTENCES = 3

# ── R2/R4: retrieval (PRD §4.2, §7.4) ─────────────────────────────────────
TOP_K = 8
MMR_LAMBDA = 0.7
CONFIDENCE_FLOOR = 0.62

# ── R7: transparency line (PRD §4.7) ───────────────────────────────────────
TRANSPARENCY_PREFIX = "Last updated from sources: "

# ── Mandated stack (brief lines 32-35) ─────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
#: all-MiniLM-L6-v2 truncates silently past 256 word-pieces. Not a tunable.
WORDPIECE_HARD_CAP = 256
CHROMA_COLLECTION = "hdfc_faq"

AMC = "HDFC Mutual Fund"


@dataclass(frozen=True)
class Scheme:
    scheme_id: str
    name: str
    category: str
    url: str
    #: Every brief URL is a Direct Growth page. Other plan variants are not in
    #: the corpus and must route to refusal, never to model knowledge.
    #: (PRD §3.2b)
    plan: str = "Direct Growth"
    #: Filename stem under data/raw/. Requested in Phase 1.
    slug: str = ""
    #: The label exactly as it appears in problemstatement.txt.
    brief_label: str = ""


#: The 5 in-scope schemes. problemstatement.txt lines 9-13.
SCHEMES: tuple[Scheme, ...] = (
    Scheme(
        "HDFC_LARGE_CAP", "HDFC Large Cap Fund", "Equity - Large Cap",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        slug="large_cap", brief_label="Large Cap",
    ),
    Scheme(
        "HDFC_FLEXI_CAP", "HDFC Flexi Cap Fund", "Equity - Flexi Cap",
        # Legacy slug: this URL now serves "HDFC Flexi Cap". HDFC Equity Fund
        # was renamed. See ALIASES. (PRD §3.2a)
        "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        slug="flexi_cap", brief_label="Flexi Cap",
    ),
    Scheme(
        "HDFC_ELSS", "HDFC ELSS Tax Saver Fund", "Equity - Tax Saving (ELSS)",
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        slug="elss", brief_label="ELSS",
    ),
    Scheme(
        "HDFC_SMALL_CAP", "HDFC Small Cap Fund", "Equity - Small Cap",
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        slug="small_cap", brief_label="Small Cap",
    ),
    Scheme(
        "HDFC_BAL_ADV", "HDFC Balanced Advantage Fund", "Hybrid - Equity Oriented",
        "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        slug="balanced_advantage", brief_label="Balanced Advantage (Hybrid)",
    ),
)

ALLOWED_SCHEME_IDS: frozenset[str] = frozenset(s.scheme_id for s in SCHEMES)
SCHEME_BY_ID: dict[str, Scheme] = {s.scheme_id: s for s in SCHEMES}
SCHEME_BY_SLUG: dict[str, Scheme] = {s.slug: s for s in SCHEMES}
ALLOWED_URLS: frozenset[str] = frozenset(s.url for s in SCHEMES)

#: Alias -> canonical scheme id, including HDFC's historical names (PRD §3.3).
ALIASES: dict[str, str] = {
    "hdfc large cap": "HDFC_LARGE_CAP", "hdfc top 100": "HDFC_LARGE_CAP",
    "hdfc top 100 fund": "HDFC_LARGE_CAP", "hdfc large cap fund": "HDFC_LARGE_CAP",
    "hdfc flexi cap": "HDFC_FLEXI_CAP", "hdfc equity": "HDFC_FLEXI_CAP",
    "hdfc equity fund": "HDFC_FLEXI_CAP", "hdfc premier equity": "HDFC_FLEXI_CAP",
    "hdfc flexi cap fund": "HDFC_FLEXI_CAP",
    "hdfc elss": "HDFC_ELSS", "hdfc tax saver fund": "HDFC_ELSS",
    "hdfc 80c fund": "HDFC_ELSS", "hdfc elss tax saver fund": "HDFC_ELSS",
    "hdfc small cap": "HDFC_SMALL_CAP", "hdfc small cap fund": "HDFC_SMALL_CAP",
    "hdfc alpha small cap": "HDFC_SMALL_CAP",
    "hdfc balanced advantage": "HDFC_BAL_ADV",
    "hdfc balanced advantage fund": "HDFC_BAL_ADV",
    "hdfc prudence balanced fund": "HDFC_BAL_ADV",
}

#: Educational page linked from every refusal (R6). Never None.
EDUCATIONAL_URL = "https://www.hdfcmf.com/faqs"

# ── Ingestion limits (architecture.md §5) ─────────────────────────────────
#: Below this, an extraction is treated as a failed JS-shell fetch, not a page.
MIN_EXTRACT_CHARS = 200
#: Sanity floor used by the Phase 1 verification table.
EXPECTED_PAGE_CHARS = 5_000

#: Topics the brief names (lines 3 and 15). Used by the Phase 2 topic probe and
#: by Phase 1 verification to prove the corpus can actually answer them.
#:
#: Every pattern must match the fact *itself*, never a near-miss. ``lock_in`` was
#: previously ``lock[-\s]?in|3\s*years|three years`` and passed on the SIP
#: calculator's "3 years" row in all five documents — a green check on a corpus
#: that stated no lock-in at all. Loose alternatives here buy a false pass, which
#: is worse than a red one: it hides the gap instead of reporting it.
BRIEF_TOPIC_PROBES: dict[str, str] = {
    "expense_ratio": r"expense ratio",
    "exit_load": r"exit load",
    "minimum_sip": r"min(?:imum)?\s*sip|minimum investment",
    "lock_in": r"lock[-\s]?in|lock\s+period|\d\s*year\s+lock",
    "riskometer": r"riskometer|level of risk",
    "benchmark": r"benchmark",
    "statement_download": r"capital gains|statement|download",
}

#: Topic probes that are expected to fail against the 5 Groww pages, with the
#: reason. A probe listed here still reports PASS/FAIL honestly; this map only
#: records *why* a red result is expected, so a reader does not mistake a known
#: corpus gap for a pipeline fault. See implementation.md Phase 2 findings.
KNOWN_TOPIC_GAPS: dict[str, str] = {
    "lock_in": "Groww carries the lock-in only as the nav-bar fragment "
               "'ELSS • 3Y Lock-in'. The statutory basis (3-year lock-in for a "
               "section 80C deduction) is not on the page. Needs a Tier-2 "
               "source: HDFC AMC scheme page or AMFI.",
}


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MANIFEST_DIR, INDEX_DIR, DELIVERABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_sources_csv() -> list[dict]:
    """Read config/sources.csv. Path is the authority for scheme_id mapping."""
    if not SOURCES_CSV.exists():
        return []
    with SOURCES_CSV.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_deliverable_sources_csv(path: Path | None = None) -> Path:
    """Deliverable D2: source list of the URLs used."""
    path = path or (DELIVERABLES_DIR / "SOURCES.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "tier": "broker",
            "scheme_id": s.scheme_id,
            "scheme_name": s.name,
            "category": s.category,
            "plan": s.plan,
            "brief_label": s.brief_label,
            "url": s.url,
        }
        for s in SCHEMES
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path
