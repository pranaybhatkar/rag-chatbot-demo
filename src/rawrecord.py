"""rawrecord.py — the on-disk format for files in ``data/raw/``.

A raw record is a plain ``.txt`` file so the corpus is auditable by eye: every
ingested byte sits in a file a human can open, with its provenance attached.

Layout::

    ====...====================================================================
    HDFC Large Cap Fund
    ====...====================================================================
    [PROVENANCE]
    scheme_id       : HDFC_LARGE_CAP
    ...
    ====...====================================================================

    --- SECTION 1: FROM problemstatement.txt (verbatim) ---
    problemstatement.txt line 9:
    Large Cap: https://...

    --- SECTION 2: SOURCE PAGE TEXT (fetched) ---
    <extracted text>

The loader (:mod:`src.ingest`) parses this back into structured records. The
provenance header is the reason citations are auditable: every answer's URL and
as-of date can be traced to a file and a fetch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

RULE = "=" * 80
SUBRULE = "-" * 80

PROV_START = "[PROVENANCE]"
SEC1_PREFIX = "--- SECTION 1: FROM problemstatement.txt (verbatim) ---"
SEC2_PREFIX = "--- SECTION 2: SOURCE PAGE TEXT (fetched) ---"

_KV = re.compile(r"^([a-z_]+)\s*:\s*(.*)$")


@dataclass
class RawRecord:
    scheme_id: str
    scheme_name: str
    brief_label: str
    category: str
    plan: str
    source_url: str
    source_tier: str
    brief_reference: str
    slug: str
    extracted_at: str = ""
    fetched_at: str = ""
    http_status: str = ""
    page_title: str = ""
    content_chars: str = ""
    as_of_date: str = ""
    brief_text: str = ""
    page_text: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def to_text(self) -> str:
        lines = [
            RULE,
            self.scheme_name,
            RULE,
            PROV_START,
            f"scheme_id       : {self.scheme_id}",
            f"scheme_name     : {self.scheme_name}",
            f"brief_label     : {self.brief_label}",
            f"category        : {self.category}",
            f"plan            : {self.plan}",
            f"source_url      : {self.source_url}",
            f"source_tier     : {self.source_tier}",
            f"brief_reference : {self.brief_reference}",
            f"slug            : {self.slug}",
            f"extracted_at    : {self.extracted_at}",
            f"fetched_at      : {self.fetched_at}",
            f"http_status     : {self.http_status}",
            f"page_title      : {self.page_title}",
            f"content_chars   : {self.content_chars}",
            f"as_of_date      : {self.as_of_date}",
        ]
        for k, v in self.extra.items():
            lines.append(f"{k:<15} : {v}")
        lines += [
            RULE,
            "",
            SEC1_PREFIX,
            self.brief_text.strip(),
            "",
            SEC2_PREFIX,
            self.page_text.strip(),
            "",
        ]
        return "\n".join(lines)


def write_record(path: Path, rec: RawRecord) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rec.to_text(), encoding="utf-8")
    return path


def parse_record(path: Path) -> RawRecord:
    """Parse a raw record file back into a :class:`RawRecord`.

    Raises ValueError on a structurally invalid file — a malformed record is a
    bug, not something to skip quietly.
    """
    text = path.read_text(encoding="utf-8")
    if PROV_START not in text:
        raise ValueError(f"{path.name}: missing {PROV_START}")

    head, _, rest = text.partition(PROV_START)
    prov_block, _, body = rest.partition(RULE)

    name = ""
    for line in head.splitlines():
        s = line.strip()
        if s and not s.startswith("="):
            name = s
            break

    kv: dict[str, str] = {}
    for line in prov_block.splitlines():
        m = _KV.match(line.strip())
        if m:
            kv[m[1]] = m[2].strip()

    brief_text, page_text = "", ""
    if SEC1_PREFIX in body:
        _, _, tail = body.partition(SEC1_PREFIX)
        if SEC2_PREFIX in tail:
            brief_text, _, page_text = tail.partition(SEC2_PREFIX)
        else:
            brief_text = tail
    else:
        page_text = body

    required = ("scheme_id", "scheme_name", "source_url", "brief_label")
    missing = [k for k in required if not kv.get(k)]
    if missing:
        raise ValueError(f"{path.name}: missing provenance keys {missing}")

    return RawRecord(
        scheme_id=kv["scheme_id"],
        scheme_name=kv["scheme_name"],
        brief_label=kv["brief_label"],
        category=kv.get("category", ""),
        plan=kv.get("plan", ""),
        source_url=kv["source_url"],
        source_tier=kv.get("source_tier", ""),
        brief_reference=kv.get("brief_reference", ""),
        slug=kv.get("slug", path.stem),
        extracted_at=kv.get("extracted_at", ""),
        fetched_at=kv.get("fetched_at", ""),
        http_status=kv.get("http_status", ""),
        page_title=kv.get("page_title", ""),
        content_chars=kv.get("content_chars", ""),
        as_of_date=kv.get("as_of_date", ""),
        brief_text=brief_text.strip(),
        page_text=page_text.strip(),
    )
