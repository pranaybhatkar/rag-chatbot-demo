"""Validate .streamlit/config.toml against the installed Streamlit.

Why this exists
---------------
`.streamlit/config.toml` carried `runner.timeout = 240` for several deploys.
That option does not exist. Streamlit logged, on every process start:

    "runner.timeout" is not a valid config option. If you previously had this
    config option set, it may have been removed.

An unknown key is logged and **discarded, not enforced**, so the key looked like
it was raising a timeout and was not. Nothing failed, which is exactly why it
survived: the file was never checked against the version actually installed.

The options a given Streamlit accepts are not a stable, hand-maintained list.
They come from `streamlit.runtime.config.AvailableSettings`, which is rebuilt
per version -- and the set of `runner.*` keys has already been emptied once in
the 1.x line. So this asks the installed package rather than a pinned list.

Run:  python scripts/check_config.py
Exits non-zero if any key is unknown, so CI or a pre-deploy hook can gate on it.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".streamlit" / "config.toml"

try:
    from streamlit.config import get_config_options
except ImportError as exc:  # pragma: no cover - depends on Streamlit internals moving
    print(f"check_config: cannot introspect Streamlit ({exc}).", file=sys.stderr)
    print("  Skipped rather than guessed - a wrong answer here would be", file=sys.stderr)
    print("  worse than no answer.", file=sys.stderr)
    sys.exit(0)

VALID = set(get_config_options())
SOURCE = "streamlit.config.get_config_options()"

# Keys this file must never carry, with the reason. These are the ones that
# survive review because they look plausible.
FORBIDDEN: dict[str, str] = {
    "server.port": "collides with the start command's $PORT; see the note above",
    "runner.timeout": "removed from Streamlit; logged and discarded, never enforced",
}


def main() -> int:
    if not CONFIG.exists():
        print(f"check_config: {CONFIG} not found")
        return 0

    try:
        data = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        # Worth failing loudly on: a malformed file stops the server before it
        # binds a port, which surfaces only as a 502 with an empty log.
        print(f"check_config: {CONFIG} is not valid TOML\n  {exc}", file=sys.stderr)
        return 1

    print(f"check_config: source = {SOURCE}")
    print(f"check_config: {len(VALID)} valid option(s) in this Streamlit")
    print()

    unknown: list[str] = []
    forbidden: list[str] = []
    known: list[str] = []

    for section, body in data.items():
        if not isinstance(body, dict):
            # A bare top-level scalar is not a Streamlit config shape.
            unknown.append(f"{section} (expected a table of options)")
            continue
        for key, value in body.items():
            dotted = f"{section}.{key}"
            if dotted in FORBIDDEN:
                forbidden.append(f"{dotted} = {value!r}  -- {FORBIDDEN[dotted]}")
            elif dotted in VALID:
                known.append(f"{dotted} = {value!r}")
            else:
                unknown.append(f"{dotted} = {value!r}")

    for line in known:
        print(f"  [ OK ] {line}")

    if forbidden:
        print()
        for line in forbidden:
            print(f"  [FAIL] {line}", file=sys.stderr)

    if unknown:
        print()
        for line in unknown:
            print(f"  [FAIL] {line}", file=sys.stderr)

    if forbidden or unknown:
        if forbidden:
            print()
            print("A key on the forbidden list is either invalid or actively", file=sys.stderr)
            print("harmful. See the reason printed above it.", file=sys.stderr)
        if unknown:
            print()
            print(f"{len(unknown)} unknown key(s). An unrecognised option is logged", file=sys.stderr)
            print("once per process start and then ignored - it does not fail the", file=sys.stderr)
            print("deploy, and it does not do whatever it appears to do.", file=sys.stderr)
        return 1

    print()
    print("All keys are valid for the installed Streamlit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
