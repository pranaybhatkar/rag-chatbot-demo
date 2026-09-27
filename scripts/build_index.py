"""build_index.py — stage runner for the ingestion pipeline.

    scripts\\build_index.py --stages 1          # data/raw -> documents.jsonl
    scripts\\build_index.py --stages 1,2,3,4    # full rebuild
    scripts\\build_index.py --stages 2,3,4 --recreate-collection

Stage 1 is implemented. Stages 2-4 raise NotImplementedError until their phase
lands, so a half-built pipeline fails with a clear message instead of silently
producing an empty index.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, ensure_dirs
from src.ingest import ingest, report

STAGE_NAMES = {
    1: "Loading (data/raw -> documents.jsonl)",
    2: "Chunking (documents.jsonl -> chunks.jsonl + chunk_preview.log)",
    3: "Embedding (chunks.jsonl -> vectors.npy)",
    4: "Store (vectors.npy -> data/index/chroma)",
}


def run_stage_1() -> None:
    docs, rows = ingest()
    print(report(docs, rows))


def run_stage_2() -> None:
    from src.chunker import chunk_all            # noqa: F401  (Phase 2)
    raise NotImplementedError("Phase 2 not built yet — see implementation.md Phase 2")


def run_stage_3() -> None:
    from src.embedder import embed_chunks       # noqa: F401  (Phase 3)
    raise NotImplementedError("Phase 3 not built yet — see implementation.md Phase 3")


def run_stage_4() -> None:
    from src.store import upsert_collection      # noqa: F401  (Phase 3)
    raise NotImplementedError("Phase 3 store not built yet — see implementation.md Phase 3")


RUNNERS = {1: run_stage_1, 2: run_stage_2, 3: run_stage_3, 4: run_stage_4}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ingestion pipeline stages.")
    ap.add_argument("--stages", default="1",
                    help="comma-separated stage numbers, e.g. 1 or 1,2,3,4")
    args = ap.parse_args()

    try:
        stages = [int(s) for s in args.stages.split(",") if s.strip()]
    except ValueError:
        print("--stages must be comma-separated integers", file=sys.stderr)
        return 2
    for s in stages:
        if s not in RUNNERS:
            print(f"unknown stage {s}; valid: {sorted(RUNNERS)}", file=sys.stderr)
            return 2

    ensure_dirs()
    t0 = time.time()
    for s in stages:
        print(f"\n>>> Stage {s}: {STAGE_NAMES[s]}")
        t = time.time()
        try:
            RUNNERS[s]()
        except NotImplementedError as exc:
            print(f"    {exc}", file=sys.stderr)
            return 3
        except Exception as exc:
            print(f"    FAILED after {time.time() - t:.1f}s: {exc}", file=sys.stderr)
            return 1
        print(f"    done in {time.time() - t:.1f}s")

    out = PROCESSED_DIR / "documents.jsonl"
    print(f"\ntotal {time.time() - t0:.1f}s — {out} "
          f"({sum(1 for _ in out.open(encoding='utf-8')) if out.exists() else 0} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
