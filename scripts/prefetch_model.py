"""Pre-download the embedding model into the build image.

Run by ``render.yaml`` at build time, before the service goes live. The model is
~87 MB, and without this the FIRST request has to fetch it while the user waits
- slow enough to look like a hang to whoever opens the link first. Baking it
into the image trades build time for a fast, predictable cold start.

Reads the model name from ``config.EMBEDDING_MODEL`` rather than repeating the
string. Two copies of a model identifier is a second source of truth, and it
will drift: the tests and the index would keep embedding with the old model
while this script warmed the new one, and the bit-identical rebuild assertion
would fail at runtime instead of here.

Fails loudly on purpose. A build-time error is visible and fixable; the same
download failing at request time is a production timeout with no other clue.
"""
import sys
from pathlib import Path

# The build runs with the repo root as the working directory, but make the
# import work regardless of how this is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EMBEDDING_MODEL  # noqa: E402


def main() -> int:
    print(f"[prefetch] model: {EMBEDDING_MODEL}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"[prefetch] sentence-transformers is not installed: {exc}")
        print("[prefetch] did the pip install in buildCommand run?")
        return 1

    print("[prefetch] downloading weights + tokenizer ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Prove the cache is warm rather than trusting that the constructor did it.
    # SentenceTransformer will silently fall back to a network fetch on a cold
    # cache, so the only way to know the image really carries the model is to
    # encode something and confirm the vector is sane.
    vector = model.encode(["warm the cache"])
    dim = len(vector[0])
    norm = float((sum(float(x) * float(x) for x in vector[0])) ** 0.5)
    print(f"[prefetch] encoded ok: dim={dim} l2_norm={norm:.6f}")

    if dim != 384:
        print(f"[prefetch] expected 384 dimensions, got {dim}")
        return 1
    if abs(norm - 1.0) > 1e-3:
        # The index stores L2-normalised vectors, so an un-normalised warm-up
        # would mean the model in the image is not the model that built it.
        print(f"[prefetch] expected an L2-normalised vector, got norm={norm:.6f}")
        return 1

    print("[prefetch] ok - model is cached in the image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
