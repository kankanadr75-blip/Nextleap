"""Dump chunks and their embeddings to plain-text files for human review.

    python tests/inspect_embeddings.py

Writes two files next to chunks.jsonl:

* ``chunks_and_embeddings.txt`` - the readable one: every chunk's metadata, its
  full text, an embedding preview, and a live similarity check proving retrieval
  works.
* ``embeddings_full.txt`` - all 384 dimensions per chunk, for anyone who wants
  the raw numbers.
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.ingest.chunk import count_tokens
from src.ingest.embed_store import embed_query, get_collection
from src.query import config

PREVIEW_DIMS = 12
LINE = "=" * 100
THIN = "-" * 100

READABLE = config.PROCESSED_DIR / "chunks_and_embeddings.txt"
FULL = config.PROCESSED_DIR / "embeddings_full.txt"

SAMPLE_QUERIES = [
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver?",
    "How do I download my capital-gains statement?",
]


def fmt_vec(vec: list[float], limit: int | None = None) -> str:
    shown = vec if limit is None else vec[:limit]
    return ", ".join(f"{v:+.4f}" for v in shown)


def main() -> int:
    rows = [
        json.loads(line)
        for line in config.CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    collection = get_collection()
    stored = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = stored["embeddings"]

    # id -> embedding
    by_id: dict[str, list[float]] = {}
    for cid, vec in zip(stored["ids"], vectors):
        by_id[cid] = list(vec)

    out: list[str] = []
    add = out.append

    # ---------------- header ----------------
    add(LINE)
    add("MUTUAL FUND FAQ ASSISTANT - CHUNKS AND EMBEDDINGS (human review dump)")
    add(LINE)
    add("")
    add(f"Generated              : {config.RECORDS_JSON.parent}")
    add(f"Embedding model        : {config.EMBED_MODEL}")
    add(f"Vector dimensions      : {config.EMBED_DIM}")
    add(f"Normalisation          : L2 (normalize_embeddings=True), so cosine")
    add(f"                        distance is meaningful in [0, 2]; lower = closer.")
    add(f"Vector store           : ChromaDB '{config.COLLECTION_NAME}', "
        f"{config.CHROMA_SPACE} space")
    add(f"Collection documents   : {collection.count()}")
    add(f"Chunks in chunks.jsonl : {len(rows)}")
    add(f"Full raw vectors       : {FULL.name}")
    add("")
    add("WHAT IS EMBEDDED")
    add(THIN)
    add("The embedded string is the chunk's context header plus its text:")
    add("    <Scheme> | <Section> | <Source type>. <fact sentence>")
    add("The header is inside the vector on purpose - it is what keeps five")
    add("near-identical scheme pages from being confused with each other.")
    add("")

    # ---------------- counts ----------------
    by_scheme: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for r in rows:
        by_scheme[r["scheme"]] = by_scheme.get(r["scheme"], 0) + 1
        by_type[r["chunk_type"]] = by_type.get(r["chunk_type"], 0) + 1
    add("CHUNKS BY SCHEME")
    add(THIN)
    for slug in [*config.CORPUS_SLUGS, config.GENERAL_SCHEME]:
        label = config.display_name(slug) if slug != config.GENERAL_SCHEME else "(scheme-agnostic)"
        add(f"  {by_scheme.get(slug, 0):>3}  {slug:<26} {label}")
    add("")
    add("CHUNKS BY TYPE")
    add(THIN)
    for kind, n in sorted(by_type.items()):
        add(f"  {n:>3}  {kind}")
    add("")

    # ---------------- similarity check ----------------
    add(LINE)
    add("LIVE SIMILARITY CHECK (proof the vectors retrieve the right chunk)")
    add(LINE)
    add("")
    add("Each query is embedded with the same model and compared against every")
    add("stored chunk. Distance 0.0 = identical direction, higher = further away.")
    add(f"The query path uses a cutoff of {config.DISTANCE_THRESHOLD}.")
    add("")
    for q in SAMPLE_QUERIES:
        qvec = embed_query(q)
        res = collection.query(
            query_embeddings=[qvec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        add(f'QUERY: "{q}"')
        add(THIN)
        for rank, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
        ):
            flag = "  <-- above cutoff" if dist > config.DISTANCE_THRESHOLD else ""
            add(f"  #{rank}  distance={dist:.4f}  scheme={meta['scheme']}  "
                f"attribute={meta['attribute']}{flag}")
            add(f"      {doc[:150]}")
        add("")

    # ---------------- per chunk ----------------
    add(LINE)
    add("EVERY CHUNK, WITH EMBEDDING PREVIEW")
    add(LINE)
    for i, r in enumerate(rows, start=1):
        vec = by_id.get(r["id"], [])
        norm = math.sqrt(sum(v * v for v in vec)) if vec else 0.0
        add("")
        add(f"[{i:>3}/{len(rows)}] {r['id']}")
        add(THIN)
        add(f"  scheme       : {r['scheme']}")
        add(f"  attribute    : {r['attribute']}")
        add(f"  chunk_type   : {r['chunk_type']}")
        add(f"  source_type  : {r['source_type']}")
        add(f"  source_url   : {r['source_url']}")
        add(f"  fetched_at   : {r['fetched_at']}")
        add(f"  tokens       : {count_tokens(r['document'])} "
            f"(MiniLM truncates silently past 256)")
        add("")
        add("  DOCUMENT (this exact string is embedded):")
        for line in _wrap(r["document"], 92):
            add(f"    {line}")
        add("")
        if vec:
            add(f"  EMBEDDING    : {len(vec)} dims, L2 norm = {norm:.4f}")
            add(f"    first {PREVIEW_DIMS}: [{fmt_vec(vec, PREVIEW_DIMS)}]")
            add(f"    ... {len(vec) - PREVIEW_DIMS} more dims in {FULL.name}")
        else:
            add("  EMBEDDING    : NOT FOUND IN STORE")

    add("")
    add(LINE)
    add("END")
    add(LINE)

    READABLE.write_text("\n".join(out) + "\n", encoding="utf-8")

    # ---------------- full vectors ----------------
    full: list[str] = [
        LINE,
        "FULL EMBEDDING VECTORS - one 384-dim vector per chunk",
        f"model: {config.EMBED_MODEL}  |  L2-normalised  |  "
        f"corpus: {config.RECORDS_JSON.parent}",
        LINE,
        "",
    ]
    for r in rows:
        vec = by_id.get(r["id"], [])
        full.append(f"# {r['id']}  scheme={r['scheme']}  attribute={r['attribute']}")
        full.append(f"# text: {r['document'][:110]}")
        for start in range(0, len(vec), 8):
            full.append("  " + fmt_vec(vec[start:start + 8]))
        full.append("")
    FULL.write_text("\n".join(full) + "\n", encoding="utf-8")

    print(f"wrote {READABLE}  ({READABLE.stat().st_size/1024:.1f} KB)")
    print(f"wrote {FULL}  ({FULL.stat().st_size/1024:.1f} KB)")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        if current and sum(len(x) + 1 for x in current) + len(w) > width:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return lines


if __name__ == "__main__":
    sys.exit(main())
