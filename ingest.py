"""One-command ingestion pipeline.

    python ingest.py                # Phase 1 (loading) - the only stage built so far
    python ingest.py --force        # ignore the 24h raw-HTML cache
    python ingest.py --fetch-only   # explicit alias for the same thing

Phases 2 (clean + chunk), 3 (embed + store) and 4+ plug in below as they land.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # --fetch-only is accepted explicitly so the command in implementation.md
    # Phase 1 works verbatim.
    argv = [a for a in argv if a != "--fetch-only"]

    from src.ingest.load import main as load_main

    print("[1/3] load     : fetching and extracting scheme facts")
    rc = load_main(argv)
    if rc != 0:
        print("\nLoading did not complete cleanly; stopping before later stages.")
        return rc

    print()
    print("[2/3] chunk    : cleaning records and building chunks")
    from src.ingest.chunk import main as chunk_main

    chunk_rc = chunk_main(argv)
    if chunk_rc != 0:
        return chunk_rc

    print("\n[3/3] embed    : embedding chunks and writing to ChromaDB")
    from src.ingest.embed_store import main as embed_main

    return embed_main(argv)


if __name__ == "__main__":
    sys.exit(main())
