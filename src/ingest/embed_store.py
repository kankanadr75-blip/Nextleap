"""Phase 3 - Embedding and storage.

Embeds the chunks with all-MiniLM-L6-v2 (384-dim, L2-normalised) and upserts
them into a persistent ChromaDB collection using cosine distance.

Ingestion is idempotent: chunk ids are derived from
(source_url, attribute, scheme), so re-running replaces records in place instead
of duplicating them.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings

from src.ingest.chunk import Chunk, build_chunks, count_tokens
from src.query import config


@lru_cache(maxsize=1)
def get_model():  # type: ignore[no-untyped-def]
    """Load the embedding model once.

    [verified] cold load takes ~22 s, so this must be cached: the <3 s median
    latency target assumes the model is resident, not reloaded per query.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def get_client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(client: chromadb.ClientAPI | None = None):  # type: ignore[no-untyped-def]
    """Fetch or create the collection, forcing cosine space.

    ChromaDB 1.x prefers ``configuration={"hnsw": {"space": "cosine"}}`` while
    older releases only accept ``metadata={"hnsw:space": "cosine"}``. Both are
    attempted so the 0.6 distance threshold means the same thing either way.
    """
    client = client or get_client()
    try:
        return client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            configuration={"hnsw": {"space": config.CHROMA_SPACE}},
        )
    except Exception:  # noqa: BLE001 - older chromadb signature
        return client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": config.CHROMA_SPACE},
        )


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk documents (header included) with normalisation on."""
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user query with the identical model and normalisation."""
    vector = get_model().encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vector[0].tolist()


def upsert_chunks(chunks: list[Chunk], *, verbose: bool = True) -> int:
    """Embed and upsert every chunk. Returns the number written.

    The vector is built from ``Chunk.embed_text()`` (document + paraphrase line)
    while the *stored* document stays fact-only, so the synonym keywords improve
    recall without ever reaching the citation or the LLM prompt.
    """
    if not chunks:
        return 0

    collection = get_collection()
    documents = [c.document for c in chunks]

    # The paraphrase line is appended after Phase 2's token cap was asserted, so
    # re-check the real limit here. MiniLM truncates past 256 word-pieces with no
    # error, which would silently drop a chunk's facts.
    MINILM_LIMIT = 256
    for chunk in chunks:
        n = count_tokens(chunk.embed_text())
        if n > MINILM_LIMIT:
            raise ValueError(
                f"embed text too long ({n} tokens > {MINILM_LIMIT}) for "
                f"{chunk.attribute}/{chunk.scheme}: the paraphrase line pushes it "
                f"past MiniLM's silent truncation point"
            )

    embeddings = embed_documents([c.embed_text() for c in chunks])

    for vector in embeddings:
        if len(vector) != config.EMBED_DIM:
            raise ValueError(
                f"expected {config.EMBED_DIM}-dim embeddings, got {len(vector)}"
            )

    collection.upsert(
        ids=[c.id for c in chunks],
        documents=documents,
        metadatas=[c.chroma_metadata() for c in chunks],
        embeddings=embeddings,
    )
    if verbose:
        _print_summary(chunks, collection)
    return len(chunks)


def _print_summary(chunks: list[Chunk], collection: Any) -> None:
    by_scheme: dict[str, int] = {}
    for c in chunks:
        by_scheme[c.scheme] = by_scheme.get(c.scheme, 0) + 1
    print(f"  model       : {config.EMBED_MODEL} ({config.EMBED_DIM}-dim)")
    print(f"  space       : {config.CHROMA_SPACE}")
    print(f"  written     : {len(chunks)}")
    print("  per scheme  :")
    for slug in [*config.CORPUS_SLUGS, config.GENERAL_SCHEME]:
        print(f"      {slug:<26} {by_scheme.get(slug, 0)}")
    print(f"  collection  : {collection.count()} documents in {collection.name}")
    print(f"  chroma dir  : {config.CHROMA_DIR}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: embed and store chunks.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    chunks = build_chunks(verbose=not args.quiet)
    upsert_chunks(chunks, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
