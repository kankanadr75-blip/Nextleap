"""Print chunks.jsonl for manual inspection (PRD requires eyeballing the output)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The Windows console defaults to cp1252 and cannot render the en-dash or the
# rupee sign. Force UTF-8 so inspection output is readable.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.query import config

only = sys.argv[1] if len(sys.argv) > 1 else None

rows = [
    json.loads(line)
    for line in config.CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

current = None
for row in rows:
    if only and row["scheme"] != only and only != "general":
        if row["scheme"] != config.GENERAL_SCHEME:
            continue
    if row["scheme"] != current:
        current = row["scheme"]
        label = (
            config.display_name(current) if current != config.GENERAL_SCHEME else "general"
        )
        print()
        print("=" * 100)
        print(f"### {current}  ({label})")
        print("=" * 100)
    print(f"\n[{row['chunk_type']:<9}] {row['attribute']:<20} {row['id']}")
    print(f"  {row['document']}")
    print(f"  src: {row['source_url']}  ({row['source_type']}, fetched {row['fetched_at']})")

print()
print(f"total: {len(rows)} chunks")
