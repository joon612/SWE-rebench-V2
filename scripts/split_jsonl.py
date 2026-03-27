#!/usr/bin/env python3
"""
Split a JSONL file into N shards for parallel processing.

Outputs:
  - {output_dir}/shard_{i}.jsonl  for i in 0..N-1
  - A JSON array to stdout: [0, 1, ..., N-1]  (for Gitea/GitHub Actions matrix)

Usage:
    python scripts/split_jsonl.py \
        --jsonl input.jsonl \
        --shards 4 \
        --output-dir /data/shards
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Split JSONL into shards.")
    parser.add_argument("--jsonl", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--shards", type=int, required=True, help="Number of shards.")
    parser.add_argument("--output-dir", required=True, help="Directory for shard files.")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print(f"JSONL file not found: {jsonl_path}", file=sys.stderr)
        return 1

    n = max(1, args.shards)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all lines
    lines: list[str] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                lines.append(raw)

    if not lines:
        print("No records in JSONL file.", file=sys.stderr)
        return 1

    # Clamp shards to actual record count
    n = min(n, len(lines))

    # Round-robin distribute
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, line in enumerate(lines):
        shards[i % n].append(line)

    for shard_idx, shard_lines in enumerate(shards):
        shard_path = output_dir / f"shard_{shard_idx}.jsonl"
        shard_path.write_text("\n".join(shard_lines) + "\n", encoding="utf-8")

    # Output matrix indices as JSON array to stdout
    print(json.dumps(list(range(n))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
