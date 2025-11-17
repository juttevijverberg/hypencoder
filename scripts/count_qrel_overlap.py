#!/usr/bin/env python3
"""Count unique query ids in two TREC qrel files and report overlap.

Defaults point to the DL_HARD files in this repo, but you can pass --a and --b.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set


def read_qids_from_qrel(path: Path) -> Set[str]:
    qids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if parts:
                qids.add(parts[0])
    return qids


def main() -> int:
    parser = argparse.ArgumentParser(description="Count unique qids in two qrel files and report overlap")
    parser.add_argument(
        "--a",
        type=Path,
        default=Path("/home/scur1744/hypencoder/data/DL_HARD/original_data/new_judgements-passage.passage-level.qrels"),
        help="First qrel file (default: new_judgements-passage.passage-level.qrels)",
    )
    parser.add_argument(
        "--b",
        type=Path,
        default=Path("/home/scur1744/hypencoder/data/DL_HARD/original_data/dl_hard-passage.qrels"),
        help="Second qrel file (default: dl_hard-passage.qrels)",
    )
    parser.add_argument("--show-common", action="store_true", help="Print the list of common qids")

    args = parser.parse_args()

    if not args.a.exists():
        print(f"Error: file not found: {args.a}")
        return 2
    if not args.b.exists():
        print(f"Error: file not found: {args.b}")
        return 2

    qids_a = read_qids_from_qrel(args.a)
    qids_b = read_qids_from_qrel(args.b)

    common = qids_a & qids_b

    print(f"Unique queries in A ({args.a}): {len(qids_a)}")
    print(f"Unique queries in B ({args.b}): {len(qids_b)}")
    print(f"Queries in both files: {len(common)}")

    if args.show_common:
        for q in sorted(common):
            print(q)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
