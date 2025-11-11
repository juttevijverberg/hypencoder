#!/usr/bin/env python3
"""
Convert a JSONL qrels file (one JSON object per line) into a single JSON
mapping of the expected shape:

{
  "<query_id>": {"<doc_id>": <relevance>, ...},
  ...
}

This script handles common key names such as "query-id", "query_id",
"corpus-id", "corpus_id", "docid", and "score"/"relevance".

Usage:
    python scripts/convert_qrels_jsonl_to_merged.py \
        --input /path/to/qrels.jsonl \
        --output /path/to/qrels_merged.json

Options:
    --binarize        Convert scores > 0 to 1, else 0 (useful when scores are floats)
    --score-key       Custom score key (default: tries common names)
    --query-key       Custom query id key
    --doc-key         Custom doc id key

"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Iterable, Dict, Any


COMMON_QUERY_KEYS = ["query-id", "query_id", "qid", "query", "queryId"]
COMMON_DOC_KEYS = ["corpus-id", "corpus_id", "docid", "doc_id", "doc", "id"]
COMMON_SCORE_KEYS = ["score", "relevance", "rel", "rating"]


def iter_json_objects_from_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON on line {lineno}: {e}")
            yield obj


def extract_fields(obj: Dict[str, Any], query_key: str | None, doc_key: str | None, score_key: str | None):
    # If obj is a dict of the form { qid: { docid: rel, ... } }, handle that
    if isinstance(obj, dict) and len(obj) == 1:
        k0, v0 = next(iter(obj.items()))
        if isinstance(v0, dict):
            # return a generator of (qid, docid, rel)
            for did, rel in v0.items():
                yield str(k0), str(did), rel
            return

    # Otherwise expect a single record representing one qrel
    # Determine keys heuristically
    q = None
    d = None
    r = None

    if query_key is not None and query_key in obj:
        q = obj[query_key]
    else:
        for k in COMMON_QUERY_KEYS:
            if k in obj:
                q = obj[k]
                break

    if doc_key is not None and doc_key in obj:
        d = obj[doc_key]
    else:
        for k in COMMON_DOC_KEYS:
            if k in obj:
                d = obj[k]
                break

    if score_key is not None and score_key in obj:
        r = obj[score_key]
    else:
        for k in COMMON_SCORE_KEYS:
            if k in obj:
                r = obj[k]
                break

    if q is None or d is None:
        # If the object looks like a nested mapping, attempt to handle that
        raise KeyError(f"Could not find query/doc keys in object. Available keys: {list(obj.keys())}")

    yield str(q), str(d), r


def merge_qrels_from_jsonl(in_path: str, query_key: str | None, doc_key: str | None, score_key: str | None, binarize: bool) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(dict)

    for obj in iter_json_objects_from_jsonl(in_path):
        for q, d, r in extract_fields(obj, query_key, doc_key, score_key):
            # Normalize relevance to integer
            if r is None:
                rel = 0
            else:
                try:
                    rel = int(r)
                except Exception:
                    try:
                        rel = int(float(r))
                    except Exception:
                        rel = 1 if float(r) > 0 else 0

            if binarize:
                rel = 1 if rel > 0 else 0

            qrels[str(q)][str(d)] = int(rel)

    return qrels


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL qrels to merged JSON mapping")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output merged JSON file")
    parser.add_argument("--query-key", default=None, help="Explicit query id key (overrides heuristics)")
    parser.add_argument("--doc-key", default=None, help="Explicit doc id key (overrides heuristics)")
    parser.add_argument("--score-key", default=None, help="Explicit score key (overrides heuristics)")
    parser.add_argument("--binarize", action="store_true", help="Binarize relevance (score>0 -> 1)")

    args = parser.parse_args()

    merged = merge_qrels_from_jsonl(
        args.input, args.query_key, args.doc_key, args.score_key, args.binarize
    )

    # Write merged mapping as a single JSON object
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f"Wrote merged qrels to {args.output}")


if __name__ == "__main__":
    main()
