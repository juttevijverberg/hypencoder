from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set

COMMON_QUERY_KEYS = ["query-id", "query_id", "qid", "query", "queryId"]

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON on line {lineno} of {path}: {e}")


def extract_query_id(obj: Any) -> str | None:
    """Heuristically extract a query id from a JSON object.

    Returns the query id as a string or None if not found.
    Handles objects of the form {qid: {docid: rel, ...}} by returning the top-level key.
    """
    if isinstance(obj, dict) and len(obj) == 1:
        k0, v0 = next(iter(obj.items()))
        if isinstance(v0, dict):
            return str(k0)

    if isinstance(obj, dict):
        for k in COMMON_QUERY_KEYS:
            if k in obj:
                return str(obj[k])

    return None


def load_query_set(path: Path) -> Set[str]:
    """Load all query ids from a qrels file (merged JSON or JSONL).
    """
    # handle TREC qrel files (.qrel/.qrels)
    if path.suffix.lower() in (".qrel", ".qrels"):
        qset = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) >= 1:
                    qset.add(parts[0])
        return qset
    # First try to load as a single JSON document
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Fallback to JSONL
        qset = set()
        for obj in iter_jsonl(path):
            q = extract_query_id(obj)
            if q is not None:
                qset.add(q)
        return qset

    # If loaded JSON is a dict that maps qid -> {docid: rel}
    if isinstance(data, dict):
        # assume keys are qids
        return set(map(str, data.keys()))

    # If it's a list of objects, extract query ids from each
    qset = set()
    if isinstance(data, list):
        for obj in data:
            q = extract_query_id(obj)
            if q is not None:
                qset.add(q)
    return qset


def process_a(path_a: Path, queries_to_remove: Set[str]) -> Any:
    """Read file A, remove any query in queries_to_remove, and return the new object.

    The return value is either a dict (merged mapping) or an iterable of objects (for JSONL or list).
    """
    # If A is a TREC qrel file, stream and filter lines
    if path_a.suffix.lower() in (".qrel", ".qrels"):
        def gen_qrel():
            with path_a.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.rstrip("\n")
                    if not s or s.lstrip().startswith("#"):
                        # preserve blank/comment lines
                        yield s
                        continue
                    parts = s.split()
                    if len(parts) >= 1:
                        q = parts[0]
                        if q in queries_to_remove:
                            continue
                    yield s

        return gen_qrel()

    # Try to load as single JSON
    try:
        with path_a.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # JSONL path: stream and yield objects not in queries_to_remove
        def gen():
            for obj in iter_jsonl(path_a):
                q = extract_query_id(obj)
                if q is None or q not in queries_to_remove:
                    yield obj

        return gen()

    # If merged mapping (dict) remove keys
    if isinstance(data, dict):
        # queries_to_remove are strings (from TREC qrels). Data dict keys may be ints or strings;
        # delete any key whose stringified form appears in queries_to_remove.
        str_qs = set(map(str, queries_to_remove))
        keys_to_delete = [k for k in list(data.keys()) if str(k) in str_qs]
        for k in keys_to_delete:
            del data[k]
        return data

    # If list, filter
    if isinstance(data, list):
        return [obj for obj in data if (extract_query_id(obj) is None or extract_query_id(obj) not in queries_to_remove)]

    # Unknown shape: return as-is
    return data


def write_output(out_path: Path, content: Any, reference_format: str):
    """Write content to out_path.

    reference_format is one of: 'json', 'jsonl', 'qrel'.
    """
    if reference_format == "qrel":
        # content is iterable of lines (strings)
        with out_path.open("w", encoding="utf-8") as f:
            for line in content:
                f.write(line + "\n")
    elif reference_format == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for obj in content:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)


def guess_is_jsonl(path: Path) -> bool:
    # If the file is not valid JSON as a whole, treat it as JSONL
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return False
    except Exception:
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove queries present in to_remove_path from qrels_path (and optionally from a queries JSON file)"
    )
    parser.add_argument("--qrels_path", required=True, help="Path to qrels JSON file (merged JSON mapping or JSONL).")
    parser.add_argument("--to_remove_path", required=True, help="Path to TREC .qrel/.qrels file listing query ids to remove from the qrels and queries files")
    parser.add_argument("--out", help="Output path for the qrels result; if omitted, qrels_path will be overwritten")
    parser.add_argument(
        "--queries_path",
        help="Optional: path to a JSON file containing queries (list, mapping, or list-of-objects). Queries present in to_remove_path will be removed from this file.",
    )
    parser.add_argument(
        "--queries_out",
        help="Optional: output path for the filtered queries JSON. If omitted and --queries_path is provided, queries_path will be overwritten.",
    )

    args = parser.parse_args()
    path_a = Path(args.qrels_path)
    path_b = Path(args.to_remove_path)
    out_path = Path(args.out) if args.out else path_a
    queries_path = Path(args.queries_path) if args.queries_path else None
    queries_out = Path(args.queries_out) if args.queries_out else (queries_path if queries_path is not None else None)

    if not path_a.exists():
        raise SystemExit(f"File A not found: {path_a}")
    if not path_b.exists():
        raise SystemExit(f"File B not found: {path_b}")

    # Expect to_remove_path to be a TREC qrels file ('.qrel' / '.qrels').
    if path_b.suffix.lower() not in (".qrel", ".qrels"):
        print(f"Warning: --to_remove_path does not look like a TREC qrels file (expected .qrel/.qrels): {path_b}")

    print(f"Loading queries to remove from (TREC qrels): {path_b}")
    qset = load_query_set(path_b)
    print(f"Found {len(qset)} unique queries to remove ({path_b})")

    print(f"Loading queries from qrels JSON: {path_a}")
    qset_a = load_query_set(path_a)
    print(f"Found {len(qset_a)} unique queries in qrels JSON ({path_a})")

    print(f"Processing qrels: {path_a}")
    # determine A format
    if path_a.suffix.lower() in (".qrel", ".qrels"):
        ref_format = "qrel"
    else:
        ref_format = "jsonl" if guess_is_jsonl(path_a) else "json"

    new_content = process_a(path_a, qset)

    # If qrel, materialize generator to count remaining lines/queries
    if ref_format == "qrel":
        new_content = list(new_content)

    remaining = None
    try:
        # If new_content is a dict mapping qid->..., compute remaining count
        if isinstance(new_content, dict):
            remaining = len(new_content)
        elif hasattr(new_content, "__len__"):
            remaining = len(new_content)
    except Exception:
        remaining = None

    print(f"Writing output to: {out_path}")
    # ensure output directory exists
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    if remaining is not None:
        print(f"Queries before: {len(qset_a)}, removed: {len(qset & qset_a)}, remaining: {remaining}")
    else:
        print(f"Queries before: {len(qset_a)}, removed: {len(qset & qset_a)}")
    # Write using the detected format
    write_output(out_path, new_content, reference_format=ref_format)

    # Optionally remove queries from a separate queries JSON
    if queries_path is not None:
        if not queries_path.exists():
            print(f"Warning: queries_path not found: {queries_path}; skipping")
        else:
            print(f"Loading queries JSON: {queries_path}")
            try:
                with queries_path.open("r", encoding="utf-8") as f:
                    queries_data = json.load(f)
            except Exception as e:
                print(f"Failed to load queries JSON: {e}; skipping")
                queries_data = None

            if queries_data is not None:
                before = None
                after = None
                removed_qs = 0
                # dict mapping qid -> ...
                if isinstance(queries_data, dict):
                    before = len(queries_data)
                    str_qs = set(map(str, qset))
                    keys_to_delete = [k for k in list(queries_data.keys()) if str(k) in str_qs]
                    for k in keys_to_delete:
                        del queries_data[k]
                    after = len(queries_data)
                    removed_qs = before - after
                elif isinstance(queries_data, list):
                    before = len(queries_data)
                    # list of strings?
                    if all(isinstance(x, str) for x in queries_data):
                        filtered = [x for x in queries_data if x not in qset]
                        queries_data = filtered
                    else:
                        filtered = []
                        for obj in queries_data:
                            q = extract_query_id(obj)
                            if q is None or q not in qset:
                                filtered.append(obj)
                        queries_data = filtered
                    after = len(queries_data)
                    removed_qs = before - after
                else:
                    print(f"Unsupported queries JSON shape: {type(queries_data)}; skipping")
                    queries_data = None

                if queries_data is not None:
                    out_queries_path = queries_out
                    if out_queries_path is None:
                        out_queries_path = queries_path
                    # ensure directory exists
                    try:
                        out_queries_path.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    with out_queries_path.open("w", encoding="utf-8") as f:
                        json.dump(queries_data, f, ensure_ascii=False, indent=2)
                    print(f"Filtered queries written to: {out_queries_path} (before={before}, removed={removed_qs}, after={after})")

    print("Done")


if __name__ == "__main__":
    main()
