import os
import json
import argparse
from pathlib import Path
import ir_datasets
from datasets import load_dataset
import requests
import zipfile
import shutil
import tempfile
import io


COMMON_QUERY_KEYS = ["query-id", "query_id", "qid", "query", "queryId"]
COMMON_DOC_KEYS = ["corpus-id", "corpus_id", "docid", "doc_id", "doc", "id"]
COMMON_SCORE_KEYS = ["score", "relevance", "rel", "rating"]


def _convert_qrels_list_to_merged(items):
    """Convert a list of qrel records into a merged mapping {qid: {docid: rel}}.

    Each item can use a variety of key names; we try common ones.
    """
    merged = {}
    for obj in items:
        # obj might be a Dataset row (dict-like)
        q = None
        d = None
        r = None
        if not isinstance(obj, dict):
            # try to convert to dict
            try:
                obj = dict(obj)
            except Exception:
                continue

        for k in COMMON_QUERY_KEYS:
            if k in obj:
                q = obj[k]
                break

        for k in COMMON_DOC_KEYS:
            if k in obj:
                d = obj[k]
                break

        for k in COMMON_SCORE_KEYS:
            if k in obj:
                r = obj[k]
                break

        if q is None or d is None:
            # skip entries we can't parse
            continue

        q = str(q)
        d = str(d)
        try:
            rel = int(r)
        except Exception:
            try:
                rel = int(float(r))
            except Exception:
                try:
                    rel = 1 if float(r) > 0 else 0
                except Exception:
                    rel = 0

        merged.setdefault(q, {})[d] = int(rel)

    return merged

def download_followir_train(dest_path):
    ds = load_dataset("samaya-ai/msmarco-w-instructions")
    base = Path(dest_path)
    base.mkdir(parents=True, exist_ok=True)
    
    for split, data in ds.items():
        data.to_json(base / f"{split}.json")
    print(f"✅ Finished downloading MSMARCO instructions dataset to {base}")

def download_followir_test(dest_path):
    datasets = [
        "jhu-clsp/robust04-instructions",
        "jhu-clsp/news21-instructions",
        "jhu-clsp/core17-instructions"
    ]
    subsets = ["corpus", "queries", "qrels_changed", "qrels_og", "top_ranked"]
    jsonl_subsets = {"corpus", "queries", "top_ranked"}

    base = Path(dest_path)
    base.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        short = name.split("/")[-1].replace("-instructions", "")
        out_dir = base / short
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in subsets:
            ds = load_dataset(name, sub)
            split = list(ds.keys())[0]
            data = ds[split]
            if sub in jsonl_subsets:
                data.to_json(out_dir / f"{sub}.jsonl", lines=True)
            else:
                items = [row for row in data]
                out_file = out_dir / f"{sub}.json"
                # If this subset is a qrels file, convert list-of-records into
                # the merged mapping {qid: {docid: rel}} which the evaluation
                # code expects. Otherwise, write the list as before.
                if sub.startswith("qrels") or "qrel" in sub:
                    merged = _convert_qrels_list_to_merged(items)
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(merged, f, indent=2, ensure_ascii=False)
                else:
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"✅ Finished downloading FollowIR datasets to {base}")

def download_tot_train(dest_path):
    """Download and extract all files from the 'data_release' folder of the TOMT GitHub repo."""
    base = Path(dest_path)
    base.mkdir(parents=True, exist_ok=True)

    def fetch_dir(api_url, out_dir):
        r = requests.get(api_url, timeout=60)
        r.raise_for_status()
        for item in r.json():
            if item["type"] == "file" and item.get("download_url"):
                url, name = item["download_url"], item["name"]
                resp = requests.get(url, stream=True, timeout=300)
                resp.raise_for_status()
                if name.lower().endswith(".zip"):
                    with io.BytesIO(resp.content) as buf, zipfile.ZipFile(buf) as zf:
                        zf.extractall(out_dir)
                else:
                    (out_dir / name).write_bytes(resp.content)
            elif item["type"] == "dir":
                subdir = out_dir / item["name"]
                subdir.mkdir(parents=True, exist_ok=True)
                fetch_dir(item["url"], subdir)

    api_url = "https://api.github.com/repos/samarthbhargav/tomt-data/contents/data_release?ref=main"
    fetch_dir(api_url, base)
    print(f"✅ Finished downloading and extracting TOMT data to {base}")

def download_tot_test(dest_path):
    ds = ir_datasets.load("trec-tot/2023/dev")
    base = Path(dest_path)
    base.mkdir(parents=True, exist_ok=True)

    with open(base / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in ds.queries_iter():
            obj = {"query_id": str(q.query_id), "text": q.text}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(base / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in ds.docs_iter():
            obj = {"doc_id": str(d.doc_id), "text": d.text}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    qrels_data = [{"query_id": qrel.query_id, "doc_id": qrel.doc_id, "relevance": qrel.relevance} for qrel in ds.qrels_iter()]
    with open(base / "qrels.json", "w", encoding="utf-8") as f:
        merged = _convert_qrels_list_to_merged(qrels_data)
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished saving TOT docs, queries and qrels to {base}")

def download_dl_hard_test(dest_path):
    """
    Save queries and qrels for msmarco-passage/trec-dl-hard to dest_path.
    (Corpus is intentionally skipped due to its very large size.)
    """
    ds = ir_datasets.load("msmarco-passage/trec-dl-hard")
    base = Path(dest_path)
    base.mkdir(parents=True, exist_ok=True)

    qrels = [
        {"query_id": r.query_id, "doc_id": r.doc_id, "relevance": r.relevance}
        for r in ds.qrels_iter()
    ]

    url = (
        "https://raw.githubusercontent.com/grill-lab/DL-Hard/"
        "2be6435c2b1f8131dfa23f3c0dee72f9dd47d849/"
        "annotations/new_judgements/new_judgements-passage.passage-level.qrels"
    )
    resp = requests.get(url)
    resp.raise_for_status()

    new_qrels_set = set()
    for line in resp.text.splitlines():
        q, _, d, _ = line.split()
        new_qrels_set.add((q, d))

    qrels_half = [
        r
        for r in qrels
        if (str(r["query_id"]), str(r["doc_id"])) not in new_qrels_set
    ]

    with open(base / "qrels.json", "w", encoding="utf-8") as f:
        merged = _convert_qrels_list_to_merged(qrels_half)
        json.dump(merged, f, ensure_ascii=False, indent=2)

    keep_query_ids = {r["query_id"] for r in qrels_half}

    with open(base / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in ds.queries_iter():
            if q.query_id in keep_query_ids:
                obj = {"query_id": q.query_id, "text": q.text}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Finished saving TREC DL Hard queries and qrels to {base}")

def download_msmarco_qrels(dest_path):
    for path_name in [
        "msmarco-passage/trec-dl-2019/judged",
        "msmarco-passage/trec-dl-2020/judged",
        "msmarco-passage/dev/small"
    ]:
        ds = ir_datasets.load(path_name)
        
        # Create subdirectory for each dataset
        dataset_name = path_name.replace("/", "_").replace("-", "_")
        base = Path(dest_path) / dataset_name
        base.mkdir(parents=True, exist_ok=True)

        qrels_data = [{"query_id": qrel.query_id, "doc_id": qrel.doc_id, "relevance": qrel.relevance} for qrel in ds.qrels_iter()]
        with open(base / "qrels.json", "w", encoding="utf-8") as f:
            merged = _convert_qrels_list_to_merged(qrels_data)
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"✅ Finished saving {path_name} qrels to {base}")

def get_train_data(dest_path):
    print("📥 Loading dataset from HuggingFace...", flush=True)
    ds = load_dataset("jfkback/hypencoder-msmarco-training-dataset")
    train_data = ds["train"]
    print(f"✅ Loaded dataset with {len(train_data):,} total examples", flush=True)
    
    print(f"💾 Writing {len(train_data):,} examples to JSON (fast manual method)...", flush=True)
    with open(dest_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(train_data):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1:,}/{len(train_data):,} examples...", end='\r', flush=True)
    print(f"\n✅ Saved {len(train_data):,} examples to {dest_path}", flush=True)

def get_subset_train_data(dest_path, max_samples=None):
    print("📥 Loading dataset from HuggingFace...", flush=True)
    ds = load_dataset("jfkback/hypencoder-msmarco-training-dataset")
    train_data = ds["train"]
    print(f"✅ Loaded dataset with {len(train_data):,} total examples", flush=True)
    
    if max_samples is not None:
        # Shuffle first to get random samples, then select
        print(f"🔀 Shuffling and selecting {max_samples:,} samples...", flush=True)
        train_data = train_data.shuffle(seed=42).select(range(min(max_samples, len(train_data))))
        print(f"📊 Randomly selected {len(train_data):,} samples", flush=True)
    
    # Fast manual JSON writing (avoids slow HF to_json)
    print(f"💾 Writing {len(train_data):,} examples to {dest_path}...", flush=True)
    with open(dest_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(train_data):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
            if (i + 1) % 1000 == 0:
                print(f"  Written {i + 1:,}/{len(train_data):,} examples...", end='\r', flush=True)
    print(f"\n✅ Saved {len(train_data):,} examples to {dest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["FollowIR_test", "TOT_test", "FollowIR_train", "TOT_train", "DL_HARD_test", "MSMARCO_qrels", "train_data", "subset_train_data"], required=True,
                        help="Select dataset type to download")
    parser.add_argument("--dest_path", required=True,
                        help="Full destination folder (e.g., data/TOT/test)")
    args = parser.parse_args()

    path = Path(args.dest_path)
    path.mkdir(parents=True, exist_ok=True)

    if args.data == "FollowIR_test":
        download_followir_test(args.dest_path)
    elif args.data == "TOT_test":
        download_tot_test(args.dest_path)
    elif args.data == "FollowIR_train":
        download_followir_train(args.dest_path)
    elif args.data == "TOT_train":
        download_tot_train(args.dest_path)
    elif args.data == "DL_HARD_test":
        download_dl_hard_test(args.dest_path)
    elif args.data == "MSMARCO_qrels":
        download_msmarco_qrels(args.dest_path)
    elif args.data == "train_data":
        get_train_data(args.dest_path)
    elif args.data == "subset_train_data":
        get_subset_train_data(args.dest_path, max_samples=50000)

