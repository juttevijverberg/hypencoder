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

    base = Path(dest_path)
    base.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        short = name.split("/")[-1].replace("-instructions", "")
        out_dir = base / short
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in subsets:
            try:
                ds = load_dataset(name, sub)
                for split, data in ds.items():
                    data.to_json(out_dir / f"{sub}_{split}.json")
            except Exception:
                pass
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

    q_data = [{"query_id": q.query_id, "text": q.text} for q in ds.queries_iter()]
    with open(base / "queries.json", "w", encoding="utf-8") as f:
        json.dump(q_data, f, ensure_ascii=False, indent=2)

    d_data = [{"doc_id": d.doc_id, "text": d.text} for d in ds.docs_iter()]
    with open(base / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(d_data, f, ensure_ascii=False, indent=2)

    qrels_data = [{"query_id": qrel.query_id, "doc_id": qrel.doc_id, "relevance": qrel.relevance} for qrel in ds.qrels_iter()]
    with open(base / "qrels.json", "w", encoding="utf-8") as f:
        json.dump(qrels_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished saving TOT docs, queries and qrels to {base}")

def download_dl_hard_test(dest_path):
    """
    Save queries and qrels for msmarco-passage/trec-dl-hard to dest_path.
    (Corpus is intentionally skipped due to its very large size.)
    """
    ds = ir_datasets.load("msmarco-passage/trec-dl-hard")
    base = Path(dest_path)
    base.mkdir(parents=True, exist_ok=True)

    # queries -> queries.json
    queries = [{"query_id": q.query_id, "text": q.text} for q in ds.queries_iter()]
    with open(base / "queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    # qrels -> qrels.json
    qrels = [{"query_id": r.query_id, "doc_id": r.doc_id, "relevance": r.relevance} for r in ds.qrels_iter()]
    with open(base / "qrels.json", "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished saving TREC DL Hard queries and qrels to {base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["FollowIR_test", "TOT_test", "FollowIR_train", "TOT_train", "DL_HARD_test"], required=True,
                        help="Select dataset type to download")
    parser.add_argument("--dest_path", required=True,
                        help="Full destination folder (e.g., data/TOT/test)")
    args = parser.parse_args()

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
