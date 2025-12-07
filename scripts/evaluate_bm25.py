#!/usr/bin/env python3
"""
Evaluate BM25 on specified IR dataset using rank_bm25.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import ir_datasets
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import time

from pyserini.search.lucene import LuceneSearcher
from pyserini.index import IndexReader, IndexCollection


# def tokenize(text: str) -> List[str]:
#     """Simple whitespace tokenization with lowercasing."""
#     return text.lower().split()


# def build_bm25_index(corpus: List, show_progress: bool = True):
#     """Build BM25 index from corpus documents."""
#     doc_ids = []
#     tokenized_corpus = []
    
#     iterator = tqdm(corpus, desc="Tokenizing corpus") if show_progress else corpus
#     for doc in iterator:
#         text = doc.text if hasattr(doc, 'text') else doc['text']
#         doc_id = doc.doc_id if hasattr(doc, 'doc_id') else doc['doc_id']
#         doc_ids.append(doc_id)
#         tokenized_corpus.append(tokenize(text))
    
#     print("Building BM25 index...")
#     bm25 = BM25Okapi(tokenized_corpus)
    
#     return doc_ids, bm25


# def retrieve_bm25(bm25: BM25Okapi, queries: List, doc_ids: List[str], top_k: int = 1000):
#     """Retrieve top-k documents for each query using BM25."""
#     print(f"Retrieving top-{top_k} documents per query...")
    
#     query_ids = []
#     results = {}

#     # Extract relevant query data first (exclude from latency measurement)
#     query_data = []
#     for query in queries:
#         query_text = query.text if hasattr(query, 'text') else query['text']
#         query_id = query.query_id if hasattr(query, 'query_id') else query['query_id']
#         query_data.append({'text': query_text, 'id': query_id})

#     print(f"Retrieving top-{top_k} documents per query...")
    
#     start_time = time.time()
#     # --- Start Core Timed Operations ---
    
#     for i, data in enumerate(query_data): 
        
#         tokenized_query = tokenize(data['text'])
#         scores = bm25.get_scores(tokenized_query)
        
#         # Get top-k indices (ranking)
#         top_indices = np.argsort(scores)[::-1][:top_k]
        
#         # --- End Core Timed Operations ---

#         # Optional: Move result formatting out of this loop if you are aiming for minimum latency
#         results[i] = [(doc_ids[idx], float(scores[idx])) for idx in top_indices] 
    
#     end_time = time.time()
#     print(f"BM25 retrieval time: {end_time - start_time:.4f} seconds. "
#           f"Avg per query: {(end_time - start_time)/len(queries):.4f} seconds.")
    
#     return query_ids, results


def build_anserini_index(corpus_iter, index_dir: str) -> bool:
    """
    Builds a Lucene index using Pyserini for the provided corpus iterator.
    
    NOTE: This is the most time-consuming step and requires a format 
    compatible with Pyserini's 'JsonCollection' (or similar).
    
    For simplicity and demonstration, we write the corpus to a temporary 
    JSON file that Pyserini can index.
    """
    index_path = Path(index_dir)
    if index_path.exists():
        print(f"Index already exists at {index_dir}. Skipping index creation.")
        return True

    # 1. Prepare data in a format Pyserini can read (e.g., JsonCollection)
    # This involves creating a list of dicts, which mimics a JSONL format
    temp_json_dir = Path("temp_anserini_collection")
    temp_json_dir.mkdir(exist_ok=True)
    
    print("Preparing documents for Pyserini indexing...")
    
    # Write documents to a temporary JSONL file
    doc_file = temp_json_dir / "docs.jsonl"
    with open(doc_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(corpus_iter, desc="Serializing Corpus"):
            # Anserini/Pyserini requires 'id' and 'contents' fields
            doc_data = {
                'id': doc.doc_id,
                'contents': doc.text
            }
            f.write(json.dumps(doc_data) + '\n')

    # 2. Build the index
    print(f"Building Lucene index at: {index_dir}...")
    
    # We use IndexCollection tool for building the index
    index_collection = IndexCollection(
        input_args=[str(temp_json_dir)],
        collection='JsonCollection',
        generator='DefaultLuceneDocumentGenerator',
        threads=4, # Use multiple threads for speed
        index=index_dir
    )
    
    # NOTE: The execution of this step depends on Pyserini being properly configured 
    # with the appropriate Java environment.
    index_collection.run()
    
    # Clean up temporary files
    doc_file.unlink()
    temp_json_dir.rmdir()
    
    if Path(index_dir).exists():
        print("Indexing complete.")
        return True
    return False


def retrieve_bm25_anserini(
    queries: List, 
    index_dir: str, 
    top_k: int = 1000
) -> Tuple[List[str], Dict]:
    """
    Performs BM25 retrieval using the Lucene index via Pyserini.
    """
    
    # 1. Initialize the Searcher (Lucene/Anserini equivalent of the Index)
    searcher = LuceneSearcher(index_dir)
    if not searcher:
        raise FileNotFoundError(f"Could not initialize LuceneSearcher from index directory: {index_dir}")
        
    # Set BM25 parameters (using defaults or custom K1/B values)
    # The default parameters in Pyserini/Anserini are typically k1=0.9, b=0.4
    # which may differ slightly from the rank_bm25 defaults but are the standard for IR research.
    # searcher.set_bm25(k1=0.9, b=0.4) # Use this line to set custom parameters

    print(f"Retrieving top-{top_k} documents per query using BM25...")
    
    query_ids = []
    results = {}
    
    # Extract query data for processing
    query_data = []
    for query in queries:
        query_text = query.text if hasattr(query, 'text') else query['text']
        query_id = query.query_id if hasattr(query, 'query_id') else query['query_id']
        query_data.append({'text': query_text, 'id': query_id})
        query_ids.append(query_id)
    
    start_time = time.time()
    
    # --- Start Core Timed Operations ---
    
    for i, data in enumerate(tqdm(query_data, desc="Searching Queries")):
        # Pyserini returns a list of result objects (hits)
        hits = searcher.search(data['text'], k=top_k)
        
        # Format results: map query index -> list of (doc_id, score) pairs
        results[i] = [(hit.docid, float(hit.score)) for hit in hits]
    
    # --- End Core Timed Operations ---
    
    end_time = time.time()
    
    print(f"BM25 retrieval time: {end_time - start_time:.4f} seconds. "
          f"Avg per query: {(end_time - start_time)/len(query_data):.4f} seconds.")
    
    return query_ids, results


def evaluate_with_ir_measures(
    results: Dict,
    query_ids: List[str],
    qrels,
    metric_names: List[str] = None
):
    """Evaluate results using ir-measures."""
    import ir_measures
    from ir_measures import calc_aggregate
    
    if metric_names is None:
        metric_names = ['nDCG@10', 'RR', 'nDCG@1000', 'R@1000', 'AP', 'nDCG@5']
    
    # Convert results to ir_measures format
    run = []
    for query_idx, doc_scores in results.items():
        query_id = query_ids[query_idx]
        for rank, (doc_id, score) in enumerate(doc_scores, 1):
            run.append(ir_measures.ScoredDoc(query_id, doc_id, score))
    
    # Calculate metrics
    metrics = [ir_measures.parse_measure(m) for m in metric_names]
    aggregated = calc_aggregate(metrics, qrels, run)
    
    return {str(key): value for key, value in aggregated.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ir_dataset_name",
        type=str,
        required=True,
        help="IR dataset name (e.g., 'trec-tot/2023/dev' or 'msmarco-passage/trec-dl-2020/judged')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of documents to retrieve per query"
    )
    parser.add_argument(
        "--metric_names",
        nargs="+",
        default=["nDCG@10", "RR", "nDCG@1000", "R@1000", "AP", "nDCG@5"],
        help="Metrics to calculate"
    )
    
    args = parser.parse_args()

    print("\n--- Parsed Arguments (Key-Value) ---")
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {args.ir_dataset_name}")
    dataset = ir_datasets.load(args.ir_dataset_name)
    
    # Build BM25 index
    corpus = list(dataset.docs_iter())
    queries = list(dataset.queries_iter())
    index_dir = "$HOME/hypencoder/encoded_items/BM25/trec-dl-2019"  # hardcoded for now
    # doc_ids, bm25 = build_bm25_index(corpus)

    # Build Anserini index
    success = build_anserini_index(corpus, index_dir)

    # Retrieve
    query_ids, results = retrieve_bm25_anserini(queries, index_dir, args.top_k)
    
    # # Retrieve
    # queries = list(dataset.queries_iter())
    # query_ids, results = retrieve_bm25(bm25, queries, doc_ids, args.top_k)
    
    # Save retrieval results in TREC format
    output_file = output_dir / "retrieved_items.txt"
    print(f"Saving retrieval results to {output_file}")
    with open(output_file, 'w') as f:
        for query_idx, doc_scores in results.items():
            query_id = query_ids[query_idx]
            for rank, (doc_id, score) in enumerate(doc_scores, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} BM25\n")
    
    # Evaluate
    print("Calculating metrics...")
    qrels = dataset.qrels_iter()
    metrics = evaluate_with_ir_measures(results, query_ids, qrels, args.metric_names)
    
    # Save metrics
    metrics_file = metrics_dir / "aggregated_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nMetrics saved to {metrics_file}")
    print("\nResults:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()