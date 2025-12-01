#!/usr/bin/env python3
"""
Evaluate TAS-B model on specified IR dataset using sentence-transformers.
This is the easiest way - no conversion needed!
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import ir_datasets
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import time
from hypencoder_cb.inference.neighbor_graph import get_embeddings

def encode_corpus(model: SentenceTransformer, corpus: List[Dict], batch_size: int = 64):
    """Encode all documents in the corpus."""
    texts = [doc.text if hasattr(doc, 'text') else doc['text'] for doc in corpus]
    doc_ids = [doc.doc_id if hasattr(doc, 'doc_id') else doc['doc_id'] for doc in corpus]
    
    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return doc_ids, embeddings


def encode_queries(model: SentenceTransformer, queries: List[Dict], batch_size: int = 64):
    """Encode all queries."""
    query_texts = [q.text if hasattr(q, 'text') else q['text'] for q in queries]
    query_ids = [q.query_id if hasattr(q, 'query_id') else q['query_id'] for q in queries]
    
    print(f"Encoding {len(query_texts)} queries...")
    embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return query_ids, embeddings


def retrieve(query_embeddings, doc_embeddings, doc_ids, top_k: int = 1000):
    """Retrieve top-k documents for each query using cosine similarity."""
    print(f"Computing similarities and retrieving top-{top_k} documents per query...")
    
    start_time = time.time()
    
    results = {}
    for i, query_emb in enumerate(tqdm(query_embeddings)):
        # Compute cosine similarity
        scores = util.cos_sim(query_emb, doc_embeddings)[0].cpu().numpy()
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Store results
        results[i] = [(doc_ids[idx], float(scores[idx])) for idx in top_indices]

    end_time = time.time()
    print(f"TAS-B retrieval time elapsed: {end_time - start_time:.5f} seconds.\n Average per query: {(end_time - start_time)/len(query_embeddings):.5f} seconds.")

    return results


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
    
    # Convert Measure keys to strings for JSON serialization
    return {str(key): value for key, value in aggregated.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/msmarco-distilbert-base-tas-b",
        help="SentenceTransformer model name"
    )
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
        "--encoded_items_path",
        type=str,
        default=None,
        help="Path to pre-encoded data, default None so encoded at runtime"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of documents to retrieve per query"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--metric_names",
        nargs="+",
        default=["nDCG@10", "RR", "nDCG@1000", "R@1000", "AP", "nDCG@5"],
        help="Metrics to calculate"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    # Load dataset
    print(f"Loading dataset: {args.ir_dataset_name}")
    dataset = ir_datasets.load(args.ir_dataset_name)

    # Encode corpus
    corpus = list(dataset.docs_iter())
    doc_ids, doc_embeddings = encode_corpus(model, corpus, args.batch_size)
    
    # Encode queries
    queries = list(dataset.queries_iter())
    query_ids, query_embeddings = encode_queries(model, queries, args.batch_size)
    
    # Retrieve
    results = retrieve(query_embeddings, doc_embeddings, doc_ids, args.top_k)
    
    # Save retrieval results in TREC format
    output_file = output_dir / "retrieved_items.txt"
    print(f"Saving retrieval results to {output_file}")
    with open(output_file, 'w') as f:
        for query_idx, doc_scores in results.items():
            query_id = query_ids[query_idx]
            for rank, (doc_id, score) in enumerate(doc_scores, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} TAS-B\n")
    
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
