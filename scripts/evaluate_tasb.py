#!/usr/bin/env python3
"""
Evaluate TAS-B model on specified IR dataset using sentence-transformers or BE-base.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import ir_datasets
import numpy as np
from tqdm import tqdm
import time
import faiss
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer

def encode_corpus(
    corpus: List[Dict],
    batch_size: int,
    model: Optional[SentenceTransformer] = None
):
    """Encode all documents in the corpus with the selected backend."""
    texts = [doc.text if hasattr(doc, "text") else doc["text"] for doc in corpus]
    doc_ids = [doc.doc_id if hasattr(doc, "doc_id") else doc["doc_id"] for doc in corpus]

    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    return doc_ids, embeddings


def encode_queries(
    queries: List[Dict],
    batch_size: int,
    model: Optional[SentenceTransformer] = None,
):
    """Encode all queries with the selected backend."""
    query_texts = [q.text if hasattr(q, "text") else q["text"] for q in queries]
    query_ids = [q.query_id if hasattr(q, "query_id") else q["query_id"] for q in queries]

    print(f"Encoding {len(query_texts)} queries...")
    embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return query_ids, embeddings

def load_corpus(input_folder: str):
    """
    Load pre-encoded documents (IDs, embeddings, and texts) from a .npz file.
    """
    print(f"Loading pre-encoded documents from: {input_folder}")
    input_file = Path(input_folder) / "encoded_corpus.npz"

    data = np.load(input_file, allow_pickle=True)
    doc_ids = data['ids'].tolist()
    doc_embeddings = data['embeddings']
    
    print(f"Loaded {len(doc_ids)} documents.")
    return doc_ids, doc_embeddings

def load_queries(input_path: str):
    """
    Load pre-encoded queries (IDs and embeddings) from a .npz file.
    """
    print(f"Loading pre-encoded queries from: {input_path}")
    input_file = Path(input_path)
    
    data = np.load(input_file, allow_pickle=True)
    query_ids = data['ids'].tolist()
    query_embeddings = data['embeddings']
    
    print(f"Loaded {len(query_ids)} queries.")
    return query_ids, query_embeddings

def retrieve(query_embeddings, doc_embeddings, doc_ids, top_k: int = 1000):
    """Retrieve top-k documents for each query using cosine similarity."""
    print(f"Computing similarities and retrieving top-{top_k} documents per query...")
    
    results = {}
    for i, query_emb in enumerate(tqdm(query_embeddings)):
        # Compute cosine similarity
        scores = util.cos_sim(query_emb, doc_embeddings)[0].cpu().numpy()
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Store results
        results[i] = [(doc_ids[idx], float(scores[idx])) for idx in top_indices]

    return results

def retrieve_faiss(query_embeddings, doc_embeddings, doc_ids, top_k: int = 1000):
    """
    Retrieve top-k documents for each query using Faiss (cosine similarity).
    - query_embeddings: (nq, d) numpy array or torch tensor
    - doc_embeddings:   (nd, d) numpy array or torch tensor
    - doc_ids:          (nd,) array-like ids (can be str)
    Returns: dict mapping query_idx -> list[(doc_id, score), ...]
    """
    print(f"Computing similarities with Faiss and retrieving top-{top_k} documents per query...")
    queries = to_numpy_float32(query_embeddings)
    docs = to_numpy_float32(doc_embeddings)
    original_doc_ids = np.asarray(doc_ids)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(docs)
    faiss.normalize_L2(queries)

    d = docs.shape[1]

    # Start time here 
    start_time = time.time()
    
    # Build index: exact inner-product search
    index = faiss.IndexFlatIP(d)          
    index.add(docs)
    t1 = time.time()
    print(f"Faiss index build time elapsed: {(t1 - start_time)*1000:.3f} miliseconds.")

    # Search (I contains the internal indices 0 to N-1)
    # start_time = time.time()
    D, I = index.search(queries, top_k)
    end_time = time.time()

    print(f"TAS-B Faiss retrieval time elapsed: {(end_time - start_time)*1000:.3f} miliseconds."
          f" Avg per query: {((end_time - start_time)/len(queries))*1000:.3f} miliseconds.")

    # Build results dict and map the internal indices back to original IDs
    results = {}
    for q_idx in range(queries.shape[0]):
        pairs = []
        
        internal_indices = I[q_idx] 
        row_scores = D[q_idx].tolist()
        original_ids = original_doc_ids[internal_indices] 
        
        for doc_id, score in zip(original_ids, row_scores):
            pairs.append((str(doc_id), float(score))) 
            
        results[q_idx] = pairs

    return results

def to_numpy_float32(x):
    """Convert tensor or array to numpy float32."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

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
            run.append(ir_measures.ScoredDoc(query_id, str(doc_id), score))
    
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
        help="Model identifier"
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
    parser.add_argument(
        "--retrieve_faiss",
        action="store_true",
        help="Use the retrieval with Faiss."
    )
    parser.add_argument(
        "--use_encoded_docs",
        action="store_true",
        help="Use the encoded documents from encoded_items_path argument. If False, encode documents at runtime."
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
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    # Load dataset
    print(f"Loading dataset: {args.ir_dataset_name}")
    dataset = ir_datasets.load(args.ir_dataset_name)

    # Encode corpus
    if args.use_encoded_docs:
        doc_ids, doc_embeddings = load_corpus(args.encoded_items_path)
    else:
        corpus = list(dataset.docs_iter())
        doc_ids, doc_embeddings = encode_corpus(
            corpus,
            args.batch_size,
            model=model
        )
    
    # Encode queries
    queries = list(dataset.queries_iter())
    query_ids, query_embeddings = encode_queries(
        queries,
        args.batch_size,
        model=model
    )
    
    # Retrieve
    if args.retrieve_faiss:
        results = retrieve_faiss(query_embeddings, doc_embeddings, doc_ids, args.top_k)
    else:
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
