"""Utility functions shared by BE-base encoding and retrieval scripts."""
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
import torch
from docarray import DocList
from tqdm import tqdm

from hypencoder_cb.inference.shared import (
    EncodedItem,
    load_encoded_items_from_disk,
)

def encode_items(
    texts: List[str],
    batch_size: int,
    tokenizer,
    encoder,
    device: torch.device,
    max_length: int,
    desc: str,
) -> np.ndarray:
    """Tokenize and encode a list of texts with the provided encoder."""

    embeddings: List[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[start : start + batch_size]
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            model_output = encoder(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
            )

        embeddings.append(model_output.representation.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def _get_doc_text(doc) -> str:
    if hasattr(doc, "text"):
        return doc.text
    return doc["text"]


def _get_doc_id(doc) -> str:
    if hasattr(doc, "doc_id"):
        return doc.doc_id
    return doc["doc_id"]


def encode_corpus(
    corpus,
    batch_size: int,
    tokenizer,
    encoder,
    device: torch.device,
    max_length: int,
) -> Tuple[List[str], np.ndarray, List[str]]:
    texts = [_get_doc_text(doc) for doc in corpus]
    doc_ids = [_get_doc_id(doc) for doc in corpus]

    print(f"Encoding {len(texts)} documents with max_length={max_length}...")
    doc_embeddings = encode_items(
        texts,
        batch_size,
        tokenizer,
        encoder,
        device,
        max_length,
        desc="Encoding documents",
    )

    return doc_ids, doc_embeddings, texts


def load_encoded_corpus(encoded_path: str) -> Tuple[List[str], np.ndarray]:
    """Load pre-encoded document embeddings (DocList format) from disk."""

    print(f"Loading pre-encoded documents from {encoded_path}")
    encoded_items = load_encoded_items_from_disk(encoded_path)

    doc_ids: List[str] = []
    doc_embeddings: List[np.ndarray] = []

    for item in tqdm(encoded_items, desc="Reading encoded docs"):
        doc_ids.append(str(item.id))
        doc_embeddings.append(np.asarray(item.representation, dtype=np.float32))

    if not doc_embeddings:
        raise ValueError(f"No encoded documents found at {encoded_path}")

    return doc_ids, np.vstack(doc_embeddings)


def save_encoded_corpus(
    doc_texts: Iterable[str],
    doc_ids: Iterable[str],
    doc_embeddings: np.ndarray,
    output_path: str,
) -> None:
    """Save encoded documents to disk in DocList format for reuse."""

    doc_texts = list(doc_texts)
    doc_ids = list(doc_ids)

    if len(doc_texts) != len(doc_ids) or len(doc_texts) != len(doc_embeddings):
        raise ValueError("Texts, IDs, and embeddings must have the same length.")

    output_path = Path(output_path)

    print(f"Saving {len(doc_ids)} encoded documents to {output_path}")

    def generator():
        for text, doc_id, embedding in zip(doc_texts, doc_ids, doc_embeddings):
            yield EncodedItem(
                text=text,
                id=str(doc_id),
                representation=np.asarray(embedding, dtype=np.float32),
            )

    DocList[EncodedItem].push_stream(generator(), f"file://{output_path}")
    print("Encoded documents saved.")


def encode_queries(
    queries,
    batch_size: int,
    tokenizer,
    encoder,
    device: torch.device,
    max_length: int,
) -> Tuple[List[str], np.ndarray]:
    query_texts = [q.text if hasattr(q, "text") else q["text"] for q in queries]
    query_ids = [
        q.query_id if hasattr(q, "query_id") else q["query_id"] for q in queries
    ]

    print(f"Encoding {len(query_texts)} queries with max_length={max_length}...")
    start_time = time.time()
    query_embeddings = encode_items(
        query_texts,
        batch_size,
        tokenizer,
        encoder,
        device,
        max_length,
        desc="Encoding queries",
    )
    elapsed = (time.time() - start_time) * 1000
    print(f"Query encoding time elapsed: {elapsed:.3f} milliseconds.")
    print(
        f"Average query encoding time: {elapsed / max(len(query_texts), 1):.3f} milliseconds."
    )

    return query_ids, query_embeddings


def to_numpy_float32(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def retrieve_faiss(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_ids: List[str],
    top_k: int = 1000,
) -> Dict[int, List[Tuple[str, float]]]:
    """Retrieve top-k documents for each query using Faiss inner-product search."""

    print(f"Computing similarities with Faiss and retrieving top-{top_k} documents per query...")
    queries = to_numpy_float32(query_embeddings)
    docs = to_numpy_float32(doc_embeddings)
    original_doc_ids = np.asarray(doc_ids)

    d = docs.shape[1]

    start_time = time.time()
    index = faiss.IndexFlatIP(d)
    index.add(docs)
    build_time = (time.time() - start_time) * 1000
    print(f"Faiss index build time elapsed: {build_time:.3f} milliseconds.")

    D, I = index.search(queries, top_k)
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    avg_per_query = total_time / len(queries)
    print(
        "BE-base Faiss retrieval time elapsed: "
        f"{total_time:.3f} milliseconds. Avg per query: {avg_per_query:.3f} milliseconds."
    )

    results: Dict[int, List[Tuple[str, float]]] = {}
    for q_idx in range(queries.shape[0]):
        pairs = []
        internal_indices = I[q_idx]
        row_scores = D[q_idx].tolist()
        original_ids = original_doc_ids[internal_indices]

        for doc_id, score in zip(original_ids, row_scores):
            pairs.append((str(doc_id), float(score)))
        results[q_idx] = pairs

    return results


def evaluate_with_ir_measures(
    results: Dict[int, List[Tuple[str, float]]],
    query_ids: List[str],
    qrels,
    metric_names: List[str],
) -> Dict[str, float]:
    import ir_measures
    from ir_measures import calc_aggregate

    run = []
    for query_idx, doc_scores in results.items():
        query_id = query_ids[query_idx]
        for rank, (doc_id, score) in enumerate(doc_scores, 1):
            run.append(ir_measures.ScoredDoc(query_id, str(doc_id), score))

    metrics = [ir_measures.parse_measure(m) for m in metric_names]
    aggregated = calc_aggregate(metrics, qrels, run)
    return {str(key): value for key, value in aggregated.items()}
