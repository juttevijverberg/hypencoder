from pathlib import Path
from typing import Dict, List, Optional, Union

import fire
import torch

from hypencoder_cb.inference.retrieve import (
    HypencoderRetriever,
    do_eval_and_pretty_print,
)
from hypencoder_cb.inference.shared import retrieve_for_jsonl_queries


def do_retrieval_multiple(
    model_name_or_path: str,
    encoded_item_path: str,
    base_data_dir: str,
    base_output_dir: str,
    attack_types: List[str],
    qrel_json: str,
    query_id_key: str = "id",
    query_text_key: str = "text",
    dtype: str = "fp32",
    top_k: int = 1000,
    batch_size: int = 100_000,
    retriever_kwargs: Optional[Dict] = None,
    query_max_length: int = 64,
    include_content: bool = True,
    do_eval: bool = True,
    metric_names: Optional[List[str]] = None,
    ignore_same_id: bool = False,
    model_type: str = "hypencoder",
) -> None:
    """Does retrieval for multiple attack types using the same loaded index.

    Args:
        model_name_or_path (str): Name or path to a HypencoderDualEncoder checkpoint.
        encoded_item_path (str): Path to the encoded items.
        base_data_dir (str): Base directory containing subdirectories for each attack type.
        base_output_dir (str): Base directory for outputs.
        attack_types (List[str]): List of attack types to process.
        qrel_json (str): Path to the qrels JSON file.
        query_id_key (str, optional): Key for query ID in JSONL. Defaults to "id".
        query_text_key (str, optional): Key for query text in JSONL. Defaults to "text".
        dtype (str, optional): Dtype for model. Defaults to "fp32".
        top_k (int, optional): Top K items to retrieve. Defaults to 1000.
        batch_size (int, optional): Batch size. Defaults to 100,000.
        retriever_kwargs (Optional[Dict], optional): Extra args for retriever. Defaults to None.
        query_max_length (int, optional): Max query length. Defaults to 64.
        include_content (bool, optional): Include content in output. Defaults to True.
        do_eval (bool, optional): Whether to do evaluation. Defaults to True.
        metric_names (Optional[List[str]], optional): Metrics to compute. Defaults to None.
        ignore_same_id (bool, optional): Ignore same ID. Defaults to False.
    """
    
    retriever_kwargs = retriever_kwargs if retriever_kwargs is not None else {}

    print(f"Initializing retriever with model: {model_name_or_path}")
    print(f"Loading encoded items from: {encoded_item_path}")
    
    # Choose retriever class based on model type
    if model_type == "biencoder":
        from hypencoder_cb.inference.retrieve_biencoder import BiEncoderRetriever
        retriever_cls = BiEncoderRetriever
    elif model_type == "hypencoder":
        retriever_cls = HypencoderRetriever
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Initialize retriever once (loads the index)
    retriever = retriever_cls(
        model_name_or_path=model_name_or_path,
        encoded_item_path=encoded_item_path,
        dtype=dtype,
        batch_size=batch_size,
        query_max_length=query_max_length,
        ignore_same_id=ignore_same_id,
        **retriever_kwargs,
    )
    
    base_data_path = Path(base_data_dir)
    base_output_path = Path(base_output_dir)
    
    for attack in attack_types:
        print(f"\n{'='*60}")
        print(f"Processing attack type: {attack}")
        print(f"{'='*60}")
        
        query_jsonl = base_data_path / attack / "adversarial_queries.jsonl"
        output_dir = base_output_path / attack
        
        if not query_jsonl.exists():
            print(f"⚠️  Warning: Input file not found: {query_jsonl}")
            continue
            
        output_dir.mkdir(parents=True, exist_ok=True)
        retrieval_file = output_dir / "retrieved_items.jsonl"
        metric_dir = output_dir / "metrics"
        
        print(f"Retrieving for {query_jsonl}...")
        retrieve_for_jsonl_queries(
            retriever=retriever,
            query_jsonl=str(query_jsonl),
            output_path=retrieval_file,
            top_k=top_k,
            include_content=include_content,
            include_type=include_content,
            query_id_key=query_id_key,
            query_text_key=query_text_key,
        )
        
        if do_eval:
            print(f"Evaluating results...")
            do_eval_and_pretty_print(
                ir_dataset_name=None,
                qrel_json=qrel_json,
                retrieval_path=retrieval_file,
                output_dir=metric_dir,
                metric_names=metric_names,
            )
            
    print("\nAll retrievals completed.")


if __name__ == "__main__":
    fire.Fire(do_retrieval_multiple)
