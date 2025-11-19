#!/usr/bin/env python3
"""
Adversarial Attack Pipeline for Hypencoder
Generates adversarial queries using TextAttack to evaluate model robustness.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import ir_datasets
import torch
from textattack.attack_recipes import (
    TextFoolerJin2019,
    BAEGarg2019,
    BERTAttackLi2020,
    DeepWordBugGao2018,
)
from textattack.datasets import Dataset
from textattack.models.wrappers import ModelWrapper
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_cb.inference.shared import load_encoded_items_from_disk
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.utils.torch_utils import dtype_lookup


class HypencoderQueryWrapper(ModelWrapper):
    """
    Wrapper around Hypencoder query encoder for TextAttack compatibility.
    
    TextAttack expects:
    - __call__(text_input_list) -> predictions (batch predictions)
    - tokenizer attribute
    """
    
    def __init__(
        self,
        model: HypencoderDualEncoder,
        tokenizer: AutoTokenizer,
        encoded_items: torch.Tensor,
        device: str = "cuda",
        query_max_length: int = 64,
        top_k: int = 10,
        query_model_kwargs: Optional[Dict] = None
    ):
        """
        Args:
            model: Hypencoder dual encoder model
            tokenizer: Tokenizer for the model
            encoded_items: Pre-encoded document embeddings
            device: Device to run on
            query_max_length: Max query length
            top_k: Number of documents to retrieve for scoring
            query_model_kwargs: Additional kwargs for query model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.encoded_items = encoded_items
        self.device = device
        self.query_max_length = query_max_length
        self.top_k = top_k
        self.query_model_kwargs = query_model_kwargs or {}
        
    def __call__(self, text_input_list: List[str]) -> torch.Tensor:
        """
        Encode queries and compute retrieval scores.
        
        Returns a tensor of shape (batch_size, 2) where:
        - [:, 0] = negative retrieval score (for attack to maximize)
        - [:, 1] = positive retrieval score (original performance)
        
        TextAttack will try to minimize [:, 1] and maximize [:, 0]
        """
        # Tokenize batch of queries
        tokenized = self.tokenizer(
            text_input_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
        ).to(self.device)
        
        with torch.no_grad():
            # Encode queries
            query_output = self.model.query_encoder(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )
            
            query_model = query_output.representation
            
            # Compute similarity with encoded documents
            # For efficiency, only use a subset of documents
            num_items = min(self.top_k * 100, len(self.encoded_items))
            batch_items = self.encoded_items[:num_items].unsqueeze(0).expand(
                len(text_input_list), -1, -1
            )
            
            # Compute similarities
            similarities = query_model(batch_items, **self.query_model_kwargs)
            
            # Get top-k scores (higher is better for retrieval)
            top_scores, _ = torch.topk(similarities, self.top_k, dim=-1)
            
            # Average top-k scores as retrieval quality metric
            retrieval_scores = top_scores.mean(dim=-1)
        
        # Return as classification scores: [negative_class, positive_class]
        # TextAttack will attack to minimize positive_class score
        batch_size = len(text_input_list)
        predictions = torch.zeros(batch_size, 2, device=self.device)
        predictions[:, 0] = -retrieval_scores  # Negative class (attack goal)
        predictions[:, 1] = retrieval_scores   # Positive class (original)
        
        return predictions


def load_queries_from_dataset(ir_dataset_name: str, max_queries: Optional[int] = None):
    """Load queries from an IR dataset."""
    dataset = ir_datasets.load(ir_dataset_name)
    queries = []
    
    for i, query in enumerate(dataset.queries_iter()):
        if max_queries and i >= max_queries:
            break
        queries.append({
            "query_id": query.query_id,
            "text": query.text,
            "original_text": query.text
        })
    
    return queries


def load_queries_from_jsonl(jsonl_path: str, query_id_key: str = "id", 
                             query_text_key: str = "text", 
                             max_queries: Optional[int] = None):
    """Load queries from JSONL file."""
    queries = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if max_queries and i >= max_queries:
                break
            query = json.loads(line)
            queries.append({
                "query_id": query[query_id_key],
                "text": query[query_text_key],
                "original_text": query[query_text_key]
            })
    
    return queries


def create_textattack_dataset(queries: List[Dict], labels: List[int] = None):
    """Create TextAttack dataset from queries."""
    texts = [q["text"] for q in queries]
    
    # Label 1 = good retrieval (we want to attack this)
    if labels is None:
        labels = [1] * len(texts)
    
    return Dataset(list(zip(texts, labels)))


def get_attack_recipe(attack_name: str, model_wrapper: ModelWrapper):
    """Get the specified attack recipe."""
    attack_recipes = {
        "textfooler": TextFoolerJin2019,
        "bae": BAEGarg2019,
        "bert-attack": BERTAttackLi2020,
        "deepwordbug": DeepWordBugGao2018,
    }
    
    if attack_name.lower() not in attack_recipes:
        raise ValueError(
            f"Unknown attack: {attack_name}. "
            f"Choose from: {list(attack_recipes.keys())}"
        )
    
    attack_class = attack_recipes[attack_name.lower()]
    return attack_class.build(model_wrapper)


def attack_queries(
    queries: List[Dict],
    model_wrapper: ModelWrapper,
    attack_name: str = "textfooler",
    output_path: str = None,
    num_examples: Optional[int] = None
):
    """
    Perform adversarial attacks on queries.
    
    Args:
        queries: List of query dictionaries
        model_wrapper: Wrapped model for TextAttack
        attack_name: Name of attack recipe
        output_path: Path to save adversarial queries
        num_examples: Number of queries to attack (None = all)
    """
    print(f"Setting up {attack_name} attack...")
    attack = get_attack_recipe(attack_name, model_wrapper)
    
    # Create dataset
    dataset = create_textattack_dataset(queries[:num_examples] if num_examples else queries)
    
    print(f"Attacking {len(dataset)} queries...")
    adversarial_queries = []
    
    for i, (example, label) in enumerate(tqdm(dataset, desc="Attacking queries")):
        try:
            # Run attack
            result = attack.attack(example, label)
            
            # Store results
            adv_query = {
                "query_id": queries[i]["query_id"],
                "original_text": queries[i]["original_text"],
                "adversarial_text": result.perturbed_text() if result.perturbed_result else queries[i]["text"],
                "attack_success": result.perturbed_result is not None,
                "num_words_changed": result.num_words_changed if hasattr(result, 'num_words_changed') else 0,
                "original_score": result.original_result.score if hasattr(result, 'original_result') else None,
                "perturbed_score": result.perturbed_result.score if result.perturbed_result else None,
            }
            
            adversarial_queries.append(adv_query)
            
        except Exception as e:
            print(f"Error attacking query {queries[i]['query_id']}: {e}")
            # Keep original on error
            adversarial_queries.append({
                "query_id": queries[i]["query_id"],
                "original_text": queries[i]["original_text"],
                "adversarial_text": queries[i]["text"],
                "attack_success": False,
                "error": str(e)
            })
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        with open(output_path, 'w') as f:
            for adv_q in adversarial_queries:
                f.write(json.dumps(adv_q) + "\n")
        
        print(f"Adversarial queries saved to: {output_path}")
        
        # Print statistics
        successful_attacks = sum(1 for q in adversarial_queries if q.get("attack_success", False))
        print(f"\nAttack Statistics:")
        print(f"  Total queries: {len(adversarial_queries)}")
        print(f"  Successful attacks: {successful_attacks} ({100*successful_attacks/len(adversarial_queries):.1f}%)")
        if successful_attacks > 0:
            avg_words_changed = sum(q.get("num_words_changed", 0) for q in adversarial_queries if q.get("attack_success")) / successful_attacks
            print(f"  Avg words changed: {avg_words_changed:.2f}")
    
    return adversarial_queries


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial queries for Hypencoder robustness evaluation"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to Hypencoder model"
    )
    parser.add_argument(
        "--encoded_item_path",
        type=str,
        required=True,
        help="Path to pre-encoded documents"
    )
    
    # Query source (mutually exclusive)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--ir_dataset_name",
        type=str,
        help="IR dataset name (e.g., 'msmarco-passage/trec-dl-2020/judged')"
    )
    query_group.add_argument(
        "--query_jsonl",
        type=str,
        help="Path to queries JSONL file"
    )
    
    # Query JSONL keys (if using JSONL)
    parser.add_argument(
        "--query_id_key",
        type=str,
        default="id",
        help="Key for query ID in JSONL"
    )
    parser.add_argument(
        "--query_text_key",
        type=str,
        default="text",
        help="Key for query text in JSONL"
    )
    
    # Attack arguments
    parser.add_argument(
        "--attack_method",
        type=str,
        default="textfooler",
        choices=["textfooler", "bae", "bert-attack", "deepwordbug"],
        help="Attack method to use"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save adversarial queries (JSONL format)"
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Maximum number of queries to attack (for testing)"
    )
    
    # Model configuration
    parser.add_argument(
        "--query_max_length",
        type=int,
        default=64,
        help="Maximum query length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32", "bf16"],
        help="Data type for model"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k documents to consider for scoring"
    )
    
    args = parser.parse_args()
    
    # Setup
    dtype = dtype_lookup(args.dtype)
    device = args.device
    
    print(f"Loading model from {args.model_name_or_path}...")
    model = HypencoderDualEncoder.from_pretrained(args.model_name_or_path).to(device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    print(f"Loading encoded items from {args.encoded_item_path}...")
    encoded_items = load_encoded_items_from_disk(args.encoded_item_path)
    encoded_item_embeddings = torch.stack([
        torch.tensor(x.representation, dtype=dtype)
        for x in tqdm(encoded_items, desc="Loading embeddings")
    ]).to(device)
    
    print(f"Loaded {len(encoded_item_embeddings)} document embeddings")
    
    # Load queries
    if args.ir_dataset_name:
        print(f"Loading queries from IR dataset: {args.ir_dataset_name}")
        queries = load_queries_from_dataset(args.ir_dataset_name, args.max_queries)
    else:
        print(f"Loading queries from JSONL: {args.query_jsonl}")
        queries = load_queries_from_jsonl(
            args.query_jsonl,
            args.query_id_key,
            args.query_text_key,
            args.max_queries
        )
    
    print(f"Loaded {len(queries)} queries")
    
    # Create model wrapper
    print("Creating model wrapper for TextAttack...")
    model_wrapper = HypencoderQueryWrapper(
        model=model,
        tokenizer=tokenizer,
        encoded_items=encoded_item_embeddings,
        device=device,
        query_max_length=args.query_max_length,
        top_k=args.top_k
    )
    
    # Run attacks
    adversarial_queries = attack_queries(
        queries=queries,
        model_wrapper=model_wrapper,
        attack_name=args.attack_method,
        output_path=args.output_path,
        num_examples=args.max_queries
    )
    
    print("\nAdversarial attack pipeline completed!")
    print(f"Results saved to: {args.output_path}")
    print("\nNext steps:")
    print("1. Run retrieval with adversarial queries using retrieve.py")
    print("2. Compare metrics with original queries to measure performance degradation")


if __name__ == "__main__":
    main()
