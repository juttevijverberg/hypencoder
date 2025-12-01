"""
Generate adversarial query variations for robustness evaluation.

This script loads queries from ir_datasets, applies adversarial transformations
using the disentangled_information_needs library, and saves the results for
downstream retrieval evaluation.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ir_datasets
from tqdm import tqdm

from disentangled_information_needs.transformations.synonym import SynonymActions
from disentangled_information_needs.transformations.paraphrase import ParaphraseActions
from disentangled_information_needs.transformations.naturality import NaturalityActions
from disentangled_information_needs.transformations.mispelling import MispellingActions
from disentangled_information_needs.transformations.ordering import OrderingActions

DEFAULT_QUERIES = ["What is the capital of France?", "Which country you want to visit?"]
# Update this path if you have a fine-tuned paraphraser checkpoint on disk.
DEFAULT_UQV_MODEL_PATH = "./"

QueryVariant = List[Tuple[int, str, str, str, str]]


def _prepare_inputs(
    queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None
) -> Tuple[List[str], List[int]]:
    """Ensure the helper functions always receive aligned queries/q_ids."""
    queries = list(queries) if queries is not None else list(DEFAULT_QUERIES)
    if not queries:
        raise ValueError("At least one query is required to run the samples.")
    if q_ids is None:
        q_ids = list(range(1, len(queries) + 1))
    else:
        q_ids = list(q_ids)
    if len(queries) != len(q_ids):
        raise ValueError(
            f"queries ({len(queries)}) and q_ids ({len(q_ids)}) must have the same length."
        )
    return queries, q_ids


def test_synonym(queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    synonym_actions = SynonymActions(queries, q_ids)
    return synonym_actions.adversarial_synonym_replacement()


def test_paraphrase(
    queries: Optional[Sequence[str]] = None,
    q_ids: Optional[Sequence[int]] = None,
    uqv_model_path: str = DEFAULT_UQV_MODEL_PATH,
):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    paraphrase_actions = ParaphraseActions(queries, q_ids, uqv_model_path=uqv_model_path)
    return paraphrase_actions.back_translation_paraphrase()


def test_naturality(queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    naturality_actions = NaturalityActions(queries, q_ids)
    return naturality_actions.remove_random_words()


def test_mispelling(queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    mispelling_actions = MispellingActions(queries, q_ids)
    return mispelling_actions.mispelling_chars()


def test_ordering(queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    ordering_actions = OrderingActions(queries, q_ids)
    return ordering_actions.shuffle_word_order()


def test_all(
    queries: Optional[Sequence[str]] = None,
    q_ids: Optional[Sequence[int]] = None,
    uqv_model_path: str = DEFAULT_UQV_MODEL_PATH,
) -> Dict[str, QueryVariant]:
    """Run every transformation and return the generated variations."""
    queries, q_ids = _prepare_inputs(queries, q_ids)
    return {
        "synonym": test_synonym(queries, q_ids),
        "paraphrase": test_paraphrase(queries, q_ids, uqv_model_path=uqv_model_path),
        "naturality": test_naturality(queries, q_ids),
        "mispelling": test_mispelling(queries, q_ids),
        "ordering": test_ordering(queries, q_ids),
    }


def print_samples(
    variations: Dict[str, QueryVariant], max_examples: int = 6
) -> None:
    """Pretty print the first few variations from each transformation."""
    for transform, rows in variations.items():
        print(f"\n[{transform}] total variations: {len(rows)}")
        for q_id, original, rewritten, method, category in rows[:max_examples]:
            print(
                f"- q_id={q_id} | {method} ({category})\n"
                f"  original : {original}\n"
                f"  variation: {rewritten}"
            )


def load_queries_from_ir_dataset(ir_dataset_name: str) -> Tuple[List[str], List[str]]:
    """
    Load queries from an ir_datasets dataset.
    
    Args:
        ir_dataset_name: Name of the ir_datasets dataset (e.g., 'msmarco-passage/trec-dl-2019/judged')
        
    Returns:
        Tuple of (query_ids, query_texts) lists
    """
    print(f"Loading queries from ir_datasets: {ir_dataset_name}")
    dataset = ir_datasets.load(ir_dataset_name)
    
    query_ids = []
    query_texts = []
    
    for query in tqdm(dataset.queries_iter(), desc="Loading queries"):
        q_id = query.query_id if hasattr(query, 'query_id') else query[0]
        q_text = query.text if hasattr(query, 'text') else query[1]
        query_ids.append(str(q_id))
        query_texts.append(q_text)
    
    print(f"Loaded {len(query_ids)} queries")
    return query_ids, query_texts


def save_variations_to_jsonl(
    variations: QueryVariant,
    output_path: str,
    transform_name: str
) -> None:
    """
    Save query variations to JSONL format compatible with retrieve.py.
    
    Args:
        variations: List of (q_id, original, rewritten, method, category) tuples
        output_path: Path to save the JSONL file
        transform_name: Name of the transformation (for metadata)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    seen_qids = set()
    saved_count = 0
    
    with open(output_path, 'w') as f:
        for q_id, original, rewritten, method, category in variations:
            # Only save the first variation for each query ID
            if str(q_id) in seen_qids:
                continue
                
            seen_qids.add(str(q_id))
            saved_count += 1
            
            record = {
                "id": str(q_id),
                "text": rewritten,
                "original_text": original,
                "method": method,
                "category": category,
                "transform_name": transform_name
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"✅ Saved {saved_count} unique variations to {output_path}")


def get_dataset_short_name(ir_dataset_name: str) -> str:
    """
    Get a short name for the dataset for directory structure.
    
    Args:
        ir_dataset_name: Full ir_datasets name
        
    Returns:
        Short name for directory
    """
    mapping = {
        "msmarco-passage/dev/small": "msmarco_passage_dev_small",
        "msmarco-passage/trec-dl-2019/judged": "msmarco_passage_trec_dl_2019_judged",
        "msmarco-passage/trec-dl-2020/judged": "msmarco_passage_trec_dl_2020_judged",
    }
    
    if ir_dataset_name in mapping:
        return mapping[ir_dataset_name]
    
    # Default: replace slashes and dashes with underscores
    return ir_dataset_name.replace("/", "_").replace("-", "_")


def run_attacks_from_ir_dataset(
    ir_dataset_name: str,
    attack_types: List[str],
    output_base_dir: str = "data/adversarial_attack",
    uqv_model_path: str = DEFAULT_UQV_MODEL_PATH,
    max_queries: Optional[int] = None,
) -> None:
    """
    Run adversarial attacks on queries from an ir_datasets dataset.
    
    Args:
        ir_dataset_name: Name of the ir_datasets dataset
        attack_types: List of attack types to run (synonym, paraphrase, naturality, mispelling, ordering, all)
        output_base_dir: Base directory for outputs
        uqv_model_path: Path to UQV model for paraphrase attacks
        max_queries: Maximum number of queries to process (for testing)
    """
    # Load queries from ir_datasets
    query_ids, query_texts = load_queries_from_ir_dataset(ir_dataset_name)
    
    # Limit queries if specified
    if max_queries:
        query_ids = query_ids[:max_queries]
        query_texts = query_texts[:max_queries]
        print(f"Limited to {len(query_ids)} queries for testing")
    
    # Get dataset short name for output directory
    dataset_short_name = get_dataset_short_name(ir_dataset_name)
    
    # Run requested attacks
    attack_functions = {
        "synonym": test_synonym,
        "paraphrase": lambda q, ids: test_paraphrase(q, ids, uqv_model_path),
        "naturality": test_naturality,
        "mispelling": test_mispelling,
        "ordering": test_ordering,
    }
    
    if "all" in attack_types:
        attack_types = list(attack_functions.keys())
    
    for attack_type in attack_types:
        if attack_type not in attack_functions:
            print(f"⚠️  Unknown attack type: {attack_type}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running {attack_type} attack...")
        print(f"{'='*60}")
        
        # Generate variations
        variations = attack_functions[attack_type](query_texts, query_ids)
        
        # Save to JSONL
        output_dir = Path(output_base_dir) / dataset_short_name / attack_type
        output_path = output_dir / "adversarial_queries.jsonl"
        save_variations_to_jsonl(variations, output_path, attack_type)
        
        # Print statistics
        num_changed = sum(
            1 for _, orig, rewritten, _, _ in variations 
            if orig.lower() != rewritten.lower()
        )
        print(f"📊 Successfully modified: {num_changed}/{len(variations)} "
              f"({100*num_changed/len(variations):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial query variations from ir_datasets"
    )
    
    parser.add_argument(
        "--ir_dataset_name",
        type=str,
        required=True,
        help="Name of the ir_datasets dataset (e.g., 'msmarco-passage/trec-dl-2019/judged')"
    )
    parser.add_argument(
        "--attack_types",
        type=str,
        nargs="+",
        default=["all"],
        choices=["synonym", "paraphrase", "naturality", "mispelling", "ordering", "all"],
        help="Types of attacks to run (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/adversarial_attack",
        help="Base output directory (default: data/adversarial_attack)"
    )
    parser.add_argument(
        "--uqv_model_path",
        type=str,
        default=DEFAULT_UQV_MODEL_PATH,
        help="Path to UQV model for paraphrase attacks"
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Maximum number of queries to process (for testing)"
    )
    
    args = parser.parse_args()
    
    run_attacks_from_ir_dataset(
        ir_dataset_name=args.ir_dataset_name,
        attack_types=args.attack_types,
        output_base_dir=args.output_dir,
        uqv_model_path=args.uqv_model_path,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    # If no arguments provided, run demo
    import sys
    if len(sys.argv) == 1:
        print("Running demo with default queries...")
        samples = test_all()
        print_samples(samples)
    else:
        main()
