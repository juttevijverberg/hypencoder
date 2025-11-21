"""Combine retrieval results from TREC-DL-2019 and TREC-DL-2020 and evaluate together."""

import json
import argparse
from pathlib import Path
from hypencoder_cb.utils.data_utils import load_qrels_from_ir_datasets
from hypencoder_cb.utils.eval_utils import calculate_metrics_to_file, load_standard_format_as_run


def combine_jsonl_files(input_files, output_file):
    """Combine multiple JSONL retrieval files into one."""
    print(f"Combining {len(input_files)} retrieval files...")
    with open(output_file, 'w') as f_out:
        for input_file in input_files:
            print(f"  Reading {input_file}...")
            with open(input_file, 'r') as f_in:
                for line in f_in:
                    f_out.write(line)
    print(f"✅ Combined retrieval saved to {output_file}")


def combine_qrels_from_datasets(dataset_names):
    """Load and combine qrels from multiple ir_datasets."""
    combined_qrels = {}
    
    for dataset_name in dataset_names:
        print(f"Loading qrels from {dataset_name}...")
        qrels = load_qrels_from_ir_datasets(dataset_name)
        combined_qrels.update(qrels)
    
    return combined_qrels


def main():
    parser = argparse.ArgumentParser(description="Combine TREC-DL retrieval results")
    parser.add_argument('--retrieval_files', nargs='+', required=True,
                       help='Paths to retrieved_items.jsonl files')
    parser.add_argument('--ir_dataset_names', nargs='+', required=True,
                       help='IR dataset names for qrels (e.g., msmarco-passage/trec-dl-2019/judged)')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save combined results')
    parser.add_argument('--metric_names', nargs='+',
                       default=['nDCG@10', 'RR', 'nDCG@1000', 'R@1000', 'AP', 'nDCG@5'],
                       help='Metrics to calculate')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(exist_ok=True)
    
    # Combine retrieval results
    combined_retrieval_file = output_dir / "retrieved_items.jsonl"
    combine_jsonl_files(args.retrieval_files, combined_retrieval_file)
    
    # Load and combine qrels
    combined_qrels = combine_qrels_from_datasets(args.ir_dataset_names)
    print(f"Combined qrels: {len(combined_qrels)} queries")
    
    # Create TREC format output
    retrieval_txt_file = output_dir / "retrieved_items.txt"
    print(f"Creating TREC format file...")
    with open(combined_retrieval_file, 'r') as f_in, open(retrieval_txt_file, 'w') as f_out:
        for line in f_in:
            item = json.loads(line)
            
            # Handle different possible structures
            if 'query' in item and 'items' in item:
                # Structure: {"query": {"id": "...", ...}, "items": [...]}
                query_id = item['query']['id']
                retrieved_items = item['items']
            elif 'query_id' in item and 'retrieved_items' in item:
                # Structure: {"query_id": "...", "retrieved_items": [...]}
                query_id = item['query_id']
                retrieved_items = item['retrieved_items']
            else:
                # Unknown structure - skip with warning
                print(f"Warning: Unknown structure in line, skipping: {list(item.keys())}")
                continue
            
            for rank, ret_item in enumerate(retrieved_items, 1):
                if isinstance(ret_item, dict):
                    doc_id = ret_item.get('id', ret_item.get('doc_id', ''))
                    score = ret_item.get('score', 0.0)
                else:
                    print(f"Warning: Unexpected ret_item type: {type(ret_item)}")
                    continue
                f_out.write(f"{query_id} Q0 {doc_id} {rank} {score} combined\n")
    
    # Evaluate
    print("Evaluating combined results...")
    run = load_standard_format_as_run(combined_retrieval_file, score_key='score')
    calculate_metrics_to_file(
        run=run,
        qrels=combined_qrels,
        output_folder=metric_dir,
        metric_names=args.metric_names
    )
    
    print(f"\n✅ Combined results saved to {output_dir}")
    
    # Print metrics
    metrics_file = metric_dir / "aggregated_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        print("\nCombined Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()
