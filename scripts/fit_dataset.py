#!/usr/bin/env python3
"""
Optimize a JSONL dataset by ensuring all examples have a constant number of items
while maximizing the total number of examples retained.

This script:
1. Finds the most common item count in the dataset
2. Keeps all examples with that item count
3. Trims examples with more items (removes extras)
4. Drops examples with fewer items
5. Saves the optimized dataset
"""
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Tuple


def analyze_dataset(input_file: str) -> Tuple[int, dict]:
    """Analyze the dataset to find the best target item count."""
    item_counts = Counter()
    
    print(f"Analyzing dataset: {input_file}")
    with open(input_file) as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            num_items = len(data['items'])
            item_counts[num_items] += 1
            
            # Print progress every 1000 lines
            if i % 1000 == 0:
                print(f"  Analyzed {i} examples...", end='\r')
    
    print(f"  Analyzed {i} examples total.    ")
    
    # Find the most common item count
    best_count, best_freq = item_counts.most_common(1)[0]
    
    return best_count, dict(item_counts)


def optimize_dataset(input_file: str, output_file: str, target_items: int = None) -> dict:
    """
    Optimize dataset to have constant item counts.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        target_items: Target item count. If None, uses most common count
    
    Returns:
        Dictionary with optimization statistics
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find best target if not specified
    if target_items is None:
        target_items, item_distribution = analyze_dataset(input_file)
        print(f"Dataset analysis:")
        print(f"  Item count distribution: {item_distribution}")
        print(f"  Selected target: {target_items} items (most common, {item_distribution[target_items]} examples)")
    else:
        item_distribution = None
    
    stats = {
        'target_items': target_items,
        'examples_kept': 0,
        'examples_trimmed': 0,
        'examples_dropped': 0,
        'total_items_removed': 0,
    }
    
    with open(input_file) as reader, open(output_path, 'w') as writer:
        for line in reader:
            data = json.loads(line)
            num_items = len(data['items'])
            
            if num_items == target_items:
                # Keep as-is
                writer.write(json.dumps(data) + '\n')
                stats['examples_kept'] += 1
                
            elif num_items > target_items:
                # Trim extra items
                # Separate positives and negatives
                positives = [item for item in data['items'] if item.get('type') == 'positive']
                negatives = [item for item in data['items'] if item.get('type') != 'positive']
                
                # Keep all positives, trim negatives to fit target_items
                negatives_to_keep = target_items - len(positives)
                
                if negatives_to_keep > 0:
                    trimmed_negatives = negatives[:negatives_to_keep]
                    data['items'] = positives + trimmed_negatives
                    items_removed = len(negatives) - len(trimmed_negatives)
                    stats['total_items_removed'] += items_removed
                    stats['examples_trimmed'] += 1
                    writer.write(json.dumps(data) + '\n')
            else:
                # Drop examples with too few items
                stats['examples_dropped'] += 1
    
    return stats


def print_summary(stats: dict, input_file: str, output_file: str):
    """Print optimization summary."""
    total_examples = (stats['examples_kept'] + stats['examples_trimmed'] + 
                      stats['examples_dropped'])
    
    print(f"\n{'='*70}")
    print(f"Optimization Summary")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Target items per example: {stats['target_items']}")
    print(f"\nResults:")
    print(f"  Examples kept (exact match): {stats['examples_kept']}")
    print(f"  Examples trimmed (extra removed): {stats['examples_trimmed']}")
    print(f"  Examples dropped (too few items): {stats['examples_dropped']}")
    print(f"  Total examples retained: {stats['examples_kept'] + stats['examples_trimmed']}")
    print(f"  Total examples removed: {stats['examples_dropped']}")
    print(f"  Total items removed: {stats['total_items_removed']}")
    print(f"\nData retention:")
    retained = stats['examples_kept'] + stats['examples_trimmed']
    retention_pct = (retained / total_examples * 100) if total_examples > 0 else 0
    print(f"  {retained}/{total_examples} examples retained ({retention_pct:.1f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize JSONL dataset to have constant item counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect best item count and optimize
  python optimize_dataset.py \\
    --input_file data/raw.jsonl \\
    --output_file data/optimized.jsonl
    
  # Use specific target item count
  python optimize_dataset.py \\
    --input_file data/raw.jsonl \\
    --output_file data/optimized.jsonl \\
    --target_items 3
    
  # Show analysis without optimizing
  python optimize_dataset.py \\
    --input_file data/raw.jsonl \\
    --analyze_only
        """
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file",
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--target_items",
        type=int,
        default=None,
        help="Target number of items per example (default: most common count)"
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze dataset, don't create output file"
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        target_items, distribution = analyze_dataset(args.input_file)
        print(f"\nDataset Analysis for: {args.input_file}")
        print(f"Item count distribution: {distribution}")
        print(f"Recommended target: {target_items} items")
        print(f"Examples with target count: {distribution[target_items]}")
    else:
        if not args.output_file:
            print("Error: --output_file is required when not using --analyze_only")
            exit(1)
        
        stats = optimize_dataset(
            args.input_file,
            args.output_file,
            args.target_items
        )
        print_summary(stats, args.input_file, args.output_file)
