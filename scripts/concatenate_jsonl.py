import argparse
import json
from pathlib import Path
from collections import Counter


def filter_and_concatenate_jsonl(input_files, output_file, target_items=3, trim_extra=True):
    """
    Trim/filter JSONL files to have target_items items per example, then concatenate.
    
    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path
        target_items: Number of items each example should have (default: 3)
        trim_extra: If True, trim extra items instead of removing entire examples
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_examples = 0
    total_trimmed = 0
    
    with open(output_path, 'w') as writer:
        for input_file in input_files:
            input_path = Path(input_file).expanduser()
            item_counts_before = Counter()
            item_counts_after = Counter()
            examples_processed = 0
            items_trimmed = 0
            
            with open(input_path, 'r') as reader:
                for line in reader:
                    data = json.loads(line)
                    num_items = len(data['items'])
                    item_counts_before[num_items] += 1
                    
                    if trim_extra and num_items > target_items:
                        # Separate positives and negatives
                        positives = [item for item in data['items'] if item.get('type') == 'positive']
                        negatives = [item for item in data['items'] if item.get('type') != 'positive']
                        
                        # Keep all positives, trim negatives to fit target_items
                        negatives_to_keep = target_items - len(positives)
                        if negatives_to_keep > 0:
                            trimmed_negatives = negatives[:negatives_to_keep]
                            data['items'] = positives + trimmed_negatives
                            items_trimmed += len(negatives) - len(trimmed_negatives)
                    elif num_items < target_items:
                        # Skip examples with fewer items than target
                        continue
                    
                    num_items_final = len(data['items'])
                    item_counts_after[num_items_final] += 1
                    writer.write(json.dumps(data) + '\n')
                    examples_processed += 1
            
            total_examples += examples_processed
            total_trimmed += items_trimmed
            
            print(f"\nFile: {input_file}")
            print(f"  Before trimming: {dict(item_counts_before)}")
            print(f"  After trimming: {dict(item_counts_after)}")
            print(f"  Examples processed: {examples_processed}")
            print(f"  Items trimmed: {items_trimmed}")
    
    print(f"\n{'='*60}")
    print(f"Output: {output_file}")
    print(f"Total examples written: {total_examples}")
    print(f"Total items trimmed: {total_trimmed}")
    print(f"{'='*60}")


def concatenate_jsonl(input_files, output_file):
    """Concatenate multiple JSONL files into a single JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as writer:
        for input_file in input_files:
            input_path = Path(input_file).expanduser()
            with open(input_path, 'r') as reader:
                for line in reader:
                    writer.write(line)
    
    print(f"Concatenated {len(input_files)} files into {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate and optionally filter JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trim extra items and concatenate (keeps all examples, trims extras)
  python concatenate_jsonl.py \\
    --input_files data/TOT/train/Movies/converted_train.jsonl data/TOT/train/Books/converted_train.jsonl \\
    --output_file data/TOT/train/converted_train.jsonl \\
    --filter --target_items 3 --trim_extra
    
  # Just concatenate without filtering
  python concatenate_jsonl.py \\
    --input_files file1.jsonl file2.jsonl \\
    --output_file combined.jsonl
        """
    )
    parser.add_argument("--input_files", nargs="+", required=True, help="Input JSONL file paths")
    parser.add_argument("--output_file", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter examples before concatenating"
    )
    parser.add_argument(
        "--target_items",
        type=int,
        default=3,
        help="Target number of items per example (default: 3)"
    )
    parser.add_argument(
        "--trim_extra",
        action="store_true",
        default=True,
        help="Trim extra items instead of removing entire examples (default: True)"
    )
    parser.add_argument(
        "--no-trim_extra",
        dest="trim_extra",
        action="store_false",
        help="Remove entire examples instead of trimming (opposite of --trim_extra)"
    )
    
    args = parser.parse_args()
    
    if args.filter:
        filter_and_concatenate_jsonl(args.input_files, args.output_file, args.target_items, args.trim_extra)
    else:
        concatenate_jsonl(args.input_files, args.output_file)
