#!/usr/bin/env python3
"""
Compare metrics between original and adversarial queries to measure robustness.
"""
import argparse
import json
from pathlib import Path


def load_metrics(metrics_path: str):
    """Load aggregated metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def calculate_degradation(original: float, adversarial: float) -> dict:
    """Calculate performance degradation metrics."""
    absolute_diff = adversarial - original
    relative_diff = (absolute_diff / original * 100) if original != 0 else 0
    
    return {
        "original": original,
        "adversarial": adversarial,
        "absolute_difference": absolute_diff,
        "relative_difference_pct": relative_diff
    }


def compare_metrics(original_metrics: dict, adversarial_metrics: dict):
    """Compare original and adversarial metrics."""
    comparison = {}
    
    for metric_name in original_metrics.keys():
        if metric_name in adversarial_metrics:
            comparison[metric_name] = calculate_degradation(
                original_metrics[metric_name],
                adversarial_metrics[metric_name]
            )
    
    return comparison


def print_comparison_table(comparison: dict):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("ADVERSARIAL ATTACK ROBUSTNESS EVALUATION")
    print("="*80)
    print()
    print(f"{'Metric':<15} {'Original':<12} {'Adversarial':<12} {'Abs. Diff':<12} {'Rel. Diff %':<12}")
    print("-"*80)
    
    for metric_name, values in comparison.items():
        print(
            f"{metric_name:<15} "
            f"{values['original']:<12.4f} "
            f"{values['adversarial']:<12.4f} "
            f"{values['absolute_difference']:<12.4f} "
            f"{values['relative_difference_pct']:<12.2f}"
        )
    
    print("-"*80)
    
    # Calculate average degradation
    avg_rel_diff = sum(v['relative_difference_pct'] for v in comparison.values()) / len(comparison)
    print(f"\nAverage Relative Performance Degradation: {avg_rel_diff:.2f}%")
    print()


def analyze_attack_success(adversarial_queries_path: str):
    """Analyze attack success statistics from adversarial queries file."""
    if not Path(adversarial_queries_path).exists():
        print(f"Warning: Adversarial queries file not found: {adversarial_queries_path}")
        return None
    
    successful_attacks = 0
    total_queries = 0
    total_words_changed = 0
    
    with open(adversarial_queries_path, 'r') as f:
        for line in f:
            query = json.loads(line)
            total_queries += 1
            if query.get('attack_success', False):
                successful_attacks += 1
                total_words_changed += query.get('num_words_changed', 0)
    
    stats = {
        "total_queries": total_queries,
        "successful_attacks": successful_attacks,
        "success_rate_pct": (successful_attacks / total_queries * 100) if total_queries > 0 else 0,
        "avg_words_changed": (total_words_changed / successful_attacks) if successful_attacks > 0 else 0
    }
    
    return stats


def print_attack_statistics(stats: dict):
    """Print attack statistics."""
    if stats is None:
        return
    
    print("\n" + "="*80)
    print("ATTACK STATISTICS")
    print("="*80)
    print(f"Total Queries:        {stats['total_queries']}")
    print(f"Successful Attacks:   {stats['successful_attacks']} ({stats['success_rate_pct']:.1f}%)")
    print(f"Avg Words Changed:    {stats['avg_words_changed']:.2f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics between original and adversarial queries"
    )
    parser.add_argument(
        "--original_metrics",
        type=str,
        required=True,
        help="Path to original aggregated_metrics.json"
    )
    parser.add_argument(
        "--adversarial_metrics",
        type=str,
        required=True,
        help="Path to adversarial aggregated_metrics.json"
    )
    parser.add_argument(
        "--adversarial_queries",
        type=str,
        default=None,
        help="Path to adversarial queries JSONL (for attack statistics)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save comparison results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Load metrics
    print(f"Loading original metrics from: {args.original_metrics}")
    original_metrics = load_metrics(args.original_metrics)
    
    print(f"Loading adversarial metrics from: {args.adversarial_metrics}")
    adversarial_metrics = load_metrics(args.adversarial_metrics)
    
    # Compare
    comparison = compare_metrics(original_metrics, adversarial_metrics)
    
    # Print results
    print_comparison_table(comparison)
    
    # Analyze attack statistics if provided
    if args.adversarial_queries:
        stats = analyze_attack_success(args.adversarial_queries)
        print_attack_statistics(stats)
        comparison["attack_statistics"] = stats
    
    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=4)
        
        print(f"\nComparison results saved to: {output_path}")


if __name__ == "__main__":
    main()
