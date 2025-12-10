#!/usr/bin/env python3
"""
Aggregate metrics from TREC-DL-2019 and TREC-DL-2020 by combining per-query metrics.
"""
import json
from pathlib import Path


def aggregate_metrics(metrics1_path: str, metrics2_path: str, output_path: str):
    """
    Load two per-query metric JSON files, combine all queries, and compute weighted average.
    Datasets with more queries will have proportionally more impact on the final aggregated metrics.
    
    Args:
        metrics1_path: Path to first per_query_metrics.json
        metrics2_path: Path to second per_query_metrics.json
        output_path: Path to save the aggregated metrics
    """
    # Load the two per-query metric files
    with open(metrics1_path, 'r') as f:
        per_query_metrics1 = json.load(f)
    
    with open(metrics2_path, 'r') as f:
        per_query_metrics2 = json.load(f)
    
    num_queries1 = len(per_query_metrics1)
    num_queries2 = len(per_query_metrics2)
    total_queries = num_queries1 + num_queries2
    
    print(f"Dataset 1: {num_queries1} queries")
    print(f"Dataset 2: {num_queries2} queries")
    print(f"Total: {total_queries} queries")
    
    # Get all metric names from the first query of each dataset
    metric_names = set()
    if per_query_metrics1:
        first_query1 = next(iter(per_query_metrics1.values()))
        metric_names.update(first_query1.keys())
    if per_query_metrics2:
        first_query2 = next(iter(per_query_metrics2.values()))
        metric_names.update(first_query2.keys())
    
    # Aggregate metrics by summing all query scores and dividing by total queries
    aggregated = {}
    for metric_name in sorted(metric_names):
        sum_metric = 0.0
        count = 0
        
        # Sum from dataset 1
        for query_metrics in per_query_metrics1.values():
            if metric_name in query_metrics:
                sum_metric += query_metrics[metric_name]
                count += 1
        
        # Sum from dataset 2
        for query_metrics in per_query_metrics2.values():
            if metric_name in query_metrics:
                sum_metric += query_metrics[metric_name]
                count += 1
        
        # Compute average
        if count > 0:
            aggregated[metric_name] = sum_metric / count
        else:
            print(f"Warning: No data found for metric {metric_name}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the aggregated metrics
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=4)
    
    print(f"\nAggregated metrics saved to: {output_path}")
    print("\nWeighted averaged metrics:")
    for metric_name, value in aggregated.items():
        print(f"  {metric_name}: {value:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics1_path", type=str, required=False)
    parser.add_argument("--metrics2_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    args = parser.parse_args()

    if args.metrics1_path and args.metrics2_path and args.output_path:
        aggregate_metrics(args.metrics1_path, args.metrics2_path, args.output_path)
    else:
        base_dir = Path(__file__).parent.parent / "retrieval_outputs"
        metrics1_path = base_dir / "trec-dl-2019" / "metrics" / "per_query_metrics.json"
        metrics2_path = base_dir / "trec-dl-2020" / "metrics" / "per_query_metrics.json"
        output_path = base_dir / "trec-dl-19-20" / "metrics" / "aggregated_metrics.json"

        aggregate_metrics(
            str(metrics1_path),
            str(metrics2_path),
            str(output_path)
        )
