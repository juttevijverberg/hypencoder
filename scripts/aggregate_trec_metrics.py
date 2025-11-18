#!/usr/bin/env python3
"""
Aggregate metrics from TREC-DL-2019 and TREC-DL-2020 by averaging them.
"""
import json
from pathlib import Path


def aggregate_metrics(metrics1_path: str, metrics2_path: str, output_path: str):
    """
    Load two metric JSON files, average corresponding metrics, and save the result.
    
    Args:
        metrics1_path: Path to first aggregated_metrics.json
        metrics2_path: Path to second aggregated_metrics.json
        output_path: Path to save the averaged metrics
    """
    # Load the two metric files
    with open(metrics1_path, 'r') as f:
        metrics1 = json.load(f)
    
    with open(metrics2_path, 'r') as f:
        metrics2 = json.load(f)
    
    # Check that both have the same metrics
    if set(metrics1.keys()) != set(metrics2.keys()):
        print("Warning: Metric keys don't match!")
        print(f"Metrics in file 1: {set(metrics1.keys())}")
        print(f"Metrics in file 2: {set(metrics2.keys())}")
        print("Will only average common metrics.")
    
    # Average the metrics
    aggregated = {}
    for metric_name in metrics1.keys():
        if metric_name in metrics2:
            aggregated[metric_name] = (metrics1[metric_name] + metrics2[metric_name]) / 2.0
        else:
            print(f"Skipping {metric_name} - not found in second file")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the aggregated metrics
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=4)
    
    print(f"Aggregated metrics saved to: {output_path}")
    print("\nAveraged metrics:")
    for metric_name, value in aggregated.items():
        print(f"  {metric_name}: {value:.6f}")


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent / "retrieval_outputs"
    
    metrics1_path = base_dir / "trec-dl-2019" / "metrics" / "aggregated_metrics.json"
    metrics2_path = base_dir / "trec-dl-2020" / "metrics" / "aggregated_metrics.json"
    output_path = base_dir / "trec-dl-19-20" / "metrics" / "aggregated_metrics.json"
    
    aggregate_metrics(
        str(metrics1_path),
        str(metrics2_path),
        str(output_path)
    )
