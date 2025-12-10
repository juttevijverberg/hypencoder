#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from scipy.stats import wilcoxon

def load_metrics(folder_path: Path):
    metrics_path = folder_path / "metrics" / "per_query_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)

def save_results(folder_path: Path, text: str):
    output_path = folder_path / "stat_test.txt"
    with open(output_path, "w") as f:
        f.write(text)
    print(f"Saved statistical test results to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Wilcoxon Signed-Rank Test for retrieval metrics.")
    parser.add_argument("--base_path", help="Path to the base experiment folder")
    parser.add_argument("--new_path", help="Path to the new experiment folder (results stored here)")
    args = parser.parse_args()

    base_path = Path(args.base_path)
    new_path = Path(args.new_path)

    # Load metric files
    A = load_metrics(base_path)
    B = load_metrics(new_path)

    # Shared QIDs
    shared_qids = sorted(set(A.keys()) & set(B.keys()))
    if not shared_qids:
        raise ValueError("No shared query IDs found between the folders.")

    # Shared metrics
    first_q = shared_qids[0]
    shared_metrics = set(A[first_q].keys()) & set(B[first_q].keys())

    results_text = []
    results_text.append(f"Compared models:\n  Base: {base_path}\n  New: {new_path}\n")
    results_text.append(f"Number of shared queries: {len(shared_qids)}\n")
    results_text.append("Wilcoxon Signed-Rank Test Results:\n")

    for metric in sorted(shared_metrics):
        a_vals = [A[q][metric] for q in shared_qids]
        b_vals = [B[q][metric] for q in shared_qids]

        try:
            stat, p = wilcoxon(a_vals, b_vals)
            results_text.append(f"{metric:12s}  p={p:.6f}  stat={stat}")
        except ValueError as e:
            results_text.append(f"{metric:12s}  ERROR: {e}")

    results_text = "\n".join(results_text)

    # Save results ONLY in the new_path folder
    save_results(new_path, results_text)

    print("\nDone.")

if __name__ == "__main__":
    main()
