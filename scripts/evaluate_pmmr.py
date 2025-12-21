import json
from pathlib import Path
from typing import List, Optional

import fire

from hypencoder_cb.utils.data_utils import (
    load_qrels_from_ir_datasets,
    load_qrels_from_json,
)
from hypencoder_cb.utils.eval_utils import (
    calculate_metrics,
    compute_followir_p_mrr,
    load_standard_format_as_run,
    pretty_print_standard_format,
)


def do_eval_multiple_runs(
    original_run_path: str,
    new_run_path: str,
    output_dir: str,
    ir_dataset_name: Optional[str] = None,
    original_qrel_json: Optional[str] = None,
    new_qrel_json: Optional[str] = None,
    metric_names: Optional[List[str]] = None,
) -> None:
    """Does evaluation for two separate runs, computes average metrics, and
    calculates p-MRR.

    Args:
        original_run_path (str): Path to the original retrieval JSONL file.
        new_run_path (str): Path to the new retrieval JSONL file.
        output_dir (str): Path to the output directory.
        ir_dataset_name (Optional[str], optional): If provided is used to
            get the qrels used for evaluation for BOTH runs. If None, then
            `original_qrel_json` must be provided. Defaults to None.
        original_qrel_json (Optional[str], optional): If provided is used as
            the qrels for the original run. Defaults to None.
        new_qrel_json (Optional[str], optional): If provided is used as the
            qrels for the new run. If not provided, `original_qrel_json` is
            used for both. Defaults to None.
        metric_names (Optional[List[str]], optional): A list of metrics to
            compute. These are passed to IR-Measures so should be compatible.
            If "p-MRR" is included, it will be calculated separately.
            If None, a default set of metrics is found. Defaults to None.

    Raises:
        ValueError: If evaluation data is not provided correctly.
    """

    if ir_dataset_name is None and original_qrel_json is None:
        raise ValueError(
            "One of ir_dataset_name or original_qrel_json must be provided."
        )

    if ir_dataset_name is not None and (
        original_qrel_json is not None or new_qrel_json is not None
    ):
        raise ValueError(
            "Only one of ir_dataset_name or qrel JSON files can be provided."
        )

    if ir_dataset_name:
        original_qrels = load_qrels_from_ir_datasets(ir_dataset_name)
        new_qrels = original_qrels
    else:
        original_qrels = load_qrels_from_json(original_qrel_json)
        if new_qrel_json:
            new_qrels = load_qrels_from_json(new_qrel_json)
        else:
            print("`new_qrel_json` not provided, using `original_qrel_json` for both runs.")
            new_qrels = original_qrels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    should_calc_p_mrr = False
    if metric_names:
        metric_names = list(metric_names)
        if "p-MRR" in metric_names:
            should_calc_p_mrr = True
            metric_names.remove("p-MRR")
        if not metric_names: 
            metric_names = None

    original_run = load_standard_format_as_run(original_run_path, score_key="score")
    new_run = load_standard_format_as_run(new_run_path, score_key="score")

    all_metrics = {}
    run_metrics = {}

    for run_name, run_data, qrels in [
        ("original", original_run, original_qrels),
        ("new", new_run, new_qrels),
    ]:
        print(f"--- Evaluating {run_name} run ---")
        agg_metrics, _ = calculate_metrics(run_data, qrels, metric_names=metric_names)
        run_metrics[run_name] = {str(k): v for k, v in agg_metrics.items()}
        print(json.dumps(run_metrics[run_name], indent=2))

    all_metrics["individual_runs"] = run_metrics

    avg_metrics = {}
    if len(run_metrics) > 0:
        all_keys = set()
        for metrics in run_metrics.values():
            all_keys.update(metrics.keys())

        for key in sorted(list(all_keys)):
            values = [metrics.get(key, 0) for metrics in run_metrics.values()]
            avg_metrics[key] = sum(values) / len(values)

    all_metrics["average"] = avg_metrics
    print("--- Average Metrics ---")
    print(json.dumps(avg_metrics, indent=2))

    if should_calc_p_mrr:
        p_mrr = compute_followir_p_mrr(original_run, new_run, original_qrels, new_qrels)
        all_metrics["p_mrr"] = p_mrr
        print("--- p-MRR (vs original qrels) ---")
        print(f"{p_mrr:.2f}")

    results_file = output_dir / "evaluation_summary.json"
    with open(results_file, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nSaved detailed evaluation summary to {results_file}")

    for run_path in [original_run_path, new_run_path]:
        run_path = Path(run_path)
        retrieval_pretty_path = run_path.with_suffix(".txt")
        pretty_print_standard_format(run_path, output_file=retrieval_pretty_path)
        print(f"Saved pretty-printed run to {retrieval_pretty_path}")


if __name__ == "__main__":
    fire.Fire(do_eval_multiple_runs)
