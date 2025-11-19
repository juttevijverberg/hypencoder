from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ir_measures
import json

from hypencoder_cb.utils.jsonl_utils import JsonlReader


DEFAULT_METRICS = [
    "nDCG@10",
    "nDCG@5",
    "P@10",
    "P@5",
    "R@10",
    "MRR",
    "R@1000",
    "MRR@10",
]


def pretty_print_aggregated_metrics(
    aggregated_metrics_json: str,
    metric_name_ordering: Optional[List[str]] = None,
) -> str:
    with open(aggregated_metrics_json) as f:
        aggregated_metrics = json.load(f)

    if metric_name_ordering is None:
        metric_name_ordering = [
            "nDCG@10",
            "nDCG@5",
            "P@10",
            "P@5",
            "R@10",
            "RR",
            "R@1000",
            "RR@10",
        ]

    output = ""

    for metric_name in metric_name_ordering:
        if metric_name in aggregated_metrics:
            output += f"{metric_name},"
    output += "\n"

    for metric_name in metric_name_ordering:
        if metric_name in aggregated_metrics:
            output += f"{aggregated_metrics[metric_name] * 100:.2f},"
    output += "\n"

    return output


def pretty_print_aggregated_metrics_to_file(
    aggregated_metrics_json: str,
    output_file: Optional[str] = None,
    metric_name_ordering: Optional[List[str]] = None,
) -> None:
    if output_file is None:
        output_file = Path(aggregated_metrics_json).with_suffix(".txt")

    with open(output_file, "w") as f:
        f.write(
            pretty_print_aggregated_metrics(
                aggregated_metrics_json,
                metric_name_ordering=metric_name_ordering,
            )
        )


def calculate_metrics(
    run: Dict[str, Dict[str, Number]],
    qrels: Dict[str, Dict[str, Number]],
    metric_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Number], Dict[str, Dict[str, Number]]]:
    if metric_names is None:
        metric_names = DEFAULT_METRICS

    metric_objects = [
        ir_measures.parse_measure(metric) for metric in metric_names
    ]
    aggregated_metrics = ir_measures.calc_aggregate(metric_objects, qrels, run)

    per_query_metrics = defaultdict(dict)
    for metric in ir_measures.iter_calc(metric_objects, qrels, run):
        per_query_metrics[metric.query_id][str(metric.measure)] = metric.value

    return aggregated_metrics, per_query_metrics


def calculate_metrics_to_file(
    run: Dict[str, Dict[str, Number]],
    qrels: Dict[str, Dict[str, Number]],
    output_folder: str,
    metric_names: Optional[List[str]] = None,
) -> None:
    aggregated_metrics, per_query_metrics = calculate_metrics(
        run, qrels, metric_names=metric_names
    )

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    aggregated_filename = output_folder / "aggregated_metrics.json"
    per_query_filename = output_folder / "per_query_metrics.json"

    aggregated_metrics = {str(k): v for k, v in aggregated_metrics.items()}

    with open(aggregated_filename, "w") as f:
        json.dump(aggregated_metrics, f, sort_keys=True, indent=4)

    with open(per_query_filename, "w") as f:
        json.dump(per_query_metrics, f, sort_keys=True, indent=4)

    pretty_aggregated_filename = aggregated_filename.with_suffix(".txt")
    pretty_print_aggregated_metrics_to_file(
        aggregated_filename,
        output_file=pretty_aggregated_filename,
        metric_name_ordering=metric_names,
    )

    print("Saved aggregated metrics to", aggregated_filename)
    print("Saved pretty aggregated metrics to", pretty_aggregated_filename)
    print("Saved per query metrics to", per_query_filename)

    return aggregated_filename, per_query_filename


def load_standard_format_as_run(
    input_jsonl: str,
    score_key: str = "score",
) -> Dict[str, Dict[str, Number]]:
    """
    Load the standard format as a run.

    Args:
        input_jsonl (str): The input jsonl file.

    Returns:
        Dict[str, Dict[str, Number]]: The run.
    """
    with JsonlReader(input_jsonl) as reader:
        run = {}
        for line in reader:
            query_id = line["query"]["id"]
            run[query_id] = {
                str(item["id"]): item[score_key] for item in line["items"]
            }

    return run


def pretty_print_standard_format(
    standard_format_jsonl: str,
    output_file: str,
    score_key: str = "score",
) -> None:
    with JsonlReader(standard_format_jsonl) as reader:
        with open(output_file, "w") as f:
            for line in reader:
                query_id = line["query"]["id"]
                query_text = line["query"]["content"]
                f.write(f"Query: {query_text} ({query_id})\n")
                for i, item in enumerate(
                    sorted(
                        line["items"],
                        key=lambda x: x[score_key],
                        reverse=True,
                    )
                ):
                    item_id = item["id"]
                    item_text = item["content"]
                    item_score = item[score_key]
                    f.write(
                        f"\t{i + 1}. {item_text} ({item_id}) - {item_score}\n"
                    )
                f.write("\n")
                f.write("-" * 80)
                f.write("\n")
                f.write("\n")


def compute_p_mrr(
    orig_run: Dict[str, Dict[str, Number]],
    new_run: Dict[str, Dict[str, Number]],
    orig_qrels: Dict[str, Dict[str, Number]],
    new_qrels: Dict[str, Dict[str, Number]],
    scale_100: bool = False,
) -> float:
    """
    Computes p-MRR as defined in the paper:

    For each query:
      - compute MRR_og and MRR_new,
      - for each *changed relevant document* d:
          p-MRR(d,q) = MRR_og/MRR_new - 1       if R_og > R_new
                     = 1 - MRR_new/MRR_og       otherwise
      - average p-MRR(d,q) over changed relevant docs in that query.
    Finally, macro-average over queries.
    """
    mrr_measure = ir_measures.MRR

    og_mrr_scores = {
        m.query_id: m.value
        for m in ir_measures.iter_calc([mrr_measure], orig_qrels, orig_run)
    }
    new_mrr_scores = {
        m.query_id: m.value
        for m in ir_measures.iter_calc([mrr_measure], new_qrels, new_run)
    }

    per_query_p_mrr = []

    # consider all queries that have any qrels
    all_qids = set(orig_qrels.keys()) | set(new_qrels.keys())

    for qid in all_qids:
        mrr_og = og_mrr_scores.get(qid, 0.0)
        mrr_new = new_mrr_scores.get(qid, 0.0)

        # build rankings for this query
        orig_ranked = sorted(
            orig_run.get(qid, {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        new_ranked = sorted(
            new_run.get(qid, {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        orig_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(orig_ranked)}
        new_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(new_ranked)}

        # relevant documents (in either qrels)
        rel_docs = set(orig_qrels.get(qid, {}).keys()) | set(
            new_qrels.get(qid, {}).keys()
        )

        query_scores = []

        for doc_id in rel_docs:
            r_og = orig_ranks.get(doc_id, float("inf"))
            r_new = new_ranks.get(doc_id, float("inf"))

            # only consider *changed* relevant docs
            if r_og == r_new:
                continue

            if r_og > r_new:
                # new rank is better -> use first branch
                if mrr_new > 0:
                    score = (mrr_og / mrr_new) - 1.0
                else:
                    # undefined in formula; treat as worst possible change
                    score = -1.0
            else:
                # original rank is better or new is tied/worse -> second branch
                if mrr_og > 0:
                    score = 1.0 - (mrr_new / mrr_og)
                else:
                    # undefined in formula; treat as best possible change
                    score = 1.0

            query_scores.append(score)

        if query_scores:
            per_query_p_mrr.append(sum(query_scores) / len(query_scores))

    if not per_query_p_mrr:
        print("WARNING: per_query_p_mrr is empty!")
        return 0.0

    p_mrr = sum(per_query_p_mrr) / len(per_query_p_mrr)
    return p_mrr * 100.0 if scale_100 else p_mrr