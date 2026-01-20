from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

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

    #Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

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
                #query_text = line["query"]["content"]
                query_text = line["query"].get("content", "N/A")
                f.write(f"Query: {query_text} ({query_id})\n")
                for i, item in enumerate(
                    sorted(
                        line["items"],
                        key=lambda x: x[score_key],
                        reverse=True,
                    )
                ):
                    item_id = item["id"]
                    item_text = item.get("content", "N/A")
                    item_score = item[score_key]
                    f.write(
                        f"\t{i + 1}. {item_text} ({item_id}) - {item_score}\n"
                    )
                f.write("\n")
                f.write("-" * 80)
                f.write("\n")
                f.write("\n")

def calculate_pmrr(original_run, new_run, changed_qrels):
    changes = []
    for qid in changed_qrels.keys():
        if qid + "-og" not in original_run or qid + "-changed" not in new_run:
            logging.warning(f"Query {qid} not found in the runs for calculating p-MRR")
            continue
        original_qid_run = original_run[qid + "-og"]
        new_qid_run = new_run[qid + "-changed"]
        for idx, changed_doc in enumerate(changed_qrels[qid]):
            original_rank, original_score = get_rank_from_dict(
                original_qid_run, changed_doc
            )
            new_rank, new_score = get_rank_from_dict(new_qid_run, changed_doc)
            change = int(original_rank - new_rank)
            changes.append(
                {
                    "qid": qid,
                    "doc_id": changed_doc,
                    "change": change,
                    "relevance": 0,
                    "og_rank": original_rank,
                    "new_rank": new_rank,
                    "og_score": original_score,
                    "new_score": new_score,
                }
            )

    # we now have a DF of [qid, doc_id, change] to run our calculations with
    changes_df = pd.DataFrame(changes)
    changes_df["p-MRR"] = changes_df.apply(lambda x: rank_score(x), axis=1)
    qid_wise = changes_df.groupby("qid").agg({"p-MRR": "mean"})
    return qid_wise["p-MRR"].mean()

def rank_score(x: dict[str, float]) -> float:
    if x["og_rank"] >= x["new_rank"]:
        return ((1 / x["og_rank"]) / (1 / x["new_rank"])) - 1
    else:
        return 1 - ((1 / x["new_rank"]) / (1 / x["og_rank"]))

def get_rank_from_dict(
    dict_of_results: dict[str, float], doc_id: str
) -> tuple[int, float]:
    tuple_of_id_score = dict_of_results.items()
    sorted_by_score = sorted(tuple_of_id_score, key=lambda x: x[1], reverse=True)
    for i, (id, score) in enumerate(sorted_by_score):
        if id == doc_id:
            return i + 1, score

    return len(sorted_by_score) + 1, 0

def compute_followir_p_mrr(
    orig_run: Dict[str, Dict[str, Number]],
    new_run: Dict[str, Dict[str, Number]],
    orig_qrels: Dict[str, Dict[str, Number]],
    new_qrels: Dict[str, Dict[str, Number]],
    scale_100: bool = True,
) -> float:
    """
    Compute p-MRR for FollowIR datasets using the calculate_pmrr function.
    
    This function transforms the standard format (separate qrels) into the 
    format expected by calculate_pmrr (which expects "-og" and "-changed" suffixes).
    """
    # Combine qrels with "-og" and "-changed" suffixes to match calculate_pmrr expectations
    # combined_qrels = {}
    # for qid, docs in orig_qrels.items():
    #     combined_qrels[qid + "-og"] = docs
    # for qid, docs in new_qrels.items():
    #     combined_qrels[qid + "-changed"] = docs
    
    # Combine runs with "-og" and "-changed" suffixes
    orig_run_suffix = {}
    for qid, docs in orig_run.items():
        orig_run_suffix[qid + "-og"] = docs
    
    new_run_suffix = {}
    for qid, docs in new_run.items():
        new_run_suffix[qid + "-changed"] = docs
    
    # Determine changed documents (those with different relevance between og and changed qrels)
    changed_qrels = {}
    for qid in orig_qrels.keys():
        orig_docs = set(orig_qrels[qid].keys())
        new_docs = set(new_qrels.get(qid, {}).keys())
        # Changed docs are those that appear in either qrels but with different status
        changed = []
        for doc in orig_docs | new_docs:
            if orig_qrels[qid].get(doc, -1) != new_qrels.get(qid, {}).get(doc, -1):
                changed.append(doc)
        if changed:
            changed_qrels[qid] = changed
    
    # Use the imported calculate_pmrr function
    p_mrr = calculate_pmrr(orig_run_suffix, new_run_suffix, changed_qrels)
    
    return p_mrr * 100.0 if scale_100 else p_mrr 