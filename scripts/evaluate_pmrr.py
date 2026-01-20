import json
from pathlib import Path
from typing import Optional

import fire

from hypencoder_cb.utils.data_utils import (
    load_qrels_from_ir_datasets,
    load_qrels_from_json,
)
from hypencoder_cb.utils.eval_utils import (
    compute_followir_p_mrr,
    load_standard_format_as_run,
)


def do_eval_pmrr(
    original_run_path: str,
    new_run_path: str,
    output_dir: str,
    ir_dataset_name: Optional[str] = None,
    original_qrel_json: Optional[str] = None,
    new_qrel_json: Optional[str] = None,
) -> None:
    """Calculates p-MRR between two retrieval runs.

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

    original_run = load_standard_format_as_run(original_run_path, score_key="score")
    new_run = load_standard_format_as_run(new_run_path, score_key="score")

    p_mrr = compute_followir_p_mrr(original_run, new_run, original_qrels, new_qrels)
    
    print(f"p-MRR: {p_mrr:.4f}")

    results = {"p_mrr": p_mrr}
    results_file = output_dir / "pmrr_result.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved p-MRR result to {results_file}")


if __name__ == "__main__":
    fire.Fire(do_eval_pmrr)
