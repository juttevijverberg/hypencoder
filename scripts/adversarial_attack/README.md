# Adversarial Attack Pipeline for Hypencoder

This directory contains scripts for evaluating the robustness of Hypencoder models against adversarial attacks on queries.

## Overview

The pipeline generates adversarial queries using TextAttack and evaluates how these perturbations affect retrieval performance.

## Pipeline Steps

### 1. Generate Adversarial Queries

Use `adversarial_attack.py` to create adversarial versions of queries:

```bash
python scripts/adversarial_attack/adversarial_attack.py \
  --model_name_or_path="jfkback/hypencoder.6_layer" \
  --encoded_item_path="$HOME/hypencoder/encoded_items/trec-dl-2019" \
  --ir_dataset_name="msmarco-passage/trec-dl-2019/judged" \
  --attack_method="textfooler" \
  --output_path="adversarial_queries/trec-dl-2019_textfooler.jsonl" \
  --query_max_length=64 \
  --dtype=fp16
```

Or use the job file:
```bash
sbatch jobs/adversarial/attack_trec19.job
```

### 2. Run Retrieval with Adversarial Queries

Use the standard retrieval script with adversarial queries:

```bash
python hypencoder_cb/inference/retrieve.py \
  --model_name_or_path="jfkback/hypencoder.6_layer" \
  --encoded_item_path="$HOME/hypencoder/encoded_items/trec-dl-2019" \
  --output_dir="retrieval_outputs/trec-dl-2019-adversarial" \
  --query_jsonl="adversarial_queries/trec-dl-2019_textfooler.jsonl" \
  --ir_dataset_name="msmarco-passage/trec-dl-2019/judged" \
  --query_id_key="query_id" \
  --query_text_key="adversarial_text" \
  --query_max_length=64 \
  --dtype=fp16 \
  --metric_names nDCG@10 RR@10 R@1000 P@10
```

Or use the job file:
```bash
sbatch jobs/adversarial/retrieve_adv_trec19.job
```

### 3. Compare Results

Compare original and adversarial performance:

```bash
python scripts/adversarial_attack/compare_metrics.py \
  --original_metrics="retrieval_outputs/trec-dl-2019/metrics/aggregated_metrics.json" \
  --adversarial_metrics="retrieval_outputs/trec-dl-2019-adversarial/metrics/aggregated_metrics.json" \
  --adversarial_queries="adversarial_queries/trec-dl-2019_textfooler.jsonl" \
  --output_path="adversarial_results/trec-dl-2019_comparison.json"
```

## Attack Methods

The pipeline supports multiple attack methods from TextAttack:

- **textfooler** (default): Word-level substitution attack using word embeddings
- **bae**: BERT-based adversarial examples
- **bert-attack**: Context-aware word substitution using BERT
- **deepwordbug**: Character-level perturbations

### Attack Method Details

**TextFooler** (Recommended for IR):
- Replaces words with synonyms to maintain semantic similarity
- Checks word importance and replaces less important words first
- Good balance between attack success and query readability

**BAE**:
- Uses BERT masked language model for context-aware replacements
- More sophisticated than simple synonym replacement
- May generate more natural-looking adversarial queries

**BERT-Attack**:
- Similar to BAE but with different selection strategy
- Uses BERT to find replacements that minimize semantic similarity

**DeepWordBug**:
- Character-level perturbations (insertions, deletions, swaps)
- Creates typos and misspellings
- Tests robustness to user input errors

## Parameters

### adversarial_attack.py

**Required:**
- `--model_name_or_path`: Path to Hypencoder model
- `--encoded_item_path`: Path to pre-encoded documents
- `--ir_dataset_name` OR `--query_jsonl`: Source of queries
- `--output_path`: Where to save adversarial queries

**Optional:**
- `--attack_method`: Attack type (default: textfooler)
- `--max_queries`: Limit number of queries (for testing)
- `--query_max_length`: Max query length (default: 64)
- `--dtype`: Model dtype (fp16, fp32, bf16)
- `--top_k`: Number of docs to consider for scoring (default: 10)

### compare_metrics.py

**Required:**
- `--original_metrics`: Path to original metrics JSON
- `--adversarial_metrics`: Path to adversarial metrics JSON

**Optional:**
- `--adversarial_queries`: Path to adversarial queries (for attack stats)
- `--output_path`: Where to save comparison results

## Output Format

### Adversarial Queries (JSONL)

Each line contains:
```json
{
  "query_id": "1030303",
  "original_text": "who is aziz hashim",
  "adversarial_text": "who is aziz hashim [perturbed version]",
  "attack_success": true,
  "num_words_changed": 2,
  "original_score": 0.95,
  "perturbed_score": 0.42
}
```

### Comparison Results (JSON)

```json
{
  "nDCG@10": {
    "original": 0.7418,
    "adversarial": 0.6523,
    "absolute_difference": -0.0895,
    "relative_difference_pct": -12.07
  },
  "attack_statistics": {
    "total_queries": 43,
    "successful_attacks": 38,
    "success_rate_pct": 88.4,
    "avg_words_changed": 2.3
  }
}
```

## Installation

Install TextAttack with TensorFlow support:

```bash
conda activate hype
pip install textattack[tensorflow]
```

## Example Workflow

Complete workflow for TREC-DL-2019:

```bash
# Step 1: Generate adversarial queries
sbatch jobs/adversarial/attack_trec19.job

# Wait for job to complete, then:

# Step 2: Run retrieval with adversarial queries
sbatch jobs/adversarial/retrieve_adv_trec19.job

# Wait for job to complete, then:

# Step 3: Compare results
python scripts/adversarial_attack/compare_metrics.py \
  --original_metrics="retrieval_outputs/trec-dl-2019/metrics/aggregated_metrics.json" \
  --adversarial_metrics="retrieval_outputs/trec-dl-2019-adversarial/metrics/aggregated_metrics.json" \
  --adversarial_queries="adversarial_queries/trec-dl-2019_textfooler.jsonl"
```

## Tips

1. **Start small**: Use `--max_queries=10` to test the pipeline quickly
2. **Memory considerations**: Attacks can be memory-intensive; use `--dtype=fp16` to reduce memory usage
3. **Time estimates**: Attacking 50 queries typically takes 1-2 hours depending on attack method
4. **Baseline comparison**: Also test attacks on the bi-encoder baseline (`TextDualEncoder`) to compare robustness

## Troubleshooting

**Out of Memory during attacks:**
- Reduce `--top_k` (default: 10)
- Use `--dtype=fp16`
- Reduce `--max_queries`
- Request more GPU memory in job script

**Attack takes too long:**
- Use faster attack methods (textfooler is faster than bert-attack)
- Reduce `--max_queries` for initial testing
- Consider running attacks in parallel for different query subsets

**Low attack success rate:**
- Try different attack methods
- Check if queries are already robust (short queries are harder to attack)
- Verify model is loaded correctly

## Citation

If using TextAttack, please cite:
```
@misc{morris2020textattack,
    title={TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP},
    author={John X. Morris and Eli Lifland and Jin Yong Yoo and Jake Grigsby and Di Jin and Yanjun Qi},
    year={2020},
    eprint={2005.05909},
    archivePrefix={arXiv}
}
```
