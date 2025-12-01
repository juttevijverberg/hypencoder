# Adversarial Attack Jobs

This directory contains job files for generating and evaluating adversarial query attacks.

## Workflow

### 1. Generate Adversarial Queries

Run one of these jobs to generate adversarial queries:

```bash
# For TREC-DL 2019
sbatch jobs/adversarial_attack/gen_attacks_trec19.job

# For TREC-DL 2020
sbatch jobs/adversarial_attack/gen_attacks_trec20.job
```

This will generate adversarial queries for all attack types and save them to:
- `data/adversarial_attack/<dataset_name>/<attack_type>/adversarial_queries.jsonl`

### 2. Run Retrieval on Adversarial Queries

After generating queries, evaluate them:

```bash
# Evaluate all attacks for TREC-DL 2019
sbatch jobs/adversarial_attack/retrieve_attacks_trec19.job
```

Results will be saved to:
- `retrieval_outputs/adversarial/<dataset_name>/<attack_type>/`

## Attack Types

The script supports the following attack types:
- **synonym**: Replace words with synonyms
- **paraphrase**: Paraphrase using back-translation
- **naturality**: Remove random words
- **mispelling**: Introduce character-level misspellings
- **ordering**: Shuffle word order

## Custom Usage

### Generate specific attack types only:

```bash
python scripts/adversarial_attack/attacks.py \
    --ir_dataset_name="msmarco-passage/trec-dl-2019/judged" \
    --attack_types mispelling ordering \
    --output_dir=data/adversarial_attack
```

### Test with limited queries:

```bash
python scripts/adversarial_attack/attacks.py \
    --ir_dataset_name="msmarco-passage/trec-dl-2019/judged" \
    --attack_types all \
    --max_queries=10
```

### Run demo (no arguments):

```bash
python scripts/adversarial_attack/attacks.py
```

This will run all attacks on the default sample queries and print results.
