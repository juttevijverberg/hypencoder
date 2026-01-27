"""Retrieve BE-base (TextDualEncoder) results using Faiss search."""
import argparse
import json
from pathlib import Path

import ir_datasets
import torch
from transformers import AutoTokenizer

from hypencoder_cb.modeling.hypencoder_bebase import TextDualEncoder

from bebase_faiss_utils import (
	encode_queries,
	evaluate_with_ir_measures,
	load_encoded_corpus,
	retrieve_faiss,
)

def parse_args():
	parser = argparse.ArgumentParser(
		description="Retrieve BE-base results from pre-encoded documents."
	)
	parser.add_argument(
		"--model_name_or_path",
		type=str,
		default="models/be_base",
		help="Path to the BE-base TextDualEncoder checkpoint.",
	)
	parser.add_argument(
		"--ir_dataset_name",
		type=str,
		required=True,
		help="IR dataset name (e.g., 'trec-tot/2023/dev').",
	)
	parser.add_argument(
		"--encoded_docs_path",
		type=str,
		required=True,
		help="Path to pre-encoded documents.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		required=True,
		help="Directory to save retrieval results and metrics.",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=64,
		help="Batch size for encoding queries.",
	)
	parser.add_argument(
		"--query_max_length",
		type=int,
		default=512,
		help="Maximum token length for queries.",
	)
	parser.add_argument(
		"--top_k",
		type=int,
		default=1000,
		help="Number of documents to retrieve per query.",
	)
	parser.add_argument(
		"--metric_names",
		nargs="+",
		default=["nDCG@10", "RR", "nDCG@1000", "R@1000", "AP", "nDCG@5"],
		help="IR metrics to compute.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	print("\n--- Parsed Arguments (Key-Value) ---")
	for key, value in vars(args).items():
		print(f"  {key}: {value}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_dir = output_dir / "metrics"
	metrics_dir.mkdir(exist_ok=True)

	print(f"Loading tokenizer and TextDualEncoder from {args.model_name_or_path}")
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	model = TextDualEncoder.from_pretrained(args.model_name_or_path)
	model.to(device)
	model.eval()

	query_encoder = model.query_encoder

	print(f"Loading dataset: {args.ir_dataset_name}")
	dataset = ir_datasets.load(args.ir_dataset_name)

	doc_ids, doc_embeddings = load_encoded_corpus(args.encoded_docs_path)

	queries = list(dataset.queries_iter())
	query_ids, query_embeddings = encode_queries(
		queries,
		args.batch_size,
		tokenizer,
		query_encoder,
		device,
		args.query_max_length,
	)

	results = retrieve_faiss(query_embeddings, doc_embeddings, doc_ids, args.top_k)

	output_file = output_dir / "retrieved_items.txt"
	print(f"Saving retrieval results to {output_file}")
	with open(output_file, "w") as f:
		for query_idx, doc_scores in results.items():
			query_id = query_ids[query_idx]
			for rank, (doc_id, score) in enumerate(doc_scores, 1):
				f.write(f"{query_id} Q0 {doc_id} {rank} {score} BE-base\n")

	print("Calculating metrics...")
	qrels = dataset.qrels_iter()
	metrics = evaluate_with_ir_measures(
		results,
		query_ids,
		qrels,
		args.metric_names,
	)

	metrics_file = metrics_dir / "aggregated_metrics.json"
	with open(metrics_file, "w") as f:
		json.dump(metrics, f, indent=4)

	print(f"\nMetrics saved to {metrics_file}")
	print("\nResults:")
	for metric, value in metrics.items():
		print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
	main()
