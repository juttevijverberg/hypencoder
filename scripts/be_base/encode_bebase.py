"""Encode BE-base (TextDualEncoder) documents for a given IR dataset."""
import argparse
from pathlib import Path

import ir_datasets
import torch
from transformers import AutoTokenizer

from hypencoder_cb.modeling.hypencoder_bebase import TextDualEncoder

from bebase_faiss_utils import encode_corpus, save_encoded_corpus

def parse_args():
	parser = argparse.ArgumentParser(
		description="Encode BE-base documents and persist them for reuse."
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
		"--output_path",
		type=str,
		required=True,
		help="Destination file for encoded documents (DocList format).",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=64,
		help="Batch size for encoding documents.",
	)
	parser.add_argument(
		"--doc_max_length",
		type=int,
		default=512,
		help="Maximum token length for documents.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	print("\n--- Parsed Arguments (Key-Value) ---")
	for key, value in vars(args).items():
		print(f"  {key}: {value}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	output_path = Path(args.output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	print(f"Loading tokenizer and TextDualEncoder from {args.model_name_or_path}")
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	model = TextDualEncoder.from_pretrained(args.model_name_or_path)
	model.to(device)
	model.eval()

	passage_encoder = model.passage_encoder

	print(f"Loading dataset: {args.ir_dataset_name}")
	dataset = ir_datasets.load(args.ir_dataset_name)
	corpus = list(dataset.docs_iter())

	doc_ids, doc_embeddings, doc_texts = encode_corpus(
		corpus,
		args.batch_size,
		tokenizer,
		passage_encoder,
		device,
		args.doc_max_length,
	)

	save_encoded_corpus(
		doc_texts,
		doc_ids,
		doc_embeddings,
		str(output_path),
	)


if __name__ == "__main__":
	main()
