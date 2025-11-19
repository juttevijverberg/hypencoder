import json
import argparse
from pathlib import Path

def convert_followir(input_file, output_file, positive_type="positive", use_instruction=False):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_idx, line in enumerate(infile):
            try:
                sample = json.loads(line)
                query_text = sample.get("query", "")
                
                if use_instruction and sample.get("only_instruction"):
                    query_text = f"{query_text} {sample['only_instruction']}"
                
                items = []
                for passage in sample.get("positive_passages", []):
                    item = {"id": passage.get("docid"), "content": passage.get("text"), "type": positive_type}
                    if passage.get("followir_score") is not None:
                        item["score"] = passage["followir_score"]
                    items.append(item)
                
                for passage in sample.get("negative_passages", []) + sample.get("new_negatives", []):
                    items.append({"id": passage.get("docid"), "content": passage.get("text"), "type": "negative"})
                
                if not items:
                    continue
                
                output_sample = {"query": {"id": sample.get("query_id"), "content": query_text}, "items": items}
                outfile.write(json.dumps(output_sample) + '\n')
                converted_count += 1
                
                if converted_count % 100 == 0:
                    print(f"Converted {converted_count} samples...")
                    
            except Exception as e:
                print(f"Error at line {line_idx + 1}: {e}")
    
    print(f"Conversion complete! Total: {converted_count}")


def convert_tot(data_dir, output_file, positive_type="positive"):
    data_path = Path(data_dir)
    queries_file = data_path / "queries.json"
    documents_file = data_path / "documents.json"
    qrels_file = data_path / "qrels.txt"
    hard_negatives_file = data_path / "bm25_hard_negatives_all.json"
    
    for file in [queries_file, documents_file, qrels_file, hard_negatives_file]:
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
    
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            query = json.loads(line)
            query_text = f"{query.get('title', '')} {query.get('description', '')}".strip()
            queries[query.get('id')] = query_text
    
    documents = {}
    with open(documents_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            documents[doc.get('id')] = doc_text
    
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, _, doc_id = parts[0], parts[1], parts[2]
                if query_id not in qrels:
                    qrels[query_id] = []
                qrels[query_id].append(doc_id)
    
    with open(hard_negatives_file, 'r') as f:
        hard_negatives = json.load(f)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    
    with open(output_file, 'w') as outfile:
        for query_id, query_text in queries.items():
            if query_id not in qrels:
                continue
            
            items = []
            for doc_id in qrels[query_id]:
                if doc_id in documents:
                    items.append({"id": doc_id, "content": documents[doc_id], "type": positive_type})
            
            if query_id in hard_negatives:
                for neg_id in hard_negatives[query_id]:
                    if neg_id in documents:
                        items.append({"id": neg_id, "content": documents[neg_id], "type": "hard_negative"})
            
            if items:
                output_sample = {"query": {"id": query_id, "content": query_text}, "items": items}
                outfile.write(json.dumps(output_sample) + '\n')
                converted_count += 1
                
                if converted_count % 100 == 0:
                    print(f"Converted {converted_count} samples...")
    
    print(f"Conversion complete! Total: {converted_count}")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to contrastive learning format")
    parser.add_argument("--dataset", type=str, choices=["FollowIR", "TOT"], required=True)
    parser.add_argument("--input", type=str, help="Input file (required for FollowIR) or data directory (required for TOT)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--positive_type", type=str, default="positive")
    parser.add_argument("--use_instruction", action="store_true", default=False)
    
    args = parser.parse_args()
    
    if args.dataset.lower() == "followir":
        if not args.input:
            raise ValueError("--input is required for FollowIR")
        convert_followir(args.input, args.output, args.positive_type, args.use_instruction)
    elif args.dataset.lower() == "tot":
        if not args.input:
            raise ValueError("--input is required for TOT (path to folder with queries.json, documents.json, qrels.txt, bm25_hard_negatives_all.json)")
        convert_tot(args.input, args.output, args.positive_type)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")


if __name__ == "__main__":
    main()
