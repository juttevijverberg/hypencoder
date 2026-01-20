import csv
import hashlib
import json
import os
import pathlib
import random
import fire

from collections import defaultdict
from typing import Dict, List
from datasets import load_dataset
from tqdm import tqdm
from hypencoder_cb.utils.jsonl_utils import JsonlReader, JsonlWriter


def stable_hex_hashing(input: str) -> str:
    return str(hashlib.sha224(input.encode()).hexdigest())[:20]


def load_qrels(qrel_file: str) -> Dict[str, Dict[str, int]]:
    # Qrel file is a tsv with 4 columns

    qrels = {}
    with open(qrel_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            query_id, _, doc_id, relevance = line
            relevance = int(relevance)
            qrels.setdefault(query_id, {})[doc_id] = relevance

    return qrels


def tomt_to_standard(
    qrel_file: str,
    query_file: str,
    hard_negative_file: str,
    document_files: List[str],
    output_file: str,
):
    # Open qrels which is tab separated
    qrels = load_qrels(qrel_file)

    queries = {}
    with JsonlReader(query_file) as reader:
        for line in reader:
            queries[line["id"]] = line

    documents = {}
    for document_file in document_files:
        with JsonlReader(document_file) as reader:
            for line in reader:
                documents[line["id"]] = line

    # Dict with query_id -> [doc_id]
    with open(hard_negative_file, "r") as f:
        hard_negatives = json.load(f)

    with JsonlWriter(output_file) as writer:
        for query_id in queries:
            query = queries[query_id]
            query_title = query["title"]
            query_text = query["description"]

            if query_id not in qrels:
                continue

            relevant_docs = qrels[query_id]

            items = []

            for doc_id in relevant_docs:
                # if doc_id not in documents:
                #     continue

                items.append(
                    {
                        "content": f'{documents[doc_id]["title"]}.\n{documents[doc_id]["text"]}',
                        "label": relevant_docs[doc_id],
                        "id": doc_id,
                    }
                )

            if query_id in hard_negatives:
                for doc_id in hard_negatives[query_id]:
                    # if doc_id not in documents:
                    #     continue

                    items.append(
                        {
                            "content": f'{documents[doc_id]["title"]}.\n{documents[doc_id]["text"]}',
                            "label": 0,
                            "id": doc_id,
                        }
                    )

            if len(items) == 0:
                continue

            writer.write(
                {
                    "query": {
                        "content": f"{query_title}.\n{query_text}",
                        "id": query_id,
                    },
                    "items": items,
                }
            )

def run_tomt():
    hard_negative_file = "data/tot/raw/Books/bm25_hard_negatives_all.json"
    document_file = [
        "data/tot/raw/Books/documents.json",
        "data/tot/raw/Books/hard_negative_documents.json",
        "data/tot/raw/Books/negative_documents.json",
    ]

    for split in ["train", "validation", "test"]:
        qrel_file = f"data/tot/raw/Books/splits/{split}/qrels.txt"
        query_file = f"data/tot/raw/Books/splits/{split}/queries.json"
        output_file = f"data/tot/{split}/books.standard.jsonl"
        tomt_to_standard(
            qrel_file,
            query_file,
            hard_negative_file,
            document_file,
            output_file,
        )

    hard_negative_file = "data/tot/raw/Movies/bm25_hard_negatives_all.json"
    document_file = [
        "data/tot/raw/Movies/documents.json",
        "data/tot/raw/Movies/hard_negative_documents.json",
        "data/tot/raw/Movies/negative_documents.json",
    ]

    for split in ["train", "validation", "test"]:
        qrel_file = f"data/tot/raw/Movies/splits/{split}/qrels.txt"
        query_file = f"data/tot/raw/Movies/splits/{split}/queries.json"
        output_file = f"data/tot/{split}/movies.standard.jsonl"
        tomt_to_standard(
            qrel_file,
            query_file,
            hard_negative_file,
            document_file,
            output_file,
        )

    # Join movies and books
    for split in ["train", "validation", "test"]:
        movies_file = f"data/tot/{split}/movies.standard.jsonl"
        books_file = f"data/tot/{split}/books.standard.jsonl"
        output_file = f"data/tot/{split}/all.standard.jsonl"

        with JsonlWriter(output_file) as writer:
            with JsonlReader(movies_file) as reader:
                for line in reader:
                    writer.write(line)

            with JsonlReader(books_file) as reader:
                for line in reader:
                    writer.write(line)


def msmarco_with_instruct_to_standard():
    ds = load_dataset(
        "samaya-ai/msmarco-w-instructions", split="train", streaming=False
    )

    output_file = "data/msmarco_with_instructions/standard/train.all_queries.jsonl"
    total_negatives = 5

    with JsonlWriter(output_file) as writer:
        for i, line in enumerate(tqdm(ds)):
            # if not line["has_instruction"]:
            #     continue

            items = []
            for passage in line["positive_passages"]:
                items.append(
                    {
                        "content": passage["text"],
                        "label": 1,
                        "id": passage["docid"],
                    }
                )

            negative_items = []
            for passage in line.get("new_negatives", 0):
                negative_items.append(
                    {
                        "content": passage["text"],
                        "label": 0,
                        "id": passage["docid"],
                    }
                )

            additional_negatives = total_negatives - len(negative_items)
            random.shuffle(line["negative_passages"])
            for passage in line["negative_passages"][:additional_negatives]:
                negative_items.append(
                    {
                        "content": passage["text"],
                        "label": 0,
                        "id": passage["docid"],
                    }
                )

            items.extend(negative_items)

            writer.write(
                {
                    "query": {
                        "content": line["query"],
                        "id": line["query_id"],
                    },
                    "items": items,
                }
            )

def followir_to_standard(
    dataset_name: str,
    output_dir: str,
    given_name: str,
    include_title: bool = True,
) -> None:
    corpus = load_dataset(dataset_name, "corpus", split="corpus")
    qrels_og = load_dataset(dataset_name, "qrels_og", split="test")
    qrels_changed = load_dataset(dataset_name, "qrels_changed", split="test")
    queries = load_dataset(dataset_name, "queries", split="queries")
    top_ranked = load_dataset(dataset_name, "top_ranked", split="top_ranked")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    qrels_og_dict = defaultdict(dict)
    for line in qrels_og:
        qrels_og_dict[line["query-id"]][line["corpus-id"]] = int(line["score"])

    qrels_changed_dict = defaultdict(dict)
    for line in qrels_changed:
        qrels_changed_dict[line["query-id"]][line["corpus-id"]] = int(
            line["score"]
        )

    with open(output_dir / f"{given_name}_qrels_og.json", "w") as f:
        json.dump(qrels_og_dict, f)

    with open(output_dir / f"{given_name}_qrels_changed.json", "w") as f:
        json.dump(qrels_changed_dict, f)

    corpus_lookup = {}
    with JsonlWriter(output_dir / f"{given_name}_corpus.jsonl") as writer:
        for line in corpus:
            writer.write(
                {
                    "text": (
                        f'{line["title"]} {line["text"]}'
                        if include_title
                        else line["text"]
                    ),
                    "id": line["_id"],
                }
            )
            corpus_lookup[line["_id"]] = (
                f'{line["title"]} {line["text"]}'
                if include_title
                else line["text"]
            )

    og_queries = []
    changed_queries = []

    queries_og = {}
    queries_changed = {}

    for line in queries:
        query_id = line["_id"]
        og_query = {
            "text": f'{line["text"]} {line["instruction_og"]}',
            "id": query_id,
        }
        changed_query = {
            "text": f'{line["text"]} {line["instruction_changed"]}',
            "id": query_id,
        }

        queries_og[query_id] = og_query["text"]
        queries_changed[query_id] = changed_query["text"]

        og_queries.append(og_query)
        changed_queries.append(changed_query)

    with JsonlWriter(output_dir / f"{given_name}_queries_og.jsonl") as writer:
        for line in og_queries:
            writer.write(line)

    with JsonlWriter(
        output_dir / f"{given_name}_queries_changed.jsonl"
    ) as writer:
        for line in changed_queries:
            writer.write(line)

    top_ranked_dict = defaultdict(list)
    for line in top_ranked:
        top_ranked_dict[line["qid"]].append(line["pid"])

    with JsonlWriter(output_dir / f"{given_name}_rerank_og.jsonl") as writer:
        for query_id, passage_ids in top_ranked_dict.items():
            writer.write(
                {
                    "query": {
                        "content": queries_og[query_id],
                        "id": query_id,
                    },
                    "items": [
                        {
                            "content": corpus_lookup[passage_id],
                            "id": passage_id,
                        }
                        for passage_id in passage_ids
                    ],
                }
            )

    with JsonlWriter(
        output_dir / f"{given_name}_rerank_changed.jsonl"
    ) as writer:
        for query_id, passage_ids in top_ranked_dict.items():
            writer.write(
                {
                    "query": {
                        "content": queries_changed[query_id],
                        "id": query_id,
                    },
                    "items": [
                        {
                            "content": corpus_lookup[passage_id],
                            "id": passage_id,
                        }
                        for passage_id in passage_ids
                    ],
                }
            )


def run_followir():
    dataset_names = [
        "jhu-clsp/robust04-instructions",
        "jhu-clsp/core17-instructions",
        "jhu-clsp/news21-instructions",
    ]

    for dataset_name in dataset_names:
        dataset_name_for_path = dataset_name.split("/")[-1]
        print(dataset_name)
        output_dir = f"data/followir/{dataset_name_for_path}/standard"
        followir_to_standard(
            dataset_name,
            output_dir,
            f"{dataset_name_for_path.replace('-', '_')}",
            include_title=False,
        )

if __name__ == "__main__":
    fire.Fire()
