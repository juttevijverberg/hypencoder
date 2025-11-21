from textattack.transformations import WordSwapNeighboringCharacterSwap, \
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, \
    WordSwapRandomCharacterSubstitution, WordSwapQWERTY
from textattack.augmentation import Augmenter
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import MinWordLength, StopwordModification

from tqdm import tqdm
from argparse import ArgumentParser
import random
import ir_datasets
import json


STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                      "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                      'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                      'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                      "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                      'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                      'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


class FixWordSwapQWERTY(WordSwapQWERTY):
    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = len(word) - (1 + self.skip_last_char)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = random.randrange(start_idx, end_idx + 1)
            if len(self._get_adjacent(word[i])) == 0:
                candidate_word = (
                    word[:i] + random.choice(list(self._keyboard_adjacency.keys())) + word[i + 1:]
                )
            else:
                candidate_word = (
                    word[:i] + random.choice(self._get_adjacent(word[i])) + word[i + 1:]
                )
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx + 1):
                for swap_key in self._get_adjacent(word[i]):
                    candidate_word = word[:i] + swap_key + word[i + 1 :]
                    candidate_words.append(candidate_word)

        return candidate_words


def read_query_lines(path_to_query):
    """Read queries from a TSV file."""
    query_lines = []
    with open(path_to_query, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading query"):
        qid, query = line.strip().split("\t")
        query_lines.append((qid, query))
    return query_lines


def load_queries_from_ir_datasets(ir_dataset_name):
    """Load queries from ir_datasets."""
    print(f"Loading queries from ir_datasets: {ir_dataset_name}")
    dataset = ir_datasets.load(ir_dataset_name)
    query_lines = []
    
    for query in tqdm(dataset.queries_iter(), desc="Loading queries"):
        query_id = query.query_id if hasattr(query, 'query_id') else query['query_id']
        query_text = query.text if hasattr(query, 'text') else query['text']
        query_lines.append((query_id, query_text))
    
    return query_lines


def write_query_file(qids, queries, output_path, output_format='tsv'):
    """
    Write queries to file in either TSV or JSONL format.
    
    Args:
        qids: List of query IDs
        queries: List of query texts
        output_path: Path to save the file
        output_format: Either 'tsv' or 'jsonl'
    """
    if output_format == 'tsv':
        query_lines = []
        for i in range(len(qids)):
            query_lines.append(str(qids[i]) + "\t" + queries[i] + "\n")
        with open(output_path, "w") as f:
            f.writelines(query_lines)
    elif output_format == 'jsonl':
        with open(output_path, "w") as f:
            for i in range(len(qids)):
                query_obj = {
                    'id': str(qids[i]),
                    'text': queries[i]
                }
                f.write(json.dumps(query_obj) + "\n")
    else:
        raise ValueError(f"Unknown output format: {output_format}. Must be 'tsv' or 'jsonl'")


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_file', type=str, help='Path to query TSV file (qid<tab>query format)')
    parser.add_argument('--ir_dataset_name', type=str, help='IR dataset name (e.g., "msmarco-passage/trec-dl-2019", "trec-covid")')
    parser.add_argument('--save_to', required=True, help='Output path to save typo queries')
    parser.add_argument('--output_format', type=str, default='jsonl', choices=['tsv', 'jsonl'], 
                       help='Output format: tsv (qid<tab>query) or jsonl (for retrieve.py). Default: jsonl')
    args = parser.parse_args()

    # Check that exactly one of query_file or ir_dataset_name is provided
    if (args.query_file is None) == (args.ir_dataset_name is None):
        parser.error("Must specify exactly one of --query_file or --ir_dataset_name")

    # Load queries from either file or ir_datasets
    if args.query_file:
        query_lines = read_query_lines(args.query_file)
    else:
        query_lines = load_queries_from_ir_datasets(args.ir_dataset_name)

    transformation = CompositeTransformation([
        WordSwapRandomCharacterDeletion(),
        WordSwapNeighboringCharacterSwap(),
        WordSwapRandomCharacterInsertion(),
        WordSwapRandomCharacterSubstitution(),
        FixWordSwapQWERTY(),
    ])
    constraints = [MinWordLength(3), StopwordModification(STOPWORDS)]
    augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0)
    qids = []
    typo_queires = []
    for qid, query in tqdm(query_lines, desc="Making typo queries"):
        while True:
            typo_query = augmenter.augment(query)[0]
            if typo_query != query and typo_query.lower() == query:
                continue
            break
        typo_query = typo_query.lower()
        qids.append(qid)
        typo_queires.append(typo_query)
    
    write_query_file(qids, typo_queires, args.save_to, output_format=args.output_format)
    print(f"Saved {len(qids)} typo queries to {args.save_to} in {args.output_format} format")


if __name__ == '__main__':
    main()