#!/usr/bin/env python3
import argparse
import json

def p_mrr(MRR_og, MRR_new, improved):
    return (MRR_og / MRR_new) - 1 if improved else 1 - (MRR_new / MRR_og)

def get_mrr(path):
    with open(path) as f:
        return json.load(f)["MRR"]

def get_first_rank(path):
    ranks = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if "first_rel_rank" in obj:
                ranks.append(obj["first_rel_rank"])
    return sum(ranks) / len(ranks) if ranks else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--og_items", required=True)
    parser.add_argument("--new_items", required=True)
    parser.add_argument("--og_metrics", required=True)
    parser.add_argument("--new_metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    MRR_og = get_mrr(args.og_metrics)
    MRR_new = get_mrr(args.new_metrics)
    R_og = get_first_rank(args.og_items)
    R_new = get_first_rank(args.new_items)
    improved = R_og and R_new and R_og > R_new

    result = {
        "p_MRR": p_mrr(MRR_og, MRR_new, improved),
        "MRR_og": MRR_og,
        "MRR_new": MRR_new,
        "R_og": R_og,
        "R_new": R_new,
        "improved": improved
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
