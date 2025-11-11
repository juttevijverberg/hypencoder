# save as download_hf_dataset.py
import os
from datasets import load_dataset

def download_and_save(dataset_name: str, subsets: list[str]):
    """
    Download specified subsets of a Hugging Face dataset and save them as JSON 
    in a structured folder hierarchy under 'data/<dataset_shortname>/<subset>.json'.
    """
    dataset_shortname = dataset_name.split("/")[-1].replace("-instructions", "")
    base_dir = os.path.join("data", dataset_shortname)
    os.makedirs(base_dir, exist_ok=True)

    for subset in subsets:
        print(f"⬇️  Downloading subset '{subset}' from {dataset_name} ...")
        try:
            ds = load_dataset(dataset_name, subset)
        except Exception as e:
            print(f"⚠️  Skipping '{subset}' (not found or failed to load): {e}")
            continue

        # Save depending on dataset type
        if isinstance(ds, dict):
            for split_name, split_data in ds.items():
                file_path = os.path.join(base_dir, f"{subset}_{split_name}.json")
                print(f"💾 Saving split '{split_name}' to {file_path} ...")
                split_data.to_json(file_path)
        else:
            file_path = os.path.join(base_dir, f"{subset}.json")
            print(f"💾 Saving to {file_path} ...")
            ds.to_json(file_path)

        print(f"✅ Saved subset '{subset}' for {dataset_shortname} successfully.\n")

if __name__ == "__main__":
    dataset_names = [
        "jhu-clsp/robust04-instructions",
        "jhu-clsp/news21-instructions",
        "jhu-clsp/core17-instructions"
    ]
    subsets = ["corpus", "queries", "qrels_changed", "qrels_og", "top_ranked"]

    for dataset_name in dataset_names:
        download_and_save(dataset_name, subsets)
