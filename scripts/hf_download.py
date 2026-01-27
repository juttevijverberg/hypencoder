from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jfkback/hypencoder-msmarco-training-dataset",
    repo_type="dataset",
    local_dir="/scratch-shared/scur1744/data/hypencoder-msmarco-training-dataset",
    local_dir_use_symlinks=False,
    resume_download=True,
)