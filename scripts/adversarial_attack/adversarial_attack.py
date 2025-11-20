from textattack.augmentation import CharSwapAugmenter

# 1. Build the augmenter
augmenter = CharSwapAugmenter()  # you can pass kwargs, but defaults are fine to start

# 2. Example list of queries
queries = [
    "how many states are there in india",
    "when do concussion symptoms appear",
    "what is hypencoder used for",
]

# 3. Generate one typo’d version per query
noisy_queries = [augmenter.augment(q)[0] for q in queries]

for q, nq in zip(queries, noisy_queries):
    print("ORIG:", q)
    print("NOISY:", nq)
    print()