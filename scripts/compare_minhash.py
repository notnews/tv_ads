"""Compare the exact n-gram index with sentence-level MinHash LSH (datasketch) on the same split.

Usage: uv run --group compare python scripts/compare_minhash.py <pages_dir> <vanderbilt.csv.gz>

Each sentence of four or more tokens becomes a MinHash of its word 3-grams (64 permutations). A sentence's
repetition count is the number of other air dates among its LSH neighbors (Jaccard threshold 0.5 or 0.7).
Every token of the sentence gets that count, and the same thresholds, tuning and test scoring as
``tv-ads validate`` follow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH

from tv_ads.cli import load_items
from tv_ads.evaluate import (
    Prepared,
    assign_splits,
    boundary_distance,
    estimate_offsets,
    prepare,
    scores,
    sentence_table,
    sentence_times,
    tune,
)
from tv_ads.index import Vocabulary, build_index
from tv_ads.parse import tokenize
from tv_ads.vanderbilt import load_rundowns


def shingles(tokens: list[str]) -> list[bytes]:
    return [" ".join(tokens[i : i + 3]).encode() for i in range(len(tokens) - 2)]


def minhash_prepared(items, rundowns, offsets, threshold, split):
    keys, sets, counts = [], [], {}
    for item in items:
        for j, sentence in enumerate(item.sentences):
            tokens = tokenize(sentence.text)
            counts[(item.identifier, j)] = len(tokens)
            if len(tokens) >= 4:
                keys.append((item.identifier, j, item.aired))
                sets.append(shingles(tokens))
    hashes = MinHash.bulk(sets, num_perm=64)
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    with lsh.insertion_session() as session:
        for key, mh in zip(keys, hashes, strict=True):
            session.insert(key, mh, check_duplication=False)
    other = {(k[0], k[1]): len({x[2] for x in lsh.query(mh)} - {k[2]}) for k, mh in zip(keys, hashes, strict=True)}
    out = {"validation": [], "test": []}
    for item in items:
        rundown = rundowns.get((item.network, item.aired))
        if rundown is None or not item.sentences:
            continue
        owner = np.repeat(
            np.arange(len(item.sentences)), [counts[(item.identifier, j)] for j in range(len(item.sentences))]
        )
        best = np.array([other.get((item.identifier, j), 0) for j in owner], dtype=np.int32)
        times = sentence_times(item, offsets.get(item.network, 0))
        prepared = Prepared(item, best, owner, rundown.kind_at(times), boundary_distance(rundown, times), rundown)
        out[split[item.identifier]].append(prepared)
    return out


def main() -> None:
    pages, vanderbilt = Path(sys.argv[1]), sys.argv[2]
    items = [i for i in load_items([pages], 8) if i.aired is not None]
    rundowns = load_rundowns(vanderbilt)
    split = assign_splits(items)
    validation = [i for i in items if split[i.identifier] == "validation"]
    offsets = estimate_offsets(validation, rundowns)
    vocabulary = Vocabulary()
    index = build_index(items, 8, vocabulary)
    params, _ = tune(prepare(validation, rundowns, [index], vocabulary, offsets))
    test = [i for i in items if split[i.identifier] == "test"]
    table = sentence_table(prepare(test, rundowns, [index], vocabulary, offsets), params)
    print("ngram8", params, scores(table))
    for threshold in (0.5, 0.7):
        halves = minhash_prepared(items, rundowns, offsets, threshold, split)
        params, _ = tune(halves["validation"])
        print(f"minhash{threshold}", params, scores(sentence_table(halves["test"], params)))


if __name__ == "__main__":
    main()
