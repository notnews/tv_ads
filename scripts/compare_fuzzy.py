"""Does edit-tolerant matching against the catalog find ads that exact n-grams miss?

Usage: uv run --group compare python scripts/compare_fuzzy.py <pages_dir> <vanderbilt.csv.gz> [n]

The catalog is built from validation-day pages only and matched against test-day pages. The exact route
is ``tv-ads flag`` with that catalog. The fuzzy route adds one step. Every sentence of four or more words
that the exact route leaves under ``min_share`` gets candidate spots from shared word 4-grams. It is then
compared with each candidate's text by rapidfuzz ``partial_ratio`` (the best-matching window of the spot,
scored as 100 x (1 - normalized Indel distance)). At or above the threshold, its tokens count as repeated
on that spot's airings. The thresholds are tuned on validation days like everything else, and the
score threshold is chosen there too.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from tv_ads.catalogue import build_catalogue, catalogue_index
from tv_ads.cli import load_items
from tv_ads.evaluate import Prepared, assign_splits, estimate_offsets, macro, prepare, scores, sentence_table, tune
from tv_ads.index import Vocabulary, build_index, ngram_hashes
from tv_ads.parse import tokenize
from tv_ads.vanderbilt import load_rundowns


def candidates_index(spots, vocabulary):
    by_hash = defaultdict(set)
    for row in spots.itertuples():
        for h in np.unique(ngram_hashes(vocabulary.ids(tokenize(row.text)), 4)).tolist():
            by_hash[h].add(row.spot)
    keys = {row.spot: " ".join(tokenize(row.text)) for row in spots.itertuples()}
    airings = dict(zip(spots["spot"], spots["airings"], strict=True))
    return by_hash, keys, airings


def fuzzy_prepared(prepared, spots, vocabulary, threshold, min_share, cache):
    by_hash, keys, airings = cache
    out = []
    for p in prepared:
        best = p.best.copy()
        sizes = np.bincount(p.owner, minlength=len(p.item.sentences))
        hits = np.bincount(p.owner, weights=best >= 1, minlength=len(p.item.sentences))
        for j, sentence in enumerate(p.item.sentences):
            if sizes[j] < 4 or hits[j] >= min_share * sizes[j]:
                continue
            words = tokenize(sentence.text)
            found = set()
            for h in ngram_hashes(vocabulary.ids(words), 4).tolist():
                found |= by_hash.get(h, set())
            text = " ".join(words)
            scored = [(fuzz.partial_ratio(text, keys[s]), s) for s in found]
            good = [airings[s] for score, s in scored if score >= threshold]
            if good:
                best[p.owner == j] = np.maximum(best[p.owner == j], max(good))
        out.append(Prepared(p.item, best, p.owner, p.kinds, p.distance, p.rundown))
    return out


def report(name, table, seconds):
    per = {k: scores(g) for k, g in table.groupby("network")}
    cells = " ".join(f"{k} P{v['precision']:.3f} R{v['recall']:.3f} F{v['f1']:.3f}" for k, v in per.items())
    print(f"{name}: macro F1 {macro(per):.3f} | {cells} | {seconds:.1f} s")


def main() -> None:
    pages, vanderbilt = Path(sys.argv[1]), sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    items = [i for i in load_items([pages], 8) if i.aired is not None]
    rundowns = load_rundowns(vanderbilt)
    split = assign_splits(items)
    validation = [i for i in items if split[i.identifier] == "validation"]
    test = [i for i in items if split[i.identifier] == "test"]
    offsets = estimate_offsets(validation, rundowns)
    vocabulary = Vocabulary()
    corpus_params, _ = tune(
        prepare(validation, rundowns, [build_index(validation, n, vocabulary)], vocabulary, offsets)
    )
    spots = build_catalogue(validation, [build_index(validation, n, vocabulary)], vocabulary, corpus_params)
    catalogue = [catalogue_index(spots, n, vocabulary)]
    print(f"catalogue from validation days: {len(spots)} spots; n={n}")
    cache = candidates_index(spots, vocabulary)

    val_exact = prepare(validation, rundowns, catalogue, vocabulary, offsets)
    params, _ = tune(val_exact)
    start = time.perf_counter()
    test_exact = prepare(test, rundowns, catalogue, vocabulary, offsets)
    report(f"exact {params}", sentence_table(test_exact, params), time.perf_counter() - start)

    best = None
    for threshold in (80, 85, 90, 95):
        tuned, grid = tune(fuzzy_prepared(val_exact, spots, vocabulary, threshold, params.min_share, cache))
        if best is None or grid.iloc[0]["macro_f1"] > best[0]:
            best = (grid.iloc[0]["macro_f1"], threshold, tuned)
    _, threshold, tuned = best
    start = time.perf_counter()
    test_fuzzy = fuzzy_prepared(
        prepare(test, rundowns, catalogue, vocabulary, offsets), spots, vocabulary, threshold, tuned.min_share, cache
    )
    report(f"fuzzy>={threshold} {tuned}", sentence_table(test_fuzzy, tuned), time.perf_counter() - start)


if __name__ == "__main__":
    main()
