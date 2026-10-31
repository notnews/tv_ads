"""Label each caption sentence as ad, boilerplate or program from how often its words recur on other dates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from tv_ads.index import RepetitionIndex, Vocabulary, item_tokens, ngram_hashes
from tv_ads.parse import Item

LABELS = ("ad", "boilerplate", "program")


@dataclass(frozen=True)
class Params:
    """Decision thresholds; the defaults are the values tuned on validation days (see README)."""

    min_dates: int = 1
    min_share: float = 0.3
    min_run_tokens: int = 30
    max_gap_tokens: int = 50


def _spread(values: np.ndarray, n_tokens: int, n: int) -> np.ndarray:
    """Per token, the largest value of any n-gram covering it (n-gram i covers tokens i .. i + n - 1)."""
    out = np.zeros(n_tokens, dtype=np.int32)
    for k in range(min(n, n_tokens)):
        np.maximum(out[k : k + len(values)], values, out=out[k : k + len(values)])
    return out


def token_dates(item: Item, indexes: list[RepetitionIndex], vocabulary: Vocabulary) -> tuple[np.ndarray, np.ndarray]:
    """For each token, the most other air dates any covering n-gram was seen on, over all indexes; and the
    index of the sentence each token belongs to.

    "Other" excludes the item's own date when the item is part of that index.
    """
    tokens, owner = item_tokens(item)
    ids = vocabulary.ids(tokens)
    best = np.zeros(len(tokens), dtype=np.int32)
    for index in indexes:
        dates, _, _ = index.lookup(ngram_hashes(ids, index.n))
        if item.identifier in index.identifiers:
            dates = np.maximum(dates - 1, 0)
        best = np.maximum(best, _spread(dates, len(tokens), index.n))
    return best, owner


def token_spots(item: Item, index: RepetitionIndex, vocabulary: Vocabulary) -> np.ndarray:
    """For each token, the catalogue spot of a covering n-gram (the most-aired one), or -1."""
    tokens, _ = item_tokens(item)
    out = np.full(len(tokens), -1, dtype=np.int32)
    if index.spot_ids is None:
        return out
    hashes = ngram_hashes(vocabulary.ids(tokens), index.n)
    spots = index.spots_of(hashes)
    dates, _, _ = index.lookup(hashes)
    for i in np.argsort(dates, kind="stable"):
        if spots[i] >= 0:
            out[i : i + index.n] = spots[i]
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index ranges where ``mask`` is true."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def label_from_dates(best: np.ndarray, owner: np.ndarray, n_sentences: int, params: Params) -> list[str]:
    """Turn token repetition counts into sentence labels.

    1. A sentence repeats when at least ``min_share`` of its tokens sit in an n-gram seen on ``min_dates`` or
       more other dates.
    2. Unrepeated stretches of at most ``max_gap_tokens`` tokens between two repeating sentences are filled in,
       so a garbled line inside an ad break does not split it. Token-less lines ("♪") between two repeating
       sentences are always filled.
    3. A run of repeating sentences with at least ``min_run_tokens`` tokens is ``ad``: commercials and promos
       run 15-30 s, 40-80 words. Shorter isolated runs are ``boilerplate``: sign-ons, sign-offs, hand-offs
       ("good night", "jake?") that recur every night but belong to the programme.
    """
    sizes = np.bincount(owner, minlength=n_sentences)
    hits = np.bincount(owner, weights=best >= params.min_dates, minlength=n_sentences)
    repeat = (sizes > 0) & (hits >= params.min_share * np.maximum(sizes, 1))
    for start, end in runs(~repeat):
        if 0 < start and end < n_sentences and sizes[start:end].sum() <= params.max_gap_tokens:
            repeat[start:end] = True
    labels = ["program"] * n_sentences
    for start, end in runs(repeat):
        label = "ad" if sizes[start:end].sum() >= params.min_run_tokens else "boilerplate"
        labels[start:end] = [label] * (end - start)
    return labels


@dataclass(frozen=True)
class SentenceLabel:
    """One output row.

    ``sentence`` is the position in the item and ``repeat_share`` the share of its tokens that repeat.
    ``spot`` is the catalogue spot an ``ad`` sentence matched, or -1.
    """

    identifier: str
    minute: int
    sentence: int
    start_second: float
    end_second: float
    label: str
    spot: int
    repeat_share: float
    text: str


def flag_item(
    item: Item,
    indexes: list[RepetitionIndex],
    vocabulary: Vocabulary,
    params: Params,
) -> list[SentenceLabel]:
    """Label every sentence of an item; an index with ``spot_ids`` (the catalogue) also names the spots."""
    best, owner = token_dates(item, indexes, vocabulary)
    n = len(item.sentences)
    labels = label_from_dates(best, owner, n, params)
    sizes = np.bincount(owner, minlength=n)
    hits = np.bincount(owner, weights=best >= params.min_dates, minlength=n)
    share = np.divide(hits, sizes, out=np.zeros(n), where=sizes > 0)
    spots = np.full(len(owner), -1, dtype=np.int32)
    for index in indexes:
        if index.spot_ids is not None:
            found = token_spots(item, index, vocabulary)
            spots = np.where(spots >= 0, spots, found)
    sentence_spot = []
    for j in range(n):
        own = spots[owner == j]
        own = own[own >= 0]
        sentence_spot.append(int(np.bincount(own).argmax()) if len(own) and labels[j] == "ad" else -1)
    return [
        SentenceLabel(
            item.identifier,
            s.minute,
            j,
            round(s.start_second, 1),
            round(s.end_second, 1),
            lab,
            spot,
            round(float(f), 3),
            s.text,
        )
        for j, (s, lab, spot, f) in enumerate(zip(item.sentences, labels, sentence_spot, share, strict=True))
    ]


def ad_free_text(rows: list[SentenceLabel]) -> str:
    """The item's text with ``ad`` sentences removed; boilerplate is kept because it is programme."""
    return " ".join(r.text for r in rows if r.label != "ad")


def as_dicts(rows: list[SentenceLabel]) -> list[dict]:
    """Rows as plain dictionaries for writing."""
    return [asdict(r) for r in rows]
