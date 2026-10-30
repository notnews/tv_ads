"""Count, for every word n-gram in a corpus, the distinct air dates and networks it appears on."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tv_ads.parse import Item, tokenize

NETWORKS = ("ABC", "CBS", "NBC", "CNN", "FOXNEWS", "MSNBC")
MULTIPLIER = np.uint64(0x100000001B3)


class Vocabulary:
    """Stable 64-bit ids for word types, so n-gram hashes agree across runs and processes."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def ids(self, tokens: list[str]) -> np.ndarray:
        """Ids of ``tokens``; blake2b gives the same id for a word in every run, unlike ``hash``."""
        out = np.empty(len(tokens), dtype=np.uint64)
        for i, token in enumerate(tokens):
            value = self._ids.get(token)
            if value is None:
                value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
                self._ids[token] = value
            out[i] = value
        return out


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Hash of every run of ``n`` consecutive ids: position i covers ids[i : i + n]."""
    if len(ids) < n:
        return np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        out = np.zeros(len(ids) - n + 1, dtype=np.uint64)
        for k in range(n):
            out = out * MULTIPLIER + ids[k : len(ids) - n + 1 + k]
    return out


def item_tokens(item: Item) -> tuple[list[str], np.ndarray]:
    """All tokens of an item in order and, aligned with them, the index of the sentence each belongs to."""
    tokens: list[str] = []
    owner: list[int] = []
    for j, sentence in enumerate(item.sentences):
        words = tokenize(sentence.text)
        tokens.extend(words)
        owner.extend([j] * len(words))
    return tokens, np.asarray(owner, dtype=np.int64)


@dataclass
class RepetitionIndex:
    """Sorted n-gram hashes with the number of distinct air dates and networks each was seen on."""

    n: int
    hashes: np.ndarray
    n_dates: np.ndarray
    n_networks: np.ndarray
    first_day: np.ndarray
    last_day: np.ndarray
    identifiers: set[str]

    def lookup(self, hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distinct dates, distinct networks and day span (last minus first seen) for each hash; 0 if absent."""
        if not len(self.hashes):
            zeros = np.zeros(len(hashes), dtype=np.int32)
            return zeros, zeros, zeros
        pos = np.minimum(np.searchsorted(self.hashes, hashes), len(self.hashes) - 1)
        found = self.hashes[pos] == hashes
        dates = np.where(found, self.n_dates[pos], 0)
        nets = np.where(found, self.n_networks[pos], 0)
        span = np.where(found, self.last_day[pos] - self.first_day[pos], 0)
        return dates, nets, span

    def save(self, path: Path) -> None:
        """Write the index as a compressed ``.npz``."""
        np.savez_compressed(
            path,
            n=self.n,
            hashes=self.hashes,
            n_dates=self.n_dates,
            n_networks=self.n_networks,
            first_day=self.first_day,
            last_day=self.last_day,
            identifiers=np.array(sorted(self.identifiers)),
        )

    @classmethod
    def load(cls, path: Path) -> RepetitionIndex:
        """Read an index written by :meth:`save`."""
        with np.load(path) as data:
            return cls(
                int(data["n"]),
                data["hashes"],
                data["n_dates"],
                data["n_networks"],
                data["first_day"],
                data["last_day"],
                set(data["identifiers"].tolist()),
            )


def build_index(items: list[Item], n: int, vocabulary: Vocabulary | None = None) -> RepetitionIndex:
    """Index the distinct n-grams of each dated item by air date and network.

    Items without an air date are skipped: repetition is counted across dates, so an undated item cannot
    contribute evidence.
    """
    vocabulary = vocabulary or Vocabulary()
    hash_parts, day_parts, net_parts, used = [], [], [], set()
    for item in items:
        if item.aired is None:
            continue
        tokens, _ = item_tokens(item)
        hashes = np.unique(ngram_hashes(vocabulary.ids(tokens), n))
        hash_parts.append(hashes)
        day_parts.append(np.full(len(hashes), item.aired.toordinal(), dtype=np.int32))
        net = NETWORKS.index(item.network) if item.network in NETWORKS else len(NETWORKS)
        net_parts.append(np.full(len(hashes), net, dtype=np.int8))
        used.add(item.identifier)
    if not hash_parts:
        empty = np.empty(0, dtype=np.uint64)
        return RepetitionIndex(n, empty, *(np.empty(0, dtype=np.int32) for _ in range(4)), used)
    hashes = np.concatenate(hash_parts)
    days = np.concatenate(day_parts)
    nets = np.concatenate(net_parts)
    order = np.lexsort((days, hashes))
    hashes, days, nets = hashes[order], days[order], nets[order]
    keys, starts = np.unique(hashes, return_index=True)
    ends = np.append(starts[1:], len(hashes))
    new_day = np.ones(len(hashes), dtype=bool)
    new_day[1:] = (hashes[1:] != hashes[:-1]) | (days[1:] != days[:-1])
    n_dates = np.add.reduceat(new_day.astype(np.int32), starts)
    net_order = np.lexsort((nets, hashes))
    h2, n2 = hashes[net_order], nets[net_order]
    new_net = np.ones(len(h2), dtype=bool)
    new_net[1:] = (h2[1:] != h2[:-1]) | (n2[1:] != n2[:-1])
    n_networks = np.add.reduceat(new_net.astype(np.int32), starts)
    return RepetitionIndex(n, keys, n_dates, n_networks, days[starts], days[ends - 1], used)
