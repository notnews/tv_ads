"""A catalogue of known spots: recurring ad copy from a corpus, grouped into spots and shipped with the package.

A *line* is the token string of one sentence inside an ``ad`` run. Lines seen on two or more air dates are
kept. Two lines belong to the same *spot* when one follows the other, within a run, on more than half the
dates the more-aired of the two was seen on. A spot's lines air together and in order. At a break boundary
the next spot changes from night to night, so the link across it is weak and breaks fall apart into spots.
A generic line that many spots share ("ask your doctor ...") joins none of them, because it mostly does not
follow any one of them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from tv_ads.flag import Params, edge_guard, label_from_dates, runs, token_dates
from tv_ads.index import DEFAULT_N, NETWORKS, RepetitionIndex, Vocabulary, ngram_hashes
from tv_ads.parse import Item, network_of, tokenize

PACKAGED_PATH = Path(str(resources.files("tv_ads") / "data" / "spots.csv.gz"))
COLUMNS = ["spot", "source", "airings", "first_aired", "last_aired", "networks", "lines", "tokens", "windows", "text"]
HUMAN_DATES = 1000


@dataclass
class _Run:
    identifier: str
    aired: date
    network: str
    keys: list[str]
    texts: list[str]


def ad_runs(
    items: list[Item], indexes: list[RepetitionIndex], vocabulary: Vocabulary, params: Params
) -> tuple[list[_Run], Counter]:
    """Every run of ``ad`` sentences in the items, as normalised lines and original text; and how many times
    each line occurs anywhere in the items."""
    out = []
    seen: Counter = Counter()
    for item in items:
        if item.aired is None or not item.sentences:
            continue
        best, owner = token_dates(item, indexes, vocabulary)
        labels = np.asarray(label_from_dates(best, owner, len(item.sentences), params, edge_guard(item)))
        for sentence in item.sentences:
            seen[" ".join(tokenize(sentence.text))] += 1
        for start, end in runs(labels == "ad"):
            keys, texts = [], []
            for sentence in item.sentences[start:end]:
                key = " ".join(tokenize(sentence.text))
                if key:
                    keys.append(key)
                    texts.append(sentence.text)
            out.append(_Run(item.identifier, item.aired, item.network, keys, texts))
    return out, seen


class _Components:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def cluster_spots(
    ad: list[_Run], seen: Counter, n: int = DEFAULT_N, min_link: float = 0.5, min_ad_share: float = 0.5
) -> pd.DataFrame:
    """Group recurring lines into spots and describe each spot (see the module docstring).

    A line is kept when it aired on two or more dates inside ad runs and at least ``min_ad_share`` of all its
    occurrences were inside ad runs. The second condition drops a reporter's sign-off
    ("linsey davis, abc news, new york.") that sometimes sits next to a break but mostly ends a story. A
    spot shorter than ``n`` words is dropped, because it has no n-gram of its own to match.
    """
    line_dates: dict[str, set[date]] = defaultdict(set)
    in_ads: Counter = Counter()
    for run in ad:
        for key in run.keys:
            line_dates[key].add(run.aired)
            in_ads[key] += 1
    kept = {k for k, d in line_dates.items() if len(d) >= 2 and in_ads[k] >= min_ad_share * seen[k]}
    pair_dates: dict[tuple[str, str], set[date]] = defaultdict(set)
    for run in ad:
        seq = [k for k in run.keys if k in kept]
        for a, b in zip(seq, seq[1:], strict=False):
            if a != b:
                pair_dates[(a, b)].add(run.aired)
    components = _Components()
    for key in kept:
        components.find(key)
    for (a, b), days in pair_dates.items():
        if len(days) >= 2 and len(days) > min_link * max(len(line_dates[a]), len(line_dates[b])):
            components.union(a, b)
    members: dict[str, set[str]] = defaultdict(set)
    for key in kept:
        members[components.find(key)].add(key)

    airings: dict[str, list[_Run]] = defaultdict(list)
    best_text: dict[str, list[str]] = {}
    best_count: dict[str, int] = {}
    for run in ad:
        hits = defaultdict(list)
        for key, text in zip(run.keys, run.texts, strict=True):
            if key in kept:
                hits[components.find(key)].append((key, text))
        for root, found in hits.items():
            distinct = {k for k, _ in found}
            if 2 * len(distinct) >= len(members[root]):
                airings[root].append(run)
            if len(distinct) > best_count.get(root, 0):
                best_count[root] = len(distinct)
                best_text[root] = list(dict.fromkeys(t for _, t in found))

    rows = []
    for root, aired in airings.items():
        days = sorted({r.aired for r in aired})
        text = " ".join(best_text[root])
        if len(days) < 2 or len(tokenize(text)) < n:
            continue
        rows.append(
            {
                "airings": len(days),
                "first_aired": days[0].isoformat(),
                "last_aired": days[-1].isoformat(),
                "networks": ";".join(sorted({r.network for r in aired})),
                "lines": len(members[root]),
                "tokens": len(tokenize(text)),
                "source": "auto",
                "windows": "",
                "text": text,
            }
        )
    frame = pd.DataFrame(rows, columns=COLUMNS[1:])
    frame = frame.sort_values(["airings", "first_aired"], ascending=[False, True], ignore_index=True)
    frame.insert(0, "spot", np.arange(len(frame)))
    return frame


def without_window(spots: pd.DataFrame, window: str) -> pd.DataFrame:
    """The catalogue as it would be had ``window`` never been annotated: human spots from it are dropped and
    spots that were ``both`` lose the window (and become ``auto`` if it was their only one)."""
    windows = spots["windows"].astype(str).str.split(";")
    has = windows.map(lambda w: window in w)
    rest = windows.map(lambda w: ";".join(x for x in w if x and x != window))
    out = spots.assign(windows=rest)
    out.loc[has & (rest == "") & (out["source"] == "both"), "source"] = "auto"
    return out[~(has & (rest == "") & (spots["source"] == "human"))]


def auto_only(spots: pd.DataFrame) -> pd.DataFrame:
    """The catalogue without any human annotation."""
    return spots[spots["source"] != "human"].assign(source="auto", windows="")


def catalogue_index(spots: pd.DataFrame, n: int, vocabulary: Vocabulary | None = None) -> RepetitionIndex:
    """An index whose n-grams are those of the spots' text, each counted as its spot's airings.

    Spots a person labelled (``human`` or ``both``) count as :data:`HUMAN_DATES` airings, so they match under
    any ``min_dates``. Where spots share an n-gram the most-aired spot wins. The index names no items, so a
    page is never discounted for being one of the airings the catalogue was built from.
    """
    vocabulary = vocabulary or Vocabulary()
    parts = []
    for row in spots.itertuples():
        weight = max(int(row.airings), HUMAN_DATES) if row.source in ("human", "both") else int(row.airings)
        hashes = np.unique(ngram_hashes(vocabulary.ids(tokenize(row.text)), n))
        nets = sum(1 for net in str(row.networks).split(";") if net in NETWORKS) or 1
        first, last = date.fromisoformat(row.first_aired).toordinal(), date.fromisoformat(row.last_aired).toordinal()
        parts.append(
            pd.DataFrame(
                {"hash": hashes, "dates": weight, "nets": nets, "first": first, "last": last, "spot": row.spot}
            )
        )
    if not parts:
        empty = np.empty(0, dtype=np.int32)
        return RepetitionIndex(n, np.empty(0, dtype=np.uint64), empty, empty, empty, empty, set(), empty)
    table = pd.concat(parts, ignore_index=True).sort_values(["hash", "dates"], ascending=[True, False])
    table = table.drop_duplicates("hash", keep="first")
    return RepetitionIndex(
        n,
        table["hash"].to_numpy(dtype=np.uint64),
        table["dates"].to_numpy(dtype=np.int32),
        table["nets"].to_numpy(dtype=np.int32),
        table["first"].to_numpy(dtype=np.int32),
        table["last"].to_numpy(dtype=np.int32),
        set(),
        table["spot"].to_numpy(dtype=np.int32),
    )


@dataclass
class Catalogue:
    """Spots and the index that matches them."""

    spots: pd.DataFrame
    index: RepetitionIndex

    @classmethod
    def read(cls, path: Path | None = None, n: int = DEFAULT_N) -> Catalogue:
        """Read a spots file; with no path, the catalogue packaged with tv_ads."""
        spots = pd.read_csv(path or PACKAGED_PATH, keep_default_na=False)
        return cls(spots, catalogue_index(spots, n))

    @staticmethod
    def index_for(spots: pd.DataFrame, n: int = DEFAULT_N) -> RepetitionIndex:
        """The matching index for a spots table."""
        return catalogue_index(spots, n)


def build_catalogue(
    items: list[Item], indexes: list[RepetitionIndex], vocabulary: Vocabulary, params: Params
) -> pd.DataFrame:
    """Spots from the ``ad`` runs of a corpus (pass the corpus's own index)."""
    return cluster_spots(*ad_runs(items, indexes, vocabulary, params), n=min(i.n for i in indexes))


def human_spots(gold: pd.DataFrame, aired: dict[str, date] | None = None) -> pd.DataFrame:
    """Spots from hand labels: consecutive ``commercial`` sentences of one item (in minute and sentence order,
    with no other label between them) form one spot. ``aired`` maps identifiers to local air dates; without
    it the date in the identifier (UTC, so sometimes a day late) is used."""
    gold = gold.sort_values(["identifier", "minute", "sentence"])
    rows = []
    for identifier, group in gold.groupby("identifier", sort=False):
        commercial = (group["label"] == "commercial").to_numpy()
        for start, end in runs(commercial):
            chunk = group.iloc[start:end]
            text = " ".join(chunk["text"].astype(str))
            day = (aired or {}).get(identifier) or datetime.strptime(identifier.split("_")[1], "%Y%m%d").date()
            rows.append(
                {
                    "source": "human",
                    "airings": 1,
                    "first_aired": day.isoformat(),
                    "last_aired": day.isoformat(),
                    "networks": network_of(identifier),
                    "lines": end - start,
                    "tokens": len(tokenize(text)),
                    "windows": ";".join(sorted(set(chunk["window"].astype(str)))),
                    "text": text,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS[1:])


def add_human(spots: pd.DataFrame, human: pd.DataFrame, n: int = DEFAULT_N, covered: float = 0.9) -> pd.DataFrame:
    """Merge hand-labelled spots into a catalogue.

    An existing spot with at least half its n-grams inside a human spot is confirmed: it becomes ``both`` and
    records the window. The human spot is appended as ``human`` unless existing spots already cover
    ``covered`` of its n-grams. Re-adding a window replaces what it added before.
    """
    for window in sorted({w for ws in human["windows"] for w in str(ws).split(";") if w}):
        spots = without_window(spots, window)
    spots = spots.reset_index(drop=True).copy()
    vocabulary = Vocabulary()
    grams = [set(ngram_hashes(vocabulary.ids(tokenize(t)), n).tolist()) for t in spots["text"]]
    owners: dict[int, list[int]] = defaultdict(list)
    for i, hashes in enumerate(grams):
        for h in hashes:
            owners[h].append(i)
    new = []
    for row in human.to_dict("records"):
        hashes = set(ngram_hashes(vocabulary.ids(tokenize(row["text"])), n).tolist())
        inside = Counter(i for h in hashes for i in owners.get(h, ()))
        for i, count in inside.items():
            if count >= 0.5 * len(grams[i]):
                windows = {w for w in str(spots.at[i, "windows"]).split(";") if w} | set(row["windows"].split(";"))
                spots.at[i, "windows"] = ";".join(sorted(windows))
                if spots.at[i, "source"] == "auto":
                    spots.at[i, "source"] = "both"
        if not hashes or sum(h in owners for h in hashes) < covered * len(hashes):
            new.append(row)
    out = pd.concat([spots, pd.DataFrame(new, columns=COLUMNS[1:])], ignore_index=True)
    out["spot"] = np.arange(len(out))
    return out[COLUMNS]


def summary(spots: pd.DataFrame) -> dict:
    """Counts for the log and the README."""
    return {
        "spots": len(spots),
        "sources": Counter(spots["source"]).most_common() if "source" in spots else [],
        "median_tokens": float(spots["tokens"].median()) if len(spots) else 0.0,
        "networks": Counter(spots["networks"]).most_common(),
        "lines": int(spots["lines"].sum()),
        "tokens": int(spots["tokens"].sum()),
    }
