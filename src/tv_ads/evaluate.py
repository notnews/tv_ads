"""Compare sentence labels with the Vanderbilt rundowns on held-out days.

A sentence's time is its minute plus a linear interpolation of its position within the minute's caption text,
shifted by the recording's start relative to the half hour and by a per-network offset. The offset is estimated
on validation days from cues that do not involve the detector: sentences beginning ``reporter:`` should fall in
story rows, and sentences with ``announcer``, ``♪`` or a ``.com`` address in commercial rows.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tv_ads.flag import Params, edge_guard, label_from_dates, token_dates
from tv_ads.index import RepetitionIndex, Vocabulary
from tv_ads.parse import Item
from tv_ads.vanderbilt import Rundown

REPORTER_RE = re.compile(r"^reporter:")
AD_CUE_RE = re.compile(r"announcer|♪|\.com\b")
OFFSETS = range(-60, 61, 5)


def _rank_key(aired: date) -> bytes:
    return hashlib.blake2b(aired.isoformat().encode(), digest_size=8).digest()


def assign_splits(items: list[Item]) -> dict[str, str]:
    """Split each network-year in half by air date: ``validation`` or ``test`` for every item.

    Within a (network, year) stratum the distinct air dates are ordered by a hash of the date and alternate
    between the halves, so every stratum, however small, lands on both sides in equal measure.
    """
    strata: dict[tuple[str, int], set[date]] = {}
    for item in items:
        if item.aired is not None:
            strata.setdefault((item.network, item.aired.year), set()).add(item.aired)
    side = {}
    for key, days in strata.items():
        for rank, day in enumerate(sorted(days, key=_rank_key)):
            side[(key[0], day)] = "validation" if rank % 2 == 0 else "test"
    return {i.identifier: side[(i.network, i.aired)] for i in items if i.aired is not None}


def sentence_times(item: Item, offset: float) -> np.ndarray:
    """Newscast clock (seconds from the half hour, Vanderbilt's frame) of each sentence's midpoint."""
    return np.array([s.approx_second for s in item.sentences]) + item.half_hour_offset - offset


def boundary_distance(rundown: Rundown, times: np.ndarray) -> np.ndarray:
    """Seconds from each time to the nearest boundary between rows of different kinds."""
    edges = [rundown.starts[0], rundown.ends.max()]
    for i in range(1, len(rundown.kinds)):
        if rundown.kinds[i] != rundown.kinds[i - 1]:
            edges.append(rundown.starts[i])
    edges = np.asarray(edges, dtype=float)
    return np.abs(times[:, None] - edges[None, :]).min(axis=1)


def estimate_offsets(items: list[Item], rundowns: dict[tuple[str, date], Rundown]) -> dict[str, int]:
    """Per network, the offset that best places cue sentences in the matching kind of row."""
    best = {}
    for network in sorted({i.network for i in items}):
        scores = {}
        for offset in OFFSETS:
            story = [0, 0]
            commercial = [0, 0]
            for item in items:
                rundown = rundowns.get((item.network, item.aired))
                if item.network != network or rundown is None:
                    continue
                for sentence, kind in zip(item.sentences, rundown.kind_at(sentence_times(item, offset)), strict=True):
                    if kind == "outside":
                        continue
                    if REPORTER_RE.search(sentence.text):
                        story[0] += kind == "story"
                        story[1] += 1
                    if AD_CUE_RE.search(sentence.text):
                        commercial[0] += kind in ("commercial", "break")
                        commercial[1] += 1
            scores[offset] = story[0] / max(story[1], 1) + commercial[0] / max(commercial[1], 1)
        best[network] = max(scores, key=scores.get)
    return best


@dataclass
class Prepared:
    """Per-item token repetition counts and reference kinds, computed once and reused across thresholds."""

    item: Item
    best: np.ndarray
    owner: np.ndarray
    kinds: list[str]
    distance: np.ndarray
    rundown: Rundown
    guard: np.ndarray | None = None


def prepare(
    items: list[Item],
    rundowns: dict[tuple[str, date], Rundown],
    indexes: list[RepetitionIndex],
    vocabulary: Vocabulary,
    offsets: dict[str, int],
) -> list[Prepared]:
    """Items with a rundown, with repetition counts and the reference kind of each sentence."""
    out = []
    for item in items:
        rundown = rundowns.get((item.network, item.aired))
        if rundown is None or not item.sentences:
            continue
        best, owner = token_dates(item, indexes, vocabulary)
        times = sentence_times(item, offsets.get(item.network, 0))
        kinds = rundown.kind_at(times)
        out.append(Prepared(item, best, owner, kinds, boundary_distance(rundown, times), rundown, edge_guard(item)))
    return out


def sentence_table(prepared: list[Prepared], params: Params) -> pd.DataFrame:
    """One row per sentence with its predicted label, reference kind and token count."""
    frames = []
    for p in prepared:
        n = len(p.item.sentences)
        frames.append(
            pd.DataFrame(
                {
                    "identifier": p.item.identifier,
                    "network": p.item.network,
                    "aired": p.item.aired,
                    "sentence": np.arange(n),
                    "tokens": np.bincount(p.owner, minlength=n),
                    "label": label_from_dates(p.best, p.owner, n, params, p.guard),
                    "kind": p.kinds,
                    "boundary_s": p.distance,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def time_recall(prepared: list[Prepared], params: Params, offsets: dict[str, int]) -> pd.DataFrame:
    """Per newscast, seconds of break time, how many have caption text, and how many an ``ad`` sentence covers.

    A second is captioned when its recording minute has a caption block. Covered means it lies inside the
    interpolated span of a sentence labeled ``ad``.
    """
    rows = []
    for p in prepared:
        item = p.item
        shift = item.half_hour_offset - offsets.get(item.network, 0)
        rundown = p.rundown
        clock = np.arange(int(rundown.starts[0]), int(rundown.ends.max()))
        kinds = np.asarray(rundown.kind_at(clock))
        is_break = (kinds == "commercial") | (kinds == "break")
        minutes = np.floor((clock - shift) / 60).astype(int)
        captioned = np.isin(minutes, [s.minute for s in item.sentences])
        labels = label_from_dates(p.best, p.owner, len(item.sentences), params, p.guard)
        covered = np.zeros(len(clock), dtype=bool)
        for sentence, label in zip(item.sentences, labels, strict=True):
            if label == "ad":
                covered |= (clock >= sentence.start_second + shift) & (clock < sentence.end_second + shift)
        rows.append(
            {
                "identifier": item.identifier,
                "network": item.network,
                "year": item.aired.year,
                "break_s": int(is_break.sum()),
                "break_captioned_s": int((is_break & captioned).sum()),
                "break_covered_s": int((is_break & covered).sum()),
                "break_captioned_covered_s": int((is_break & captioned & covered).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarise_time(frame: pd.DataFrame) -> dict[str, float]:
    """Recall of break time over all of it and over the part with caption text, and the captioned share."""
    total = frame[["break_s", "break_captioned_s", "break_covered_s", "break_captioned_covered_s"]].sum()
    return {
        "break_seconds": int(total["break_s"]),
        "captioned_share": float(total["break_captioned_s"] / max(total["break_s"], 1)),
        "recall_all_time": float(total["break_covered_s"] / max(total["break_s"], 1)),
        "recall_captioned_time": float(total["break_captioned_covered_s"] / max(total["break_captioned_s"], 1)),
    }


def scores(table: pd.DataFrame) -> dict[str, float]:
    """Precision, recall and F1 of ``ad`` for break time versus story time, weighting sentences by tokens.

    Break time is ``commercial`` rows plus ``break`` rows ("Upcoming Items & COMMERCIALS", used from 2019),
    which also hold a few seconds of tease. ``furniture``, ``gap`` and ``outside`` sentences are left out.
    """
    return _scores(table["tokens"].to_numpy(), table["kind"].to_numpy(), table["label"].to_numpy() == "ad")


def _scores(tokens: np.ndarray, kinds: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    truth = (kinds == "commercial") | (kinds == "break")
    keep = truth | (kinds == "story")
    w, truth, flagged = tokens[keep], truth[keep], flagged[keep]
    tp = w[flagged & truth].sum()
    precision = tp / max(w[flagged].sum(), 1)
    recall = tp / max(w[truth].sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "break_share": float(w[truth].sum() / max(w.sum(), 1)),
        "tokens": int(w.sum()),
    }


GRID = {
    "min_dates": (1, 2, 3, 5, 8, 12, 20),
    "min_share": (0.3, 0.5, 0.7),
    "min_run_tokens": (0, 15, 30, 50, 80, 120),
    "max_gap_tokens": (0, 10, 25, 50),
}


def macro(per_network: dict[str, dict[str, float]]) -> float:
    """Mean F1 over networks, so each network counts the same however many words it has."""
    return float(np.mean([v["f1"] for v in per_network.values()])) if per_network else 0.0


def tune(prepared: list[Prepared]) -> tuple[Params, pd.DataFrame]:
    """Grid-search the thresholds for the highest macro F1 across networks on the given (validation) items.

    The returned grid also has each network's own F1 (``f1_ABC``, ...), to check whether one shared setting
    is close to the best each network could get alone.
    """
    tokens = np.concatenate([np.bincount(p.owner, minlength=len(p.item.sentences)) for p in prepared])
    kinds = np.concatenate([np.asarray(p.kinds) for p in prepared])
    networks = np.concatenate([np.full(len(p.item.sentences), p.item.network) for p in prepared])
    names = sorted(set(networks.tolist()))
    results = []
    for values in itertools.product(*GRID.values()):
        setting = dict(zip(GRID, values, strict=True))
        params = Params(**setting)
        flagged = (
            np.concatenate(
                [
                    np.asarray(label_from_dates(p.best, p.owner, len(p.item.sentences), params, p.guard))
                    for p in prepared
                ]
            )
            == "ad"
        )
        per = {n: _scores(tokens[networks == n], kinds[networks == n], flagged[networks == n]) for n in names}
        row = {**setting, "macro_f1": macro(per), "pooled_f1": _scores(tokens, kinds, flagged)["f1"]}
        row.update({f"f1_{n}": per[n]["f1"] for n in names})
        results.append(row)
    frame = pd.DataFrame(results).sort_values("macro_f1", ascending=False, ignore_index=True)
    top = frame.iloc[0]
    return Params(**{k: type(getattr(Params(), k))(top[k]) for k in GRID}), frame
