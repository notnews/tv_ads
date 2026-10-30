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

from tv_ads.flag import Params, label_from_dates, token_dates
from tv_ads.index import RepetitionIndex, Vocabulary
from tv_ads.parse import Item
from tv_ads.vanderbilt import Rundown

REPORTER_RE = re.compile(r"^reporter:")
AD_CUE_RE = re.compile(r"announcer|♪|\.com\b")
OFFSETS = range(-60, 61, 5)


def split_of(aired: date) -> str:
    """Assign an air date, and every newscast on it, to ``validation`` or ``test`` by a hash of the date."""
    digest = hashlib.blake2b(aired.isoformat().encode(), digest_size=2).digest()
    return "validation" if digest[0] % 2 == 0 else "test"


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
                        commercial[0] += kind == "commercial"
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


def prepare(
    items: list[Item],
    rundowns: dict[tuple[str, date], Rundown],
    index: RepetitionIndex,
    vocabulary: Vocabulary,
    offsets: dict[str, int],
) -> list[Prepared]:
    """Items with a rundown, with repetition counts and the reference kind of each sentence."""
    out = []
    for item in items:
        rundown = rundowns.get((item.network, item.aired))
        if rundown is None or not item.sentences:
            continue
        best, owner = token_dates(item, index, vocabulary)
        times = sentence_times(item, offsets.get(item.network, 0))
        out.append(Prepared(item, best, owner, rundown.kind_at(times), boundary_distance(rundown, times)))
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
                    "label": label_from_dates(p.best, p.owner, n, params),
                    "kind": p.kinds,
                    "boundary_s": p.distance,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def scores(table: pd.DataFrame) -> dict[str, float]:
    """Precision, recall and F1 of ``ad`` for commercial versus story time, weighting sentences by tokens.

    Only sentences in ``commercial`` or ``story`` rows count; ``furniture``, ``break`` (teases and ads in one
    row), ``gap`` and ``outside`` are left out because they mix both or have no reference.
    """
    return _scores(table["tokens"].to_numpy(), table["kind"].to_numpy(), table["label"].to_numpy() == "ad")


def _scores(tokens: np.ndarray, kinds: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    keep = (kinds == "commercial") | (kinds == "story")
    w, truth, flagged = tokens[keep], kinds[keep] == "commercial", flagged[keep]
    tp = w[flagged & truth].sum()
    precision = tp / max(w[flagged].sum(), 1)
    recall = tp / max(w[truth].sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "commercial_share": float(w[truth].sum() / max(w.sum(), 1)),
        "tokens": int(w.sum()),
    }


GRID = {
    "min_dates": (1, 2, 3, 5, 8, 12, 20),
    "min_share": (0.3, 0.5, 0.7),
    "min_run_tokens": (0, 15, 30, 50, 80, 120),
    "max_gap_tokens": (0, 10, 25, 50),
}


def tune(prepared: list[Prepared]) -> tuple[Params, pd.DataFrame]:
    """Grid-search the thresholds for the highest token-weighted F1 on the given (validation) items."""
    tokens = np.concatenate([np.bincount(p.owner, minlength=len(p.item.sentences)) for p in prepared])
    kinds = np.concatenate([np.asarray(p.kinds) for p in prepared])
    results = []
    for values in itertools.product(*GRID.values()):
        setting = dict(zip(GRID, values, strict=True))
        params = Params(**setting)
        labels = np.concatenate(
            [np.asarray(label_from_dates(p.best, p.owner, len(p.item.sentences), params)) for p in prepared]
        )
        results.append({**setting, **_scores(tokens, kinds, labels == "ad")})
    frame = pd.DataFrame(results).sort_values("f1", ascending=False, ignore_index=True)
    top = frame.iloc[0]
    return Params(**{k: type(getattr(Params(), k))(top[k]) for k in GRID}), frame
