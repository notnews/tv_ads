"""Second-by-second reference labels from the Vanderbilt Television News Archive index.

The index lists each evening newscast as timed rows (Central time). From 2010 on, commercial breaks have their
own rows ("Commercial", "(Commercial: Geico; ...)"), as do teases and furniture ("Upcoming Items",
"Preview/Introduction", "Program Note", "Good Night"). In some years (seen from 2019) a break is a single row
titled "Upcoming Items & COMMERCIALS: ...". Time not covered by any row is recorded as a gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

COMMERCIAL_RE = re.compile(r"^\(?commercials?\b", re.IGNORECASE)
BREAK_RE = re.compile(r"^upcoming items\s*&\s*commercials", re.IGNORECASE)
FURNITURE_RE = re.compile(r"^(upcoming items|preview|program note|good night|introduction)", re.IGNORECASE)
KINDS = ("story", "furniture", "commercial", "break", "gap", "outside")


def row_kind(title: str) -> str:
    """Classify a Vanderbilt row title as ``commercial``, ``break`` (teases and ads), ``furniture`` or ``story``."""
    title = (title or "").strip()
    if BREAK_RE.match(title):
        return "break"
    if COMMERCIAL_RE.match(title):
        return "commercial"
    if FURNITURE_RE.match(title):
        return "furniture"
    return "story"


def _clock(text: str) -> int:
    moment = datetime.strptime(text.strip(), "%I:%M:%S %p")
    return moment.hour * 3600 + moment.minute * 60 + moment.second


@dataclass(frozen=True)
class Rundown:
    """One newscast's rows as seconds from the half hour it started on."""

    starts: np.ndarray
    ends: np.ndarray
    kinds: list[str]
    titles: list[str]

    def kind_at(self, seconds: np.ndarray) -> list[str]:
        """Row kind at each time; ``gap`` between rows and ``outside`` before the first or after the last."""
        out = []
        for t in np.asarray(seconds, dtype=float):
            if t < self.starts[0] or t >= self.ends.max():
                out.append("outside")
                continue
            hit = np.flatnonzero((self.starts <= t) & (t < self.ends))
            out.append(self.kinds[hit[0]] if len(hit) else "gap")
        return out


def load_rundowns(path: str) -> dict[tuple[str, date], Rundown]:
    """Read the exported index rows into one :class:`Rundown` per (channel, air date).

    Rows of regular evening newscasts only (``program_title`` of the form "ABC Evening News"); zero-length
    rows such as "Good Night" at 05:58:40-05:58:40 are kept for their title but cover no time.
    """
    rows = pd.read_csv(path)
    rows = rows[rows["program_title"].str.fullmatch(r"(ABC|CBS|NBC) Evening News", na=False)]
    clocks = rows["broadcast_time"].str.split("-", expand=True)
    rows = rows.assign(
        start=clocks[0].map(_clock),
        end=clocks[1].map(_clock),
        aired=pd.to_datetime(rows["date"], format="%A, %b %d, %Y").dt.date,
        kind=rows["broadcast_title"].fillna("").map(row_kind),
    )
    out = {}
    for (channel, aired), group in rows.groupby(["channel", "aired"]):
        group = group.sort_values(["start", "end"])
        origin = group["start"].min() // 1800 * 1800
        out[(channel, aired)] = Rundown(
            (group["start"] - origin).to_numpy(),
            (group["end"] - origin).to_numpy(),
            group["kind"].tolist(),
            group["broadcast_title"].fillna("").tolist(),
        )
    return out
