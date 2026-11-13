"""Read caption pages into minute blocks, sentences and tokens."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup

START_RE = re.compile(r"/start/(\d+)/end/(\d+)")
TITLE_DATE_RE = re.compile(r":\s*([A-Z][a-z]+ \d{1,2}, \d{4})\s+\d{1,2}:\d{2}\s*[ap]m", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")

# Captions are lower case, so a period after these is rarely a sentence end. A regex is used rather than
# pysbd (no release since 2021) or nltk punkt (needs a model download and learns abbreviations from
# capitalization the captions lack).
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "gen", "gov", "sen", "rep", "lt", "col", "sgt", "capt",
    "prof", "rev", "mt", "ft", "no", "inc", "corp", "co", "etc", "u.s", "u.n", "u.k", "a.m", "p.m", "d.c",
    "l.a", "n.y", "e.u",
}  # fmt: skip
BOUNDARY_RE = re.compile(r"(?<=[.?!])([\"')\]]*)\s+|\s*>>+\s*")


@dataclass(frozen=True, slots=True)
class Block:
    """One minute of captions: seconds [start, end) from the start of the recording."""

    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class Sentence:
    """A sentence with its place in the recording."""

    minute: int
    index: int
    text: str
    start_char: int
    block_chars: int

    def _second(self, char: float) -> float:
        return 60 * self.minute + 60 * char / max(self.block_chars, 1)

    @property
    def start_second(self) -> float:
        """Seconds from recording start, placing characters evenly across the minute."""
        return self._second(self.start_char)

    @property
    def end_second(self) -> float:
        """Seconds from recording start at the sentence's last character, interpolated as for the start."""
        return self._second(self.start_char + len(self.text))

    @property
    def approx_second(self) -> float:
        """Midpoint of :attr:`start_second` and :attr:`end_second`."""
        return (self.start_second + self.end_second) / 2


@dataclass(frozen=True, slots=True)
class Item:
    """A broadcast: identifier, air date in local time, network and sentences in order."""

    identifier: str
    aired: date | None
    network: str
    station: str
    sentences: list[Sentence]

    @property
    def half_hour_offset(self) -> int:
        """Seconds between the nearest half hour and the recording start, from the identifier's HHMMSS.

        ``WJLA_20100611_223437_...`` started 4 min 37 s after the half hour, so its second 0 is second 277 of
        the newscast. Negative when a recording started just before the half hour.
        """
        parts = self.identifier.split("_")
        if len(parts) < 3 or not re.fullmatch(r"\d{6}", parts[2]):
            return 0
        clock = parts[2]
        seconds = (int(clock[:2]) * 3600 + int(clock[2:4]) * 60 + int(clock[4:])) % 1800
        return seconds - 1800 if seconds > 900 else seconds


def read_text(path: Path) -> str:
    """Read a file that may be gzip-compressed."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def parse_blocks(html: str) -> list[Block]:
    """Return the caption blocks of a details page in recording order.

    Each ``div.tvcol`` carries a share link ``/start/S/end/E`` and the caption snippet ``div.snipin.nosel``
    (the selector archive_news_cc uses). Columns without a snippet are skipped.
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks = []
    for col in soup.select("div.tvcol"):
        snippet = col.select_one("div.snipin.nosel")
        link = col.find("a", href=START_RE)
        if snippet is None or link is None:
            continue
        start, end = (int(g) for g in START_RE.search(link["href"]).groups())
        text = " ".join(snippet.get_text(" ", strip=True).split())
        if text:
            blocks.append(Block(start, end, text))
    return sorted(blocks, key=lambda b: b.start)


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Split caption text into ``(char_offset, sentence)`` pairs.

    Boundaries are terminal punctuation followed by space, and the ``>>`` speaker-change mark. A period after
    a known abbreviation or a single letter (as in ``j. crew``) does not end a sentence.
    """
    pieces: list[tuple[int, str]] = []
    start = 0
    for match in BOUNDARY_RE.finditer(text):
        chunk = text[start : match.start() + len(match.group(1) or "")]
        last = chunk.rsplit(" ", 1)[-1].rstrip(".").lower()
        if ">" not in match.group() and (last in ABBREVIATIONS or len(last) == 1):
            continue
        if chunk.strip():
            pieces.append((start, chunk.strip()))
        start = match.end()
    if text[start:].strip():
        pieces.append((start, text[start:].strip()))
    return pieces


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; punctuation, symbols such as ® and URL dots drop out."""
    return TOKEN_RE.findall(text.lower())


def title_date(title: str) -> date | None:
    """Local air date from an item title such as ``... : KPIX : July 17, 2016 5:30pm-6:31pm PDT``."""
    match = TITLE_DATE_RE.search(title or "")
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%B %d, %Y").date()


def network_of(identifier: str) -> str:
    """Network token in an identifier such as ``KGO_20120103_013000_ABC_World_News...``."""
    parts = identifier.split("_")
    return parts[3] if len(parts) > 3 else ""


def sentences_of(blocks: list[Block]) -> list[Sentence]:
    """Sentences of every block, keeping the minute each came from."""
    out = []
    for block in blocks:
        for i, (offset, text) in enumerate(split_sentences(block.text)):
            out.append(Sentence(block.start // 60, i, text, offset, len(block.text)))
    return out


def load_item(html_path: Path, meta_path: Path | None = None) -> Item:
    """Parse a details page and, when present, its metadata JSON into an :class:`Item`."""
    name = html_path.name.split(".html")[0]
    title = ""
    if meta_path is not None and meta_path.exists():
        meta = json.loads(read_text(meta_path)).get("metadata", {})
        name = meta.get("identifier", name)
        title = meta.get("title", "")
    station = name.split("_")[0]
    return Item(name, title_date(title), network_of(name), station, sentences_of(parse_blocks(read_text(html_path))))


def meta_path_for(html_path: Path) -> Path | None:
    """Locate ``meta/<id>_meta.json(.gz)`` beside an ``html/`` directory, or next to the page."""
    name = html_path.name.split(".html")[0]
    for folder in (html_path.parent.parent / "meta", html_path.parent):
        for suffix in ("_meta.json", "_meta.json.gz"):
            candidate = folder / f"{name}{suffix}"
            if candidate.exists():
                return candidate
    return None
