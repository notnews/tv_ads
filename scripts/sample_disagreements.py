"""Draw a random sample of test sentences where the detector and the Vanderbilt reference disagree.

Usage: uv run python scripts/sample_disagreements.py <validate_dir> <pages_dir> <out.csv> [per_cell]

Writes sentences labelled ``ad`` in story rows and sentences labelled ``program`` in commercial rows (four
or more tokens), with their text, so a reader can judge which side is wrong. The ``judged_ad`` column is
left empty for that reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from tv_ads.parse import load_item, meta_path_for


def main() -> None:
    validate_dir, pages, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    per_cell = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    table = pd.read_csv(validate_dir / "test_sentences.csv.gz")
    table = table[table["tokens"] >= 4]
    cells = {
        "ad_in_story_row": (table["kind"] == "story") & (table["label"] == "ad"),
        "program_in_commercial_row": (table["kind"] == "commercial") & (table["label"] == "program"),
    }
    frames = []
    for name, mask in cells.items():
        pick = table[mask].sample(min(per_cell, int(mask.sum())), random_state=20261030)
        frames.append(pick.assign(cell=name))
    sample = pd.concat(frames, ignore_index=True)
    texts = []
    for row in sample.itertuples():
        page = next(pages.glob(f"{row.identifier}.html*"))
        texts.append(load_item(page, meta_path_for(page)).sentences[row.sentence].text)
    sample["text"] = texts
    sample["judged_ad"] = ""
    columns = ["cell", "identifier", "sentence", "boundary_s", "kind", "label", "text", "judged_ad"]
    sample[columns].to_csv(out, index=False)


if __name__ == "__main__":
    main()
