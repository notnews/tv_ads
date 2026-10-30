"""Score sentence labels against moment-level codes (one coded instant per sampled newscast).

notwork_news samples one moment per newscast (``captions_sample.csv.gz``: ``moment_id``, ``identifier``,
``minute``, ``second``) and codes each one (``captions_codes.csv.gz``: ``moment_id``, ``kind`` in story,
headlines, commercial, other). A moment is matched to the sentence whose interpolated span contains it.
"""

from __future__ import annotations

import pandas as pd


def match_moments(sentences: pd.DataFrame, sample: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:
    """Each coded moment with the label of the sentence spanning it (``label`` is missing when none does)."""
    moments = codes.merge(sample[["moment_id", "identifier", "minute", "second"]], on="moment_id", how="inner")
    moments["at"] = 60 * moments["minute"] + moments["second"]
    spans = sentences[["identifier", "start_second", "end_second", "label", "text"]]
    joined = moments.merge(spans, on="identifier", how="left")
    inside = (joined["start_second"] <= joined["at"]) & (joined["at"] < joined["end_second"])
    hits = joined[inside].drop_duplicates("moment_id")
    return moments.merge(hits[["moment_id", "label", "text"]], on="moment_id", how="left")


def moment_scores(matched: pd.DataFrame) -> dict:
    """Confusion of code kind by predicted label, and precision and recall of ``ad`` for ``commercial``."""
    rows = matched.dropna(subset=["label"])
    flagged = rows["label"] == "ad"
    truth = rows["kind"] == "commercial"
    tp = int((flagged & truth).sum())
    return {
        "moments": len(matched),
        "matched": len(rows),
        "precision": tp / max(int(flagged.sum()), 1),
        "recall": tp / max(int(truth.sum()), 1),
        "confusion": pd.crosstab(rows["kind"], rows["label"]).to_dict(orient="index"),
    }
