"""Score ``flag`` output against hand-labelled sentences.

The gold file has ``identifier``, ``minute``, ``label`` (news, tease, commercial, promo, other, unclear) and
either ``sentence`` (the position in the item, as ``flag`` writes it) or ``text``. Rows with ``text`` are
matched to a sentence of the same item and minute by their tokens, which survives a different sentence
splitter. ``commercial`` and ``promo`` are positives; ``unclear`` rows are left out.
"""

from __future__ import annotations

import pandas as pd

from tv_ads.parse import tokenize

POSITIVE = {"commercial", "promo"}
NEGATIVE = {"news", "tease", "other"}


def _key(text: str) -> str:
    return " ".join(tokenize(str(text)))


def match_gold(sentences: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Gold rows with the predicted ``predicted`` label of their sentence (missing when none matches)."""
    gold = gold.copy()
    if "text" in gold.columns and gold["text"].notna().any():
        left = gold.assign(key=gold["text"].map(_key))
        right = sentences.assign(key=sentences["text"].map(_key))[["identifier", "minute", "key", "label"]]
        right = right.drop_duplicates(["identifier", "minute", "key"])
        out = left.merge(right.rename(columns={"label": "predicted"}), on=["identifier", "minute", "key"], how="left")
        return out.drop(columns="key")
    right = sentences[["identifier", "sentence", "label"]].rename(columns={"label": "predicted"})
    return gold.merge(right, on=["identifier", "sentence"], how="left")


def gold_scores(matched: pd.DataFrame) -> dict:
    """Precision and recall of ``ad`` for commercial and promo sentences, and the full confusion table."""
    rows = matched[matched["label"].isin(POSITIVE | NEGATIVE) & matched["predicted"].notna()]
    flagged = rows["predicted"] == "ad"
    truth = rows["label"].isin(POSITIVE)
    tp = int((flagged & truth).sum())
    precision = tp / max(int(flagged.sum()), 1)
    recall = tp / max(int(truth.sum()), 1)
    return {
        "gold_rows": len(matched),
        "scored": len(rows),
        "unmatched": int(matched["predicted"].isna().sum()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "confusion": pd.crosstab(matched["label"], matched["predicted"].fillna("unmatched")).to_dict(orient="index"),
    }
