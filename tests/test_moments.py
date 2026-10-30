import pandas as pd

from tv_ads.moments import match_moments, moment_scores


def test_moment_matches_spanning_sentence():
    sentences = pd.DataFrame(
        {
            "identifier": ["A", "A", "A", "B"],
            "start_second": [0.0, 30.0, 60.0, 0.0],
            "end_second": [30.0, 60.0, 90.0, 60.0],
            "label": ["program", "ad", "ad", "program"],
            "text": ["news", "buy", "now", "news"],
        }
    )
    sample = pd.DataFrame(
        {"moment_id": [1, 2, 3], "identifier": ["A", "A", "B"], "minute": [0, 1, 5], "second": [45, 10, 0]}
    )
    codes = pd.DataFrame({"moment_id": [1, 2, 3], "kind": ["commercial", "story", "story"]})
    matched = match_moments(sentences, sample, codes)
    assert matched.set_index("moment_id")["text"].fillna("").to_dict() == {1: "buy", 2: "now", 3: ""}
    result = moment_scores(matched)
    assert (result["matched"], result["precision"], result["recall"]) == (2, 0.5, 1.0)
