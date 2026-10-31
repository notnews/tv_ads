import pandas as pd

from tv_ads.gold import gold_scores, match_gold

SENTENCES = pd.DataFrame(
    {
        "identifier": ["A"] * 4,
        "minute": [0, 0, 1, 1],
        "sentence": [0, 1, 2, 3],
        "label": ["program", "ad", "ad", "program"],
        "text": ["Good evening.", "Ask your doctor about Nexium®.", "Visit cialis.com.", "Still ahead, snow."],
    }
)


def test_match_by_text_ignores_case_and_symbols():
    gold = pd.DataFrame(
        {
            "identifier": ["A", "A", "A", "A"],
            "minute": [0, 0, 1, 1],
            "text": ["good evening", "ask your doctor about nexium", "still ahead snow", "not in output"],
            "label": ["news", "commercial", "tease", "promo"],
        }
    )
    result = gold_scores(match_gold(SENTENCES, gold))
    assert (result["scored"], result["unmatched"], result["precision"], result["recall"]) == (3, 1, 1.0, 1.0)


def test_match_by_index_and_unclear_dropped():
    gold = pd.DataFrame(
        {
            "identifier": ["A"] * 3,
            "minute": [0, 1, 1],
            "sentence": [0, 2, 3],
            "label": ["commercial", "promo", "unclear"],
        }
    )
    result = gold_scores(match_gold(SENTENCES, gold))
    assert (result["scored"], result["precision"], result["recall"]) == (2, 1.0, 0.5)
