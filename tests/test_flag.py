import numpy as np
from conftest import AD, SIGNOFF

from tv_ads.flag import Params, ad_free_text, flag_item, label_from_dates
from tv_ads.index import Vocabulary, build_index


def labels(per_sentence_dates, sizes, params):
    owner = np.repeat(np.arange(len(sizes)), sizes)
    best = np.repeat(np.asarray(per_sentence_dates), sizes)
    return label_from_dates(best, owner, len(sizes), params)


def test_long_runs_are_ads_short_ones_boilerplate():
    params = Params(min_dates=2, min_share=0.5, min_run_tokens=30, max_gap_tokens=0)
    got = labels([0, 5, 5, 5, 0, 9, 0], [10, 12, 12, 12, 10, 3, 10], params)
    assert got == ["program", "ad", "ad", "ad", "program", "boilerplate", "program"]


def test_gap_filling_joins_a_break_but_not_the_edges():
    params = Params(min_dates=2, min_share=0.5, min_run_tokens=30, max_gap_tokens=8)
    got = labels([3, 0, 3, 3, 0, 3, 0], [8, 15, 20, 20, 6, 20, 5], params)
    assert got == ["boilerplate", "program", "ad", "ad", "ad", "ad", "program"]


def test_tokenless_line_inside_a_break_is_filled():
    params = Params(min_dates=2, min_share=0.5, min_run_tokens=30, max_gap_tokens=0)
    assert labels([3, 0, 3], [20, 0, 20], params) == ["ad", "ad", "ad"]


def test_ads_found_and_news_kept(corpus):
    index = build_index(corpus, 8)
    params = Params(min_dates=2, min_share=0.5, min_run_tokens=30, max_gap_tokens=0)
    rows = flag_item(corpus[0], [index], Vocabulary(), params)
    assert [r.text for r in rows if r.label == "ad"] == AD
    assert {r.label for r in rows if r.text not in AD and r.text != SIGNOFF} == {"program"}
    assert [r.label for r in rows if r.text == SIGNOFF] == ["boilerplate"]
    assert not any(a in ad_free_text(rows) for a in AD)


def test_own_airing_is_not_evidence(corpus):
    index = build_index(corpus[:1], 8)
    rows = flag_item(corpus[0], [index], Vocabulary(), Params(min_dates=1, min_run_tokens=0))
    assert {r.label for r in rows} == {"program"}


def test_unindexed_item_counts_every_date(corpus):
    index = build_index(corpus[1:3], 8)
    rows = flag_item(
        corpus[0], [index], Vocabulary(), Params(min_dates=2, min_share=0.5, min_run_tokens=30, max_gap_tokens=0)
    )
    assert [r.text for r in rows if r.label == "ad"] == AD
