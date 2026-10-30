from datetime import date

import numpy as np
from conftest import AD, make_item

from tv_ads.index import RepetitionIndex, Vocabulary, build_index, ngram_hashes


def test_ngram_hash_depends_on_words_not_position_or_run():
    a = Vocabulary().ids("x y a b c z".split())
    b = Vocabulary().ids("a b c".split())
    assert ngram_hashes(a, 3)[2] == ngram_hashes(b, 3)[0]
    assert len(set(ngram_hashes(a, 3).tolist())) == 4
    assert len(ngram_hashes(b, 4)) == 0


def test_counts_distinct_dates_and_networks(corpus):
    index = build_index(corpus, 5)
    vocabulary = Vocabulary()
    from tv_ads.parse import tokenize

    probe = ngram_hashes(vocabulary.ids(tokenize(AD[1])), 5)[:1]
    dates, nets, span = index.lookup(probe)
    assert (dates[0], nets[0], span[0]) == (3, 2, 2)
    absent = ngram_hashes(vocabulary.ids("never seen words in this corpus".split()), 5)
    assert index.lookup(absent)[0].tolist() == [0, 0]


def test_same_date_counts_once():
    items = [make_item(f"KGO_2012010{i}_013000_ABC_X", date(2012, 1, 1), AD) for i in range(3)]
    assert build_index(items, 5).n_dates.max() == 1


def test_round_trip(tmp_path, corpus):
    index = build_index(corpus, 8)
    index.save(tmp_path / "i.npz")
    back = RepetitionIndex.load(tmp_path / "i.npz")
    assert back.n == 8 and back.identifiers == index.identifiers
    assert np.array_equal(back.hashes, index.hashes) and np.array_equal(back.n_dates, index.n_dates)
