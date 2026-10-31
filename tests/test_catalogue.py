from datetime import date

import pandas as pd
from conftest import AD, make_item, news

from tv_ads.catalogue import Catalogue, build_catalogue, catalogue_index
from tv_ads.flag import Params, flag_item
from tv_ads.index import Vocabulary, build_index

PROMO = [  # a promo is a spot like any other here
    "tonight on abc, an all new episode of the bachelor at 8/7c.",
    "and join us tomorrow morning on good morning america for the full interview with the families.",
    "only on abc, it is the season finale everyone has been waiting for all year long.",
]
PARAMS = Params(min_dates=2, min_share=0.5, min_run_tokens=30, max_gap_tokens=0)


def nights():
    out = []
    for day in range(1, 7):
        order = [AD, PROMO] if day % 2 else [PROMO, AD]
        texts = news(f"d{day}", 3) + order[0] + order[1] + news(f"e{day}", 3)
        out.append(make_item(f"KGO_201201{day:02d}_013000_ABC_World_News", date(2012, 1, day), texts))
    return out


def test_breaks_split_into_spots_that_keep_their_lines():
    items = nights()
    vocabulary = Vocabulary()
    spots = build_catalogue(items, [build_index(items, 8, vocabulary)], vocabulary, PARAMS)
    assert sorted(spots["text"]) == sorted([" ".join(AD), " ".join(PROMO)])
    assert spots["airings"].tolist() == [6, 6]
    assert (spots["first_aired"].tolist(), spots["networks"].tolist()) == (["2012-01-01"] * 2, ["ABC"] * 2)


def test_catalogue_alone_flags_a_new_page(tmp_path):
    items = nights()
    vocabulary = Vocabulary()
    spots = build_catalogue(items, [build_index(items, 8, vocabulary)], vocabulary, PARAMS)
    spots.to_csv(tmp_path / "spots.csv.gz", index=False)
    catalogue = Catalogue.read(tmp_path / "spots.csv.gz")
    page = make_item("KGO_20130101_013000_ABC_World_News", date(2013, 1, 1), news("new", 3) + AD + news("x", 2))
    rows = flag_item(page, [catalogue.index], Vocabulary(), PARAMS)
    assert [r.text for r in rows if r.label == "ad"] == AD
    ad_spot = int(spots.loc[spots["text"] == " ".join(AD), "spot"].iloc[0])
    assert {r.spot for r in rows if r.label == "ad"} == {ad_spot}
    assert {r.spot for r in rows if r.label != "ad"} == {-1}


def test_signoff_next_to_breaks_stays_out():
    items = nights()
    signoff = "reporter jane doe, abc news, new york for world news tonight."
    for k, item in enumerate(items):
        texts = [s.text for s in item.sentences]
        texts.insert(3, signoff)
        texts += news(f"z{k}", 2) + [signoff] + news(f"y{k}", 2) + [signoff] + news(f"w{k}", 2)
        items[k] = make_item(item.identifier, item.aired, texts)
    vocabulary = Vocabulary()
    spots = build_catalogue(items, [build_index(items, 8, vocabulary)], vocabulary, PARAMS)
    assert not spots["text"].str.contains("jane doe").any()


def test_shared_ngram_goes_to_most_aired_spot():
    spots = pd.DataFrame(
        {
            "spot": [0, 1],
            "airings": [9, 3],
            "first_aired": ["2012-01-01"] * 2,
            "last_aired": ["2012-02-01"] * 2,
            "networks": ["ABC;CBS", "NBC"],
            "source": ["auto", "auto"],
            "text": ["one two three four five six seven eight", "one two three four five six seven eight nine"],
        }
    )
    index = catalogue_index(spots, 8)
    assert sorted(zip(index.n_dates.tolist(), index.spot_ids.tolist())) == [(3, 1), (9, 0)]


def test_packaged_catalogue_loads():
    catalogue = Catalogue.read()
    assert len(catalogue.spots) > 0 and len(catalogue.index.hashes) > 0
    auto = catalogue.spots[catalogue.spots["source"] != "human"]
    assert (auto["airings"] >= 2).all() and (auto["tokens"] >= 8).all()
    assert set(catalogue.spots["source"]) <= {"auto", "human", "both"}


def gold_rows():
    rows = []
    for k, (label, text) in enumerate(
        [("news", "reporter: the storm hit at dawn."), *[("commercial", t) for t in AD], ("news", "good evening.")]
    ):
        rows.append({"identifier": "KGO_20120110_013000_ABC_X", "minute": 5, "sentence": k, "text": text,
                     "label": label, "window": "w001"})  # fmt: skip
    return pd.DataFrame(rows)


def test_human_spots_and_merge():
    from tv_ads.catalogue import add_human, auto_only, human_spots, without_window

    human = human_spots(gold_rows())
    assert len(human) == 1 and human["text"].iloc[0] == " ".join(AD)
    assert (human["first_aired"].iloc[0], human["networks"].iloc[0]) == ("2012-01-10", "ABC")
    items = nights()
    vocabulary = Vocabulary()
    auto = build_catalogue(items, [build_index(items, 8, vocabulary)], vocabulary, PARAMS)
    merged = add_human(auto, human, n=8)
    assert sorted(merged["source"]) == ["auto", "both"] and len(merged) == len(auto)
    assert merged.loc[merged["source"] == "both", "windows"].tolist() == ["w001"]
    assert sorted(without_window(merged, "w001")["source"]) == ["auto", "auto"]
    alone = add_human(auto.iloc[0:0], human, n=8)
    assert alone["source"].tolist() == ["human"] and without_window(alone, "w001").empty
    assert auto_only(alone).empty
    assert len(add_human(alone, human, n=8)) == 1


def test_human_spot_matches_under_any_min_dates():
    from tv_ads.catalogue import human_spots

    spots = human_spots(gold_rows()).assign(spot=0)
    index = catalogue_index(spots, 8)
    page = make_item("KGO_20130101_013000_ABC_World_News", date(2013, 1, 1), news("new", 3) + AD + news("x", 2))
    rows = flag_item(page, [index], Vocabulary(), Params(min_dates=50, min_share=0.5, min_run_tokens=30))
    assert [r.text for r in rows if r.label == "ad"] == AD
