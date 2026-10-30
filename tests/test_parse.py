from datetime import date

import pytest
from conftest import FIXTURES

from tv_ads.parse import Item, load_item, meta_path_for, parse_blocks, read_text, split_sentences, title_date, tokenize

KGO = FIXTURES / "KGO_20120103_013000_ABC_World_News_With_Diane_Sawyer.html"
KNTV = FIXTURES / "KNTV_20120306_013000_NBC_Nightly_News.html"


def test_blocks_keep_recording_minute():
    blocks = parse_blocks(read_text(KGO))
    assert [b.start for b in blocks] == [0, 780, 840, 900]
    assert blocks[0].text.startswith("i'm cheryl jennings. >> from all of us")


def test_empty_minute_is_skipped_but_minutes_stay_aligned():
    item = load_item(KNTV, meta_path_for(KNTV))
    assert sorted({s.minute for s in item.sentences}) == [14, 16]


def test_item_metadata_comes_from_title():
    item = load_item(KGO, meta_path_for(KGO))
    assert item.aired == date(2012, 1, 2)
    assert (item.network, item.station) == ("ABC", "KGO")
    assert item.sentences[0].text == "i'm cheryl jennings."
    assert item.sentences[1].text == "from all of us, thanks for watching."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("u.s. officials say no. >> dr. ho agrees!", ["u.s. officials say no.", "dr. ho agrees!"]),
        ("learn more at isitlowt.com. ask your doctor.", ["learn more at isitlowt.com.", "ask your doctor."]),
        ('j. crew opens. "your voice, your vote." next', ["j. crew opens.", '"your voice, your vote."', "next"]),
    ],
)
def test_split_sentences(text, expected):
    assert [s for _, s in split_sentences(text)] == expected


def test_split_offsets_point_into_text():
    text = "one two. >> three four?"
    assert [text[o : o + len(s)] for o, s in split_sentences(text)] == ["one two.", "three four?"]


def test_tokenize_drops_symbols():
    assert tokenize("Levemir® FlexPen — ask isitlowt.com, it's") == [
        "levemir",
        "flexpen",
        "ask",
        "isitlowt",
        "com",
        "it's",
    ]


def test_title_date_uses_local_date():
    assert title_date("CBS Evening News : KPIX : July 17, 2016 5:30pm-6:31pm PDT") == date(2016, 7, 17)
    assert title_date("no date here") is None


@pytest.mark.parametrize(("clock", "offset"), [("013000", 0), ("223437", 277), ("225950", -10), ("230000", 0)])
def test_half_hour_offset(clock, offset):
    assert Item(f"WJLA_20100611_{clock}_ABC_X", None, "ABC", "WJLA", []).half_hour_offset == offset


def test_sentence_seconds_interpolate_within_minute():
    item = load_item(KGO, meta_path_for(KGO))
    late = [s for s in item.sentences if s.minute == 13]
    assert 780 <= late[0].start_second < late[-1].end_second <= 840
