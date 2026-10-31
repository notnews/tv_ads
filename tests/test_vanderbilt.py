from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_item

from tv_ads.evaluate import assign_splits, boundary_distance
from tv_ads.vanderbilt import load_rundowns, row_kind


@pytest.mark.parametrize(
    ("title", "kind"),
    [
        ("(Commercial: FreeStyle; USAA; BP/Gulf Coast; Aleve.)", "commercial"),
        ("Commercial: Geico", "commercial"),
        ("Commercials", "commercial"),
        ("Upcoming Items & COMMERCIALS: Progressive; Prevagen", "break"),
        ("Upcoming Items (Tampa: Scott Pelley)", "furniture"),
        ("Preview/Introduction\tScott Pelley (Tampa, Florida)", "furniture"),
        ("Good Night", "furniture"),
        ("Space: Commercial Travel", "story"),
        ("Aviation / Commercial Drones", "story"),
    ],
)
def test_row_kind(title, kind):
    assert row_kind(title) == kind


@pytest.fixture
def rundowns(tmp_path):
    rows = pd.DataFrame(
        {
            "channel": "CBS",
            "date": "Tuesday, Aug 28, 2012",
            "program_title": "CBS Evening News",
            "broadcast_time": [
                "05:30:20 pm-05:31:10 pm",
                "05:31:10 pm-05:37:50 pm",
                "05:42:20 pm-05:45:00 pm",
                "05:58:40 pm-05:58:40 pm",
            ],
            "broadcast_title": ["Preview/Introduction", "Hurricane Isaac", "(Commercial: Aleve.)", "Good Night"],
        }
    )
    path = tmp_path / "v.csv"
    rows.to_csv(path, index=False)
    return load_rundowns(str(path))


def test_rundown_times_from_half_hour(rundowns):
    r = rundowns[("CBS", date(2012, 8, 28))]
    assert r.kind_at(np.array([0, 20, 70, 500, 800, 899, 900, 1720])) == [
        "outside",
        "furniture",
        "story",
        "gap",
        "commercial",
        "commercial",
        "gap",
        "outside",
    ]


def test_boundary_distance(rundowns):
    r = rundowns[("CBS", date(2012, 8, 28))]
    assert boundary_distance(r, np.array([75.0, 750.0])).tolist() == [5.0, 10.0]


def test_split_balances_every_network_year():
    days = pd.date_range("2012-01-01", "2013-12-31").date
    items = [make_item(f"S_{d:%Y%m%d}_013000_{net}_X", d, ["x."]) for d in days for net in ("ABC", "NBC")]
    items += [make_item(f"S_{d:%Y%m%d}_013000_CBS_X", d, ["x."]) for d in days[:3]]
    split = assign_splits(items)
    for net, year in [("ABC", 2012), ("NBC", 2013)]:
        sides = [split[i.identifier] for i in items if i.network == net and i.aired.year == year]
        assert abs(sides.count("test") - sides.count("validation")) <= 1
    assert sorted(split[i.identifier] for i in items if i.network == "CBS") == ["test", "validation", "validation"]


def test_scores_count_break_rows_and_skip_furniture():
    from tv_ads.evaluate import scores

    table = pd.DataFrame(
        {
            "tokens": [10, 10, 10, 10, 50],
            "kind": ["commercial", "break", "story", "story", "furniture"],
            "label": ["ad", "program", "ad", "program", "ad"],
        }
    )
    got = scores(table)
    assert (got["precision"], got["recall"], got["tokens"], got["break_share"]) == (0.5, 0.5, 40, 0.5)
