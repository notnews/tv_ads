from datetime import date

import numpy as np
import pandas as pd
import pytest

from tv_ads.evaluate import boundary_distance, split_of
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


def test_split_is_by_date_and_balanced():
    days = pd.date_range("2010-01-01", "2021-12-31").date
    splits = [split_of(d) for d in days]
    assert set(splits) == {"validation", "test"}
    assert abs(splits.count("test") / len(days) - 0.5) < 0.03
