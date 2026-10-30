import json

import pandas as pd
from conftest import FIXTURES

from tv_ads.cli import main


def test_index_then_flag(tmp_path):
    main(["--workers", "1", "index", str(FIXTURES), "--out", str(tmp_path / "i.npz"), "--n", "8"])
    main(["--workers", "1", "flag", str(FIXTURES), "--index", str(tmp_path / "i.npz"), "--out", str(tmp_path / "o")])
    table = pd.read_csv(tmp_path / "o" / "sentences.csv.gz")
    records = [json.loads(line) for line in (tmp_path / "o" / "ad_free.jsonl").read_text().splitlines()]
    assert len(records) == 2 and set(table["identifier"]) == {r["identifier"] for r in records}
    assert set(table["label"]) == {"program"}
    assert list(table.columns) == [
        "identifier", "minute", "sentence", "start_second", "end_second", "label", "repeat_share", "text"
    ]  # fmt: skip
