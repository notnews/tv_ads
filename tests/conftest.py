from datetime import date
from pathlib import Path

import pytest

from tv_ads.parse import Item, Sentence

FIXTURES = Path(__file__).parent / "fixtures"

AD = [
    "sometimes investing opportunities are hard to spot.",
    "the etf market tracker is one more innovative reason serious investors are choosing fidelity.",
    "talk to your doctor about nexium and ask whether the purple pill is right for you today.",
    "possible side effects include headache, diarrhea and abdominal pain, so call your doctor right away.",
]


SIGNOFF = "and that is world news for this evening, i'm diane sawyer, good night."


def make_item(identifier: str, aired: date, texts: list[str]) -> Item:
    sentences = [Sentence(i // 4, i % 4, t, 0, len(t)) for i, t in enumerate(texts)]
    return Item(identifier, aired, identifier.split("_")[3], identifier.split("_")[0], sentences)


def news(tag: str, k: int) -> list[str]:
    return [
        f"reporter {tag} said item {i} of the {tag} story happened in town number {i} {tag} today." for i in range(k)
    ]


@pytest.fixture
def corpus() -> list[Item]:
    """Four nights; the same ad block airs on three of them, on two networks, amid nightly-unique news."""
    nights = [
        ("KGO_20120101_013000_ABC_World_News", date(2012, 1, 1), True),
        ("KPIX_20120102_013000_CBS_Evening_News", date(2012, 1, 2), True),
        ("KGO_20120103_013000_ABC_World_News", date(2012, 1, 3), True),
        ("KGO_20120104_013000_ABC_World_News", date(2012, 1, 4), False),
    ]
    items = []
    for identifier, aired, has_ad in nights:
        tag = f"n{aired.day}"
        texts = news(tag, 3) + (AD if has_ad else []) + news(tag + "b", 3) + [SIGNOFF]
        items.append(make_item(identifier, aired, texts))
    return items
