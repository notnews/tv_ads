"""Command line: ``tv-ads index``, ``flag``, ``validate`` and ``score-codes``."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from tv_ads.flag import Params, ad_free_text, as_dicts, flag_item
from tv_ads.index import RepetitionIndex, Vocabulary, build_index
from tv_ads.parse import Item, load_item, meta_path_for

log = logging.getLogger("tv_ads")


def find_pages(paths: list[Path]) -> list[Path]:
    """Caption pages named on the command line, or found (``*.html``, ``*.html.gz``) under directories."""
    out = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*.html*") if p.name.endswith((".html", ".html.gz"))))
        else:
            out.append(path)
    return out


def _load(path: Path) -> Item:
    return load_item(path, meta_path_for(path))


def load_items(paths: list[Path], workers: int) -> list[Item]:
    """Parse pages in parallel; parsing HTML is the slow step."""
    pages = find_pages(paths)
    with ProcessPoolExecutor(workers) as pool:
        items = list(pool.map(_load, pages, chunksize=16))
    undated = sum(i.aired is None for i in items)
    log.info("parsed %d pages (%d without an air date in the title)", len(items), undated)
    return items


def add_params(parser: argparse.ArgumentParser) -> None:
    """Expose every :class:`Params` field as an option with its tuned default."""
    for field in fields(Params):
        default = getattr(Params(), field.name)
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(default), default=default)


def params_from(args: argparse.Namespace) -> Params:
    """Collect :class:`Params` from parsed options."""
    return Params(**{f.name: getattr(args, f.name) for f in fields(Params)})


def cmd_index(args: argparse.Namespace) -> None:
    """Build and save the repetition index."""
    items = load_items(args.pages, args.workers)
    index = build_index(items, args.n)
    index.save(args.out)
    log.info(
        "indexed %d items, %d distinct %d-grams -> %s", len(index.identifiers), len(index.hashes), args.n, args.out
    )


def cmd_flag(args: argparse.Namespace) -> None:
    """Label sentences and write them with the ad-free text."""
    index = RepetitionIndex.load(args.index)
    items = load_items(args.pages, args.workers)
    params = params_from(args)
    vocabulary = Vocabulary()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / "ad_free.jsonl").open("w") as handle:
        for item in items:
            labels = flag_item(item, index, vocabulary, params)
            rows.extend(as_dicts(labels))
            record = {"identifier": item.identifier, "aired": str(item.aired), "text": ad_free_text(labels)}
            handle.write(json.dumps(record) + "\n")
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "sentences.csv.gz", index=False)
    log.info("labels: %s", table["label"].value_counts().to_dict() if len(table) else {})


def cmd_validate(args: argparse.Namespace) -> None:
    """Tune thresholds on validation days and report scores on test days against Vanderbilt rundowns."""
    from tv_ads.evaluate import estimate_offsets, prepare, scores, sentence_table, split_of, tune
    from tv_ads.vanderbilt import load_rundowns

    items = [i for i in load_items(args.pages, args.workers) if i.aired is not None]
    rundowns = load_rundowns(args.vanderbilt)
    validation = [i for i in items if split_of(i.aired) == "validation"]
    test = [i for i in items if split_of(i.aired) == "test"]
    offsets = estimate_offsets(validation, rundowns)
    vocabulary = Vocabulary()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"items": len(items), "offsets": offsets, "by_n": {}}
    best = None
    for n in args.ngram:
        index = build_index(items, n, vocabulary)
        params, grid = tune(prepare(validation, rundowns, index, vocabulary, offsets))
        grid.to_csv(args.out / f"grid_n{n}.csv", index=False)
        top = grid.iloc[0].to_dict()
        summary["by_n"][n] = {"params": asdict(params), "validation_f1": top["f1"]}
        if best is None or top["f1"] > best[0]:
            best = (top["f1"], n, params, index)
    _, n, params, index = best
    prepared = prepare(test, rundowns, index, vocabulary, offsets)
    table = sentence_table(prepared, params)
    table.to_csv(args.out / "test_sentences.csv.gz", index=False)
    interior = table[table["boundary_s"] >= args.boundary]
    summary["chosen"] = {"n": n, "params": asdict(params)}
    summary["test"] = {
        "newscasts": len(prepared),
        "sentences": len(table),
        "all": scores(table),
        f"interior_{args.boundary}s": scores(interior),
        "by_network": {k: scores(g) for k, g in table.groupby("network")},
        "by_year": {str(k): scores(g) for k, g in table.groupby(pd.to_datetime(table["aired"]).dt.year)},
        "label_by_kind": pd.crosstab(table["kind"], table["label"], values=table["tokens"], aggfunc="sum")
        .fillna(0)
        .astype(int)
        .to_dict(orient="index"),
    }
    summary["validation_newscasts"] = len([i for i in validation if (i.network, i.aired) in rundowns])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("test: %s", summary["test"]["all"])


def cmd_score_codes(args: argparse.Namespace) -> None:
    """Compare ``flag`` output with moment-level codes."""
    from tv_ads.moments import match_moments, moment_scores

    matched = match_moments(pd.read_csv(args.sentences), pd.read_csv(args.sample), pd.read_csv(args.codes))
    result = moment_scores(matched)
    args.out.write_text(json.dumps(result, indent=2, default=float))
    log.info(
        "moments: %d matched of %d; precision %.3f recall %.3f",
        result["matched"],
        result["moments"],
        result["precision"],
        result["recall"],
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``tv-ads`` command."""
    parser = argparse.ArgumentParser(prog="tv-ads", description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="build the repetition index from caption pages")
    index.add_argument("pages", nargs="+", type=Path)
    index.add_argument("--out", type=Path, required=True)
    index.add_argument("--n", type=int, default=8, help="words per n-gram")
    index.set_defaults(func=cmd_index)

    flag = sub.add_parser("flag", help="label sentences and write ad-free text")
    flag.add_argument("pages", nargs="+", type=Path)
    flag.add_argument("--index", type=Path, required=True)
    flag.add_argument("--out", type=Path, required=True)
    add_params(flag)
    flag.set_defaults(func=cmd_flag)

    validate = sub.add_parser("validate", help="tune on validation days, score on test days")
    validate.add_argument("pages", nargs="+", type=Path)
    validate.add_argument("--vanderbilt", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--ngram", type=int, nargs="+", default=[5, 8, 12])
    validate.add_argument("--boundary", type=int, default=20, help="seconds from a row boundary for 'interior'")
    validate.set_defaults(func=cmd_validate)

    codes = sub.add_parser("score-codes", help="score flag output against moment-level codes")
    codes.add_argument("--sentences", type=Path, required=True, help="sentences.csv.gz written by flag")
    codes.add_argument("--sample", type=Path, required=True, help="captions_sample.csv.gz")
    codes.add_argument("--codes", type=Path, required=True, help="captions_codes.csv.gz")
    codes.add_argument("--out", type=Path, required=True, help="JSON file to write")
    codes.set_defaults(func=cmd_score_codes)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
