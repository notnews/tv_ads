"""Command line: ``tv-ads index``, ``flag``, ``catalogue``, ``validate`` and ``score-codes``."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

from tv_ads.catalogue import PACKAGED_PATH, Catalogue
from tv_ads.flag import Params, ad_free_text, as_dicts, flag_item
from tv_ads.index import DEFAULT_N, RepetitionIndex, Vocabulary, build_index
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


def indexes_from(args: argparse.Namespace) -> list[RepetitionIndex]:
    """User-built indexes plus, unless ``--no-catalogue``, the catalogue (packaged or ``--catalogue``)."""
    indexes = [RepetitionIndex.load(path) for path in args.index]
    if not args.no_catalogue:
        catalogue = Catalogue.read(args.catalogue)
        indexes.append(catalogue.index)
        log.info("catalogue: %d spots", len(catalogue.spots))
    if not indexes:
        raise SystemExit("nothing to match against: give --index or drop --no-catalogue")
    return indexes


def cmd_flag(args: argparse.Namespace) -> None:
    """Label sentences and write them with the ad-free text."""
    indexes = indexes_from(args)
    items = load_items(args.pages, args.workers)
    params = params_from(args)
    vocabulary = Vocabulary()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / "ad_free.jsonl").open("w") as handle:
        for item in items:
            labels = flag_item(item, indexes, vocabulary, params)
            rows.extend(as_dicts(labels))
            record = {"identifier": item.identifier, "aired": str(item.aired), "text": ad_free_text(labels)}
            handle.write(json.dumps(record) + "\n")
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "sentences.csv.gz", index=False)
    log.info("labels: %s", table["label"].value_counts().to_dict() if len(table) else {})


def cmd_catalogue_build(args: argparse.Namespace) -> None:
    """Build the automatic part of the spot catalogue from a corpus."""
    from tv_ads.catalogue import build_catalogue, summary

    items = load_items(args.pages, args.workers)
    vocabulary = Vocabulary()
    index = build_index(items, args.n, vocabulary)
    spots = build_catalogue(items, [index], vocabulary, params_from(args))
    spots.to_csv(args.out, index=False)
    log.info("catalogue %s -> %s (%d bytes)", summary(spots), args.out, args.out.stat().st_size)


def _air_dates(identifiers: set[str], pages: list[Path]) -> dict:
    found = {}
    for page in find_pages(pages):
        item_id = page.name.split(".html")[0]
        if item_id in identifiers:
            found[item_id] = _load(page).aired
    return found


def cmd_catalogue_add(args: argparse.Namespace) -> None:
    """Add hand-labelled commercials to a catalogue."""
    from tv_ads.catalogue import add_human, human_spots, summary

    gold = pd.read_csv(args.gold)
    aired = _air_dates(set(gold["identifier"]), args.pages) if args.pages else None
    spots = add_human(pd.read_csv(args.catalogue or PACKAGED_PATH, keep_default_na=False), human_spots(gold, aired))
    spots.to_csv(args.out, index=False)
    log.info("catalogue %s -> %s (%d bytes)", summary(spots), args.out, args.out.stat().st_size)


def _per_group(table: pd.DataFrame, times: pd.DataFrame, keys: list[str]) -> dict:
    from tv_ads.evaluate import scores, summarise_time

    table = table.assign(year=pd.to_datetime(table["aired"]).dt.year)
    out = {}
    for key, group in table.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        name = "_".join(str(k) for k in key)
        mask = np.logical_and.reduce([times[k] == v for k, v in zip(keys, key, strict=True)])
        out[name] = {**scores(group), **summarise_time(times[mask]), "newscasts": int(group["identifier"].nunique())}
    return out


def cmd_validate(args: argparse.Namespace) -> None:
    """Tune thresholds on validation days and report scores on test days against Vanderbilt rundowns."""
    from tv_ads.catalogue import build_catalogue, summary
    from tv_ads.evaluate import (
        assign_splits,
        estimate_offsets,
        macro,
        prepare,
        scores,
        sentence_table,
        summarise_time,
        time_recall,
        tune,
    )
    from tv_ads.vanderbilt import load_rundowns

    items = [i for i in load_items(args.pages, args.workers) if i.aired is not None]
    rundowns = load_rundowns(args.vanderbilt)
    split = assign_splits(items)
    validation = [i for i in items if split[i.identifier] == "validation"]
    test = [i for i in items if split[i.identifier] == "test"]
    offsets = estimate_offsets(validation, rundowns)
    vocabulary = Vocabulary()
    args.out.mkdir(parents=True, exist_ok=True)
    pages = pd.Series([i.network for i in items]).value_counts().to_dict()
    result: dict = {"items": len(items), "pages_by_network": pages, "offsets": offsets, "by_n": {}}
    best = None
    for n in args.ngram:
        index = build_index(items, n, vocabulary)
        params, grid = tune(prepare(validation, rundowns, [index], vocabulary, offsets))
        grid.to_csv(args.out / f"grid_n{n}.csv", index=False)
        top = grid.iloc[0]
        own_best = {c[3:]: float(grid[c].max()) for c in grid.columns if c.startswith("f1_")}
        shared = {c[3:]: float(top[c]) for c in grid.columns if c.startswith("f1_")}
        result["by_n"][n] = {
            "params": asdict(params),
            "validation_macro_f1": float(top["macro_f1"]),
            "validation_f1_shared_setting": shared,
            "validation_f1_best_setting_per_network": own_best,
        }
        if best is None or top["macro_f1"] > best[0]:
            best = (top["macro_f1"], n, params, index)
    _, n, params, index = best
    prepared = prepare(test, rundowns, [index], vocabulary, offsets)
    table = sentence_table(prepared, params)
    times = time_recall(prepared, params, offsets)
    table.to_csv(args.out / "test_sentences.csv.gz", index=False)
    times.to_csv(args.out / "test_break_time.csv", index=False)
    per_network = _per_group(table, times, ["network"])
    result["chosen"] = {"n": n, "params": asdict(params)}
    result["test"] = {
        "newscasts": len(prepared),
        "sentences": len(table),
        "macro_f1": macro(per_network),
        "pooled": {**scores(table), **summarise_time(times)},
        f"interior_{args.boundary}s": scores(table[table["boundary_s"] >= args.boundary]),
        "by_network": per_network,
        "by_network_year": _per_group(table, times, ["network", "year"]),
        "label_by_kind": pd.crosstab(table["kind"], table["label"], values=table["tokens"], aggfunc="sum")
        .fillna(0)
        .astype(int)
        .to_dict(orient="index"),
    }
    held_out = build_index(validation, n, vocabulary)
    spots = build_catalogue(validation, [held_out], vocabulary, params)
    catalogue = Catalogue(spots, Catalogue.index_for(spots, n))
    alone = sentence_table(prepare(test, rundowns, [catalogue.index], vocabulary, offsets), params)
    result["catalogue_only"] = {
        "built_from": "validation-day pages only",
        **summary(spots),
        "macro_f1": macro({k: scores(g) for k, g in alone.groupby("network")}),
        "pooled": scores(alone),
        "by_network": {k: scores(g) for k, g in alone.groupby("network")},
    }
    result["validation_newscasts"] = len([i for i in validation if (i.network, i.aired) in rundowns])
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    log.info(
        "test macro F1 %.3f; by network %s",
        result["test"]["macro_f1"],
        {k: round(v["f1"], 3) for k, v in per_network.items()},
    )


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


def cmd_score_gold(args: argparse.Namespace) -> None:
    """Score against hand labels, flagging each window without the human spots taken from that window."""
    from tv_ads.catalogue import auto_only, catalogue_index, without_window
    from tv_ads.gold import gold_scores, match_gold

    gold = pd.read_csv(args.gold)
    spots = pd.read_csv(args.catalogue or PACKAGED_PATH, keep_default_na=False)
    corpus = [RepetitionIndex.load(path) for path in args.index]
    pages = {p.name.split(".html")[0]: p for p in find_pages(args.pages)}
    params = params_from(args)
    vocabulary = Vocabulary()
    auto_index = catalogue_index(auto_only(spots), DEFAULT_N, vocabulary)
    matched = {"auto_only": [], "with_human_leave_window_out": []}
    for window, rows in gold.groupby("window"):
        identifier = rows["identifier"].iloc[0]
        if identifier not in pages:
            log.warning("no page for %s (window %s)", identifier, window)
            continue
        item = _load(pages[identifier])
        held_out = catalogue_index(without_window(spots, str(window)), DEFAULT_N, vocabulary)
        for name, index in (("auto_only", auto_index), ("with_human_leave_window_out", held_out)):
            labels = pd.DataFrame(as_dicts(flag_item(item, corpus + [index], vocabulary, params)))
            matched[name].append(match_gold(labels, rows))
    result = {}
    for name, parts in matched.items():
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        frame.to_csv(args.out.with_name(f"{args.out.stem}_{name}.csv"), index=False)
        result[name] = gold_scores(frame) if len(frame) else {}
    args.out.write_text(json.dumps(result, indent=2, default=float))
    for name, r in result.items():
        if r:
            log.info("%s: %d scored; precision %.3f recall %.3f", name, r["scored"], r["precision"], r["recall"])


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``tv-ads`` command."""
    parser = argparse.ArgumentParser(prog="tv-ads", description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="build the repetition index from caption pages")
    index.add_argument("pages", nargs="+", type=Path)
    index.add_argument("--out", type=Path, required=True)
    index.add_argument("--n", type=int, default=DEFAULT_N, help="words per n-gram")
    index.set_defaults(func=cmd_index)

    flag = sub.add_parser("flag", help="label sentences and write ad-free text")
    flag.add_argument("pages", nargs="+", type=Path)
    flag.add_argument("--index", type=Path, action="append", default=[], help="a corpus index; repeatable")
    flag.add_argument("--catalogue", type=Path, help="spots file (default: the one packaged with tv_ads)")
    flag.add_argument("--no-catalogue", action="store_true", help="match only against --index")
    flag.add_argument("--out", type=Path, required=True)
    add_params(flag)
    flag.set_defaults(func=cmd_flag)

    cat = sub.add_parser("catalogue", help="build or extend the spot catalogue").add_subparsers(
        dest="action", required=True
    )
    build = cat.add_parser("build", help="detect recurring spots in caption pages")
    build.add_argument("pages", nargs="+", type=Path)
    build.add_argument("--out", type=Path, required=True, help="spots.csv.gz to write")
    build.add_argument("--n", type=int, default=DEFAULT_N, help="words per n-gram")
    add_params(build)
    build.set_defaults(func=cmd_catalogue_build)
    add = cat.add_parser("add", help="add hand-labelled commercials (gold file) to a catalogue")
    add.add_argument("--gold", type=Path, required=True, help="identifier, minute, sentence, text, label, window")
    add.add_argument("--catalogue", type=Path, help="catalogue to extend (default: the packaged one)")
    add.add_argument("--pages", type=Path, nargs="*", help="caption pages, for local air dates")
    add.add_argument("--out", type=Path, required=True, help="spots.csv.gz to write")
    add.set_defaults(func=cmd_catalogue_add)

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

    gold = sub.add_parser("score-gold", help="score against hand labels, leaving each window's own spots out")
    gold.add_argument("pages", nargs="+", type=Path, help="caption pages of the labelled items")
    gold.add_argument("--gold", type=Path, required=True, help="identifier, minute, sentence, text, label, window")
    gold.add_argument("--catalogue", type=Path, help="spots file (default: the packaged one)")
    gold.add_argument("--index", type=Path, action="append", default=[], help="a corpus index; repeatable")
    gold.add_argument("--out", type=Path, required=True, help="JSON file to write (matched rows go beside it)")
    add_params(gold)
    gold.set_defaults(func=cmd_score_gold)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
