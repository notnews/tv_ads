# tv_ads: find commercials in TV news closed captions

[![CI](https://github.com/notnews/tv_ads/actions/workflows/ci.yml/badge.svg)](https://github.com/notnews/tv_ads/actions/workflows/ci.yml)

The Internet Archive's TV News Archive keeps closed captions for decades of newscasts. Caption text has no
marks for commercials. A paid spot for Nexium runs straight on from a reporter's sign-off:

> ... clayton sandell, abc news, denver. ... and still ahead, the first blast of winter ... [ male announcer ]
> don't have the hops for hoops with your buddies? ... learn more at isitlowt.com.

`tv_ads` labels each caption sentence as `ad`, `boilerplate` or `program` and writes the text with the ads
taken out. It needs no hand labels. It ships with a database of 9,599 ad spots from the 2010–2021 evening news
of ABC, CBS and NBC, detected automatically and from hand labels, so a single page can be cleaned straight away. For other channels or years, you build an
index from your own corpus. That works on any channel, including the broadcast networks, where the only
earlier source of commercial times, the Stanford Cable TV News Analyzer, has no coverage. (Stanford's
intervals cover CNN, Fox News and MSNBC and are found from the video.)

## Quick start

```bash
uv sync
uv run tv-ads flag KGO_20120103_013000_ABC_World_News_With_Diane_Sawyer.html.gz --out flagged
```

This matches the page against the packaged catalogue. It writes `flagged/sentences.csv.gz` and
`flagged/ad_free.jsonl`, and takes about a second. An excerpt of `sentences.csv.gz`:

| minute | label | spot | text |
|---|---|---|---|
| 13 | program | -1 | and still ahead tonight, we have new details on an amazing rescue, ten brave strangers ... |
| 13 | program | -1 | sometimes investing opportunities are hard to spot. *(missed: first line of the Fidelity spot)* |
| 13 | ad | 3746 | fidelity's etf market tracker shows you the big picture on how different asset classes ... |
| 14 | ad | -1 | and explore your next investing idea. *(no match; filled in because it sits inside the break)* |
| 14 | ad | 1721 | and we've established a 500 million dollar fund so independent scientists can study the gulf ... |
| 15 | ad | 66 | for many, prescription nexium not only provides 24-hour heartburn relief, but can also ... |
| 15 | ad | 12 | possible side effects include headache, diarrhea and abdominal pain. |
| 16 | program | -1 | reporter: it was a new year's eve near-tragedy. |

`spot` is the catalogue row a sentence matched. This page aired in the corpus the catalogue was built from;
see the validation section for how well the catalogue does on pages it has never seen.

`sentences.csv.gz` has these columns: `identifier`, `minute`, `sentence` (position in the item),
`start_second` and `end_second` (seconds from the recording start, interpolated within the minute), `label`,
`spot`, `repeat_share` and `text`. `ad_free.jsonl` has one `{identifier, aired, text}` record per item.

**What the catalogue knows.** It holds only spots that aired, captioned, at least twice in the corpus it was
built from: 2010–2021 ABC, CBS and NBC evening newscasts, mostly from the San Francisco stations. Local spots,
spots from other channels and spots from other years are not in it. For those, build an index from your own
pages. The catalogue is then used alongside it, with each n-gram taking the higher of the two counts:

```bash
uv run tv-ads index my_pages/ --out my_index.npz
uv run tv-ads flag my_pages/ --index my_index.npz --out flagged            # your index + the catalogue
uv run tv-ads flag my_pages/ --index my_index.npz --no-catalogue --out x   # your index only
uv run tv-ads catalogue build my_pages/ --out my_spots.csv.gz              # a catalogue of your own
uv run tv-ads catalogue add --gold gold.csv --catalogue my_spots.csv.gz --pages my_pages/ --out my_spots.csv.gz
```

`catalogue add` folds hand-labelled commercials into a catalogue. The gold file has the columns `identifier`,
`minute`, `sentence`, `text`, `label` and `window`. Consecutive `commercial` sentences of one item, with no
other label between them, form one spot. `--pages` supplies local air dates; without it, the identifier's UTC
date is used.

## How it works

**Why repetition finds ads.** A commercial is produced once and then aired, word for word, on many nights and
on several networks. News text is written for one night and mostly does not come back. So a stretch of
captions that also appears on *other air dates* is very likely an ad or a network promo.

1. **Parse.** Each item's details page holds one caption block per minute (`div.tvcol`, with the snippet in
   `div.snipin.nosel`, the selector archive_news_cc uses). The block's share link (`/start/840/end/900`)
   gives the minute, so minutes with no captions are skipped without shifting the rest. Blocks are split into
   sentences at `.`, `?`, `!` and at the `>>` speaker-change mark. A period after a short abbreviation
   (`u.s.`, `dr.`) or a single letter does not end a sentence. Words are lower-cased tokens; `®`, `♪` and the
   dots in `cialis.com` drop out. The air date is the local date in the item title (`... : KPIX : July 17, 2016
   5:30pm-6:31pm PDT`), not the UTC date in the metadata.
2. **Index.** Every run of *n* consecutive words (an *n*-gram; *n* = 12 by default) in every item is hashed.
   For each distinct *n*-gram the index records how many distinct air dates and networks it appeared on.
   Repeats on the same date count once, so a story that two stations carry on the same night is not evidence.
3. **Score.** For a sentence, each word gets the largest "other dates" count among the *n*-grams that cover
   it, leaving out the item's own airing. A sentence *repeats* when at least `min_share` of its words have a
   count of at least `min_dates`.
4. **Label.** Consecutive repeating sentences form runs. Stretches of at most `max_gap_tokens` words between
   two repeating sentences are filled in, so a garbled caption line does not split an ad break. A run of at
   least `min_run_tokens` words is `ad`; a shorter run is `boilerplate`. Reporter sign-offs and `reporter:`
   lines at a run's edges are trimmed off first, because they are news airtime. Boilerplate covers sign-offs,
   hand-offs and sound bites replayed on another night. It is kept in the ad-free text, because it is part of
   the programme.

**The catalogue.** `tv-ads catalogue build` runs steps 1–4 over a corpus and groups the `ad` sentences into
spots.
- A *line* is the token string of one `ad` sentence. A line is kept when it aired on two or more dates and at
  least half its occurrences were inside ad runs. The second rule drops a reporter's sign-off that sometimes
  sits next to a break but mostly ends a story.
- Two lines join the same spot when one follows the other, within a break, on more than half the dates the
  more-aired line was seen on. A spot's lines air together and in order. The spot after a break boundary
  changes from night to night, so breaks fall apart into spots.
- A generic line that many spots share follows no single spot, so it becomes a spot of its own.
- Each spot records its text, its airings (distinct dates on which at least half its lines aired together),
  its first and last air date, its networks, its `source` and the gold `windows` it came from.

`source` is `auto` (detected), `human` (from hand labels) or `both`:
- An auto spot with at least half its *n*-grams inside a hand-labelled break becomes `both`.
- A hand-labelled break is added as a `human` spot unless auto spots already cover 90% of it.
- Human and `both` spots count as 1,000 airings, so they match under any `min_dates`.

At load time, the spots' text is hashed into an *n*-gram index whose counts are the airings. That index is
then used like any other.

The packaged file `src/tv_ads/data/spots.csv.gz` is 791 KB, built from the 4,818 pages present on 2026-10-30
(mostly ABC; it will be rebuilt when the fetch completes) plus the hand labels so far. It holds 9,599 spots:
9,563 `auto`, 34 `both` and 2 `human`, the last being two whole annotated breaks. The median spot is 24
words. Spots are often fragments of a commercial, not the whole thing: the BP spot above matched three
catalogue rows. That does not hurt matching, but a row is not a one-to-one count of commercials. Some
catalogue rows, from the most aired down:

| spot | airings | first – last | networks | text |
|---|---|---|---|---|
| 0 | 778 | 2010-01-02 – 2016-12-11 | ABC;CBS;NBC | side effects may include headache, upset stomach, delayed backache, or muscle ache. |
| 2 | 564 | 2013-09-20 – 2021-12-30 | ABC;CBS;NBC | eliquis can cause serious and in rare cases fatal bleeding. ... |
| 10 | 321 | 2010-03-19 – 2018-09-21 | ABC;CBS | common side effects are dizziness, sleepiness, weight gain ... |
| 200 | 61 | 2010-12-14 – 2011-12-26 | ABC;CBS | her morning begins with arthritis pain. that's a coffee and ... |
| 1000 | 22 | 2011-09-27 – 2011-12-15 | ABC;CBS | all-natural benefiber, the fiber supplement that's taste-free ... |

**Commercials versus network promos.** Both are labelled `ad`, because no split could be made credibly.
Promos barely exist as recurring caption copy in this corpus:
- Searching the 7,587-row trial catalogue for programme names, "tonight on", "only on abc", network news
  titles and "/c" air times found 8 rows. About 4 of them were promos ("tomorrow on "abc's world news tonight
  with david muir."", an "american bandstand" anniversary spot). The rest were sign-offs or sponsorship
  lines.
- In 150 random ABC newscasts (2012–2015), 15 sentences matched promo cues. Nearly all were anchor-read
  programme notes ("will it work, game seven of the nba finals 9:00 eastern right here on abc.") or local-news
  teases ("only on abc7 news"). These are unique copy and labelled `program`.
- Vanderbilt names a programme in 13% of break rows (`"Good Morning America"`, `"60 Minutes"`). So promos air,
  but they seem to be uncaptioned or not repeated verbatim.

A classifier would have almost nothing to classify, and too few cases to be scored.

**Exact n-grams rather than MinHash or edit distance.** I compared each alternative with the same split and
the same tuning (validation days for every setting, test days for the numbers):

- *Sentence-level MinHash with LSH* ([datasketch](https://github.com/ekzhu/datasketch) 2.0.0, released
  2026-07-05; repository active and not archived). This used 64 permutations over word 3-grams, at Jaccard
  thresholds of 0.5 and 0.7. On 1,197 pages, test F1 was 0.841 for 8-grams against 0.842 and 0.840 for
  MinHash: no gain (`scripts/compare_minhash.py`, `validation/compare_minhash.txt`; that run used the earlier
  pooled objective).
- *Shorter n-grams* are more tolerant of caption errors. Validation macro F1 was 0.733 for *n* = 5, 0.736 for
  8 and 0.738 for 12: no gain.
- *Edit-distance verification against the catalogue* ([rapidfuzz](https://github.com/rapidfuzz/RapidFuzz)
  3.14.6, released 2026-08-30; active). Each sentence that the exact route leaves unmatched gets candidate
  spots from shared word 4-grams. It is then scored against each candidate by `partial_ratio`: 100 × (1 −
  normalised Indel distance) over the best-matching window of the spot. The threshold (80, 85, 90 or 95) was
  chosen on validation days, and 80 won. The catalogue came from validation days and was tested on the
  other half (`scripts/compare_fuzzy.py`, `validation/compare_fuzzy.txt`):

  | test, catalogue only (*n* = 8) | ABC P / R | CBS P / R | NBC P / R | macro F1 | time for 1,408 pages |
  |---|---|---|---|---|---|
  | exact *n*-grams | 0.936 / 0.769 | 0.936 / 0.650 | 0.984 / 0.369 | 0.716 | 3.2 s |
  | + rapidfuzz ≥ 80 | 0.919 / 0.797 | 0.912 / 0.677 | 0.953 / 0.393 | 0.729 | 8.1 s |

  Edit-distance matching gains 2–3 points of recall on every network but loses 2–3 points of precision. The
  bar was more recall *without* losing precision, so it is not adopted.

## Validation

**Reference.** The Vanderbilt Television News Archive indexes every network evening newscast as timed rows
in Central time. From 2010 on, commercial breaks have their own rows (`(Commercial: Progressive; Just for Men;
...)`). In some years, seen from 2019, a break is a single row titled `Upcoming Items & COMMERCIALS: ...`.
Teases and furniture have their own rows too (`Upcoming Items`, `Preview/Introduction`, `Good Night`).
`scripts/export_vanderbilt.R` exports the 173,788 ABC/CBS/NBC rows for 2010–2021 from notwork_news's
`archive_all.rds`. The underlying Vanderbilt export is access-restricted on Dataverse, so it is not
redistributed here.

**Lining up the clocks.** A sentence's time is its minute plus its character position within that minute's
text, scaled to 60 seconds. Two shifts are then applied. The first is the recording's start relative to the
half hour, read from the identifier: `WJLA_20100611_223437_...` began 277 s late. The second is a per-network
offset. The offset is estimated on validation days only, without using the detector: it is the shift that
best puts sentences beginning `reporter:` in story rows, and sentences with `announcer`, `♪` or a `.com`
address in break rows. The estimates were 0 s (ABC), −10 s (CBS) and 0 s (NBC).

**Resolution.** Sentence times are interpolated within a minute. The two sources also record different feeds:
Vanderbilt records Nashville, and most sampled captions come from the San Francisco stations' West Coast
airing. Expect the reference to be off by 10–30 s at every boundary, and occasionally by a minute or more.

**Split.** Within each network-year, air dates are put in hash order and alternate between validation and
test. Every network-year, however small, is therefore split in half. Validation days choose the offsets, the
*n*-gram size (5, 8 or 12) and the thresholds (a 504-cell grid). The objective is the **macro F1: the mean of
the ABC, CBS and NBC F1s**, so ABC's larger share of words does not decide the setting. The index itself uses
no labels, so it is built from all pages. The catalogue-only check builds both its index and its catalogue
from validation days alone.

**Metric.** The positive class is caption words in `commercial` or `break` rows; the negative class is
caption words in story rows. Furniture rows, and words outside the first and last rows, are reported but not
scored. Two time-based recalls are also reported. They answer a question word-based recall cannot: how much
break time has no caption text at all? For each second of break time in the rundown:

- *captioned* means its recording minute has a caption block;
- *covered* means it lies inside the interpolated span of a sentence labelled `ad`.

**One setting or one per network?** The best setting for each network on its own improves its validation F1
by at most 0.006 over the shared setting: ABC 0.863 against 0.869, CBS 0.865 against 0.865, NBC 0.487 against
0.487. So there is one shared setting: *n* = 12, `min_dates=1`, `min_share=0.3`, `min_run_tokens=30`,
`max_gap_tokens=50`.

**Results — provisional.** These come from 2,934 pages present on 2026-10-30: 2,782 ABC, 78 CBS and 74 NBC
(`validation/summary.json`). The test half has 1,408 newscasts with a rundown. The CBS and NBC rows rest on
19 and 25 newscasts and will be redone when the full sample is fetched.

| test | newscasts | precision | recall (words) | F1 | break time captioned | recall, all break time | recall, captioned break time |
|---|---|---|---|---|---|---|---|
| ABC | 1,364 | 0.904 | 0.825 | 0.862 | 98.5% | 0.791 | 0.803 |
| CBS | 19 | 0.900 | 0.733 | 0.808 | 91.0% | 0.607 | 0.668 |
| NBC | 25 | 0.944 | 0.392 | 0.554 | 47.2% | 0.121 | 0.256 |
| **macro F1** | | | | **0.741** | | | |

Words at least 20 s from any change of row kind give precision / recall of 0.927 / 0.898 for ABC,
0.944 / 0.811 for CBS and 0.989 / 0.575 for NBC.

| ABC F1 by year | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.918 | 0.877 | 0.828 | 0.743 | 0.924 | 0.914 | 0.874 | 0.850 | 0.844 | 0.749 |

The remaining CBS and NBC years (CBS 2012 and 2019, NBC 2012 and 2019) are in `summary.json` under
`by_network_year`.

**What limits NBC is missing captions, not missed ads.**
- Half of NBC's break time (52.8%) has no caption block at all. On `KNTV_20120306_013000_NBC_Nightly_News`,
  minutes 15, 19, 20, 24 and 25 are empty. NBC newscasts carry 337 caption words in break rows, against
  1,100 for ABC.
- Even in captioned NBC break minutes, most words are not ads. I read 30 random NBC sentences labelled
  `program` that fell in break rows. Nearly all were news or teases spilling over a boundary ("and when "nbc
  nightly news" continues on this sunday evening, making the grade, ..."), with a median distance of 16 s
  from the boundary. Two were real ads that were missed: a State Farm line and an area-code public service
  announcement.
- Recall over all break time is therefore bounded by the 47% that is captioned. The word-based recall is
  dragged down by boundary spill into minutes that are mostly empty.
- NBC is also thin in the current corpus (74 pages), so NBC-only spots have had few chances to repeat.

**The catalogue on pages it has never seen.** A catalogue was built from validation-day pages only (4,284
spots) and used alone, with no corpus index, on test-day pages:

| test, catalogue only | precision | recall | F1 |
|---|---|---|---|
| ABC | 0.938 | 0.750 | 0.833 |
| CBS | 0.908 | 0.593 | 0.717 |
| NBC | 0.981 | 0.281 | 0.437 |
| macro F1 | | | 0.663 |

Compared with a corpus index, the catalogue alone gives up 7–14 points of recall. It is more precise, because it
holds only lines that mostly occur inside ad runs.

**How much of the error is the reference?** `scripts/sample_disagreements.py` draws 40 random test sentences
of each kind of disagreement (`validation/disagreements.csv`, from the first run on 1,569 pages, column
`judged_ad`). I read each one and judged from the text alone whether it is an ad:

- *Labelled ad, but in a story row:* 26 of 40 are plainly ads (`ask your doctor about cialis for daily use and
  a 30-tablet free trial.`). 3 are programme. 11 cannot be told from the text.
- *Labelled program, but in a commercial row:* 32 of 40 are plainly news or teases, 0 are plainly ads, and 8
  are unclear.

Most measured errors are misalignment in the reference, so the tables understate accuracy. The 2013 ABC dip
points the same way. The share of `reporter:` sentences that land in story rows, a check that does not use
the detector, is 0.96–0.98 in other ABC years but 0.89 in 2013.

**Gold labels.** These are hand-labelled caption windows:

```bash
tv-ads score-gold PAGES --gold gold.csv --out gold.json
```

Each window is flagged twice:
- with the catalogue's auto spots only;
- with the whole catalogue except the human spots taken from that window (leave-one-window-out), so a
  window is never scored against its own labels.

Rows are matched to sentences by their tokens within the same item and minute. Commercial and promo are
positives; unclear is dropped.

On the first 8 windows (325 rows, 74 commercial sentences), the catalogue alone, with no corpus index, gets
precision 0.986 and recall 0.986 both ways: 73 of 74 commercials found, 1 news sentence flagged. The human
spots change nothing yet, because the two annotated breaks are different ads from those in other windows.
The result may be a little flattering: the gold items are in the corpus the auto spots were built
from, although every auto spot also aired on at least one other date.

**LLM codes.** The moment-level codes from notwork_news use
`tv-ads score-codes --sentences ... --sample captions_sample.csv.gz --codes captions_codes.csv.gz --out codes.json`.
Stanford's commercial intervals for cable (tvnews.stanford.edu/export/commercial) could not be used: the
site's TLS certificate has expired, and verification was not bypassed.

## Known failure modes

- **Ads that air once in the corpus.** These include local spots, the first or last airing of a campaign, and
  whole infomercials that replace a pre-empted newscast (`KGO_20190706_230000_...` is 30 minutes of a
  pain-relief device). With nothing to match, they stay `program`. Gap filling recovers a single-airing spot
  only when it is short and sits between two repeating ones.
- **Show furniture and replayed news.** Sign-offs and hand-offs recur and are labelled `boilerplate`, which is
  kept. A news package rerun on a later date, such as a weekend replay or a year-end look back, can be long
  enough to be labelled `ad`. Reporter sign-offs ("linsey davis, abc news, new york.") and `reporter:` lines
  at either edge of an ad run are trimmed back to `program`. On 2,237 test newscasts, the trim moved 407
  sentences out of `ad`, 280 of them in story rows. Macro F1 went from 0.7319 to 0.7325, with recall
  unchanged.
- **Teases are programme.** "Still ahead ..." is written for the night, so it is `program`. The ad-free text is
  free of ads, not of everything except stories.
- **Missing ad captions.** Some stations, notably KNTV for NBC, leave commercial minutes uncaptioned. There is
  nothing to remove.
- **Interleaved streams.** Recordings start with the tail of the previous local show ("i'm cheryl jennings.
  >> from all of us, thanks for watching.") and end with the next one. Local news is unique, so it is
  `program`, not removed. Garbled lines that merge two streams ("climb nothing freezing waters dy") match
  nothing; they are removed only when they sit inside a break.
- **Short sentences** ("geico.") have fewer words than *n*. They are judged by the *n*-grams that run across
  them from the neighbouring sentences, so they follow their neighbours.
- **Corpus density.** The thresholds depend on how many nights are indexed. `min_dates=1` was tuned on a
  corpus that is dense for ABC and sparse for CBS and NBC. Re-tune with `make validate` when the corpus
  changes.

## Development

```bash
make lint      # black (line length 120), isort (black profile), flake8
make test      # pytest; fixtures in tests/fixtures with provenance.json
make validate  # export Vanderbilt rows if needed, tune on validation days, score test days -> data/validate
uv run tv-ads catalogue build ../notwork_news/data/derived/captions/html --out data/spots_auto.csv.gz
uv run tv-ads catalogue add --gold gold.csv --catalogue data/spots_auto.csv.gz --pages ../notwork_news/data/derived/captions/html --out src/tv_ads/data/spots.csv.gz
```

## License

MIT. See `LICENSE`.
