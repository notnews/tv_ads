# tv_ads: find commercials in TV news closed captions

The Internet Archive's TV News Archive keeps closed captions for decades of newscasts. Caption text has no
marks for commercials. A paid spot for Nexium runs straight on from a reporter's sign-off:

> ... clayton sandell, abc news, denver. ... and still ahead, the first blast of winter ... [ male announcer ]
> don't have the hops for hoops with your buddies? ... learn more at isitlowt.com.

`tv_ads` labels each caption sentence as `ad`, `boilerplate` or `program` and writes the text with the ads
taken out. It needs no hand labels. It works on any channel, including the broadcast networks (ABC, CBS, NBC),
where the only earlier source of commercial times, the Stanford Cable TV News Analyzer, has no coverage.
(Stanford's intervals cover CNN, Fox News and MSNBC and are found from the video.)

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
2. **Index.** Every run of *n* consecutive words (an *n*-gram; *n* = 8 by default) in every item is hashed.
   For each distinct *n*-gram the index records how many distinct air dates and networks it appeared on.
   Repeats on the same date count once, so a story that two stations carry on the same night is not evidence.
3. **Score.** For a sentence, each word gets the largest "other dates" count among the *n*-grams that cover
   it, leaving out the item's own airing. A sentence *repeats* when at least `min_share` of its words have a
   count of at least `min_dates`.
4. **Label.** Consecutive repeating sentences form runs. Stretches of at most `max_gap_tokens` words between
   two repeating sentences are filled in, so one garbled caption line does not split an ad break. A run of at
   least `min_run_tokens` words is `ad`. A 30-second spot is roughly 60–80 words, and breaks string several
   spots together. A shorter run is `boilerplate`: sign-offs ("steve osunsami, abc news, atlanta."), hand-offs
   and sound bites replayed on another night. Boilerplate is kept in the ad-free text, because it is part of
   the programme.

The `ad` label covers network promos as well as commercials: both are pre-produced and air repeatedly. The
tool does not try to tell them apart.

**Exact n-grams rather than MinHash.** MinHash with locality-sensitive hashing (LSH) is the usual way to find
near-duplicate text. I checked [datasketch](https://github.com/ekzhu/datasketch), the maintained library for
it: release 2.0.0 came out on 2026-07-05, the last push was on 2026-10-26, and the repository is not archived.
`scripts/compare_minhash.py` runs it at the sentence level (64 permutations, word 3-gram shingles, Jaccard
thresholds 0.5 and 0.7) through the same tuning and test split. On 1,197 pages the two came out level on test
F1: 0.841 for 8-grams against 0.842 and 0.840 for MinHash (`validation/compare_minhash.txt`; that run scored
only rows titled "Commercial", before `break` rows were counted). I kept exact *n*-grams for three reasons:
they are simpler, they need no LSH settings, and the index for the full 11,520-item sample (roughly 40 million
hashes, extrapolating from 6.2 million for 1,724 pages) fits in memory as sorted numpy arrays, while per-sentence MinHash signatures would need several GB.
Ad captions are pre-produced, so they repeat nearly verbatim, and a single mis-captioned word breaks only the
*n*-grams that contain it.

## Install and use

```bash
uv sync
# 1. build the repetition index from a directory of pages (html or html.gz, with meta/<id>_meta.json beside html/)
uv run tv-ads index ../notwork_news/data/derived/captions/html --out data/index_n8.npz
# 2. label sentences and write ad-free text
uv run tv-ads flag ../notwork_news/data/derived/captions/html --index data/index_n8.npz --out data/flagged
```

`flag` writes `sentences.csv.gz` with these columns: `identifier`, `minute`, `sentence` (position in the item),
`start_second` and `end_second` (seconds from the recording start, interpolated within the minute), `label`,
`repeat_share` and `text`. It also writes `ad_free.jsonl`, with one `{identifier, aired, text}` record per
item. The thresholds can be overridden with `--min-dates`, `--min-share`, `--min-run-tokens` and
`--max-gap-tokens`.

The counts depend on how many nights the index holds, so **build the index from the whole corpus you are
cleaning, and re-tune when the corpus changes a lot** (`make validate`). The defaults (`min_dates=1`,
`min_share=0.3`, `min_run_tokens=80`, `max_gap_tokens=25`) were tuned on 1,569 pages, mostly ABC 2010–2013.
With a denser corpus a higher `min_dates` will probably win, because common news phrases will start to recur
too.

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
address in commercial rows. The estimates were −5 s (ABC), −15 s (CBS) and +5 s (NBC). The earlier pilot
estimates of +10 to +30 s did not reproduce here.

**Resolution.** Sentence times are interpolated within a minute. The two sources also record different feeds:
Vanderbilt records Nashville, and most sampled captions come from the San Francisco stations' West Coast
airing. Expect the reference to be off by 10–30 s at every boundary, and occasionally by a minute or more.

**Split.** Air dates are split in half by a hash of the date. Every newscast on a date falls on the same side.
The offsets, the *n*-gram size (5, 8 or 12) and the thresholds (a 504-cell grid) were chosen to maximise F1 on
validation days. The numbers below are for the other half, the test days. The index itself uses no labels, so
it is built from all pages.

**Metric.** The positive class is time in `commercial` or `break` rows; the negative class is time in story
rows. Sentences are weighted by word count, so the scores are shares of caption words. Sentences in furniture
rows, or outside the first and last rows, are reported separately rather than scored.

**Results** (1,569 pages present on 2026-10-30, `validation/summary.json`). The test half has 758 newscasts
with a rundown and 309,857 sentences. Selection on validation days chose *n* = 8 (validation F1: 0.852 for
*n* = 5, 0.852 for 8, 0.851 for 12).

| test words | precision | recall | F1 |
|---|---|---|---|
| all scored sentences (2.97 M words; 25.7% break time) | 0.894 | 0.808 | 0.849 |
| sentences ≥ 20 s from any change of row kind (2.50 M words) | 0.923 | 0.866 | 0.894 |

| by network | words | precision | recall | F1 |
|---|---|---|---|---|
| ABC | 2,835,374 | 0.893 | 0.813 | 0.851 |
| CBS | 63,282 | 0.922 | 0.759 | 0.832 |
| NBC | 75,308 | 0.971 | 0.342 | 0.506 |

| by year | 2010 | 2011 | 2012 | 2013 | 2014 | 2019 |
|---|---|---|---|---|---|---|
| F1 | 0.901 | 0.882 | 0.844 | 0.761 | 0.928 | 0.707 |

Where the labels went, in test words:

| reference row | ad | boilerplate | program |
|---|---|---|---|
| story | 3.3% | 2.8% | 93.9% |
| commercial | 81.5% | 0.7% | 17.8% |
| furniture (teases, intro, good night) | 15.0% | 5.4% | 79.6% |

**How much of the error is the reference?** `scripts/sample_disagreements.py` draws 40 random test sentences
of each kind of disagreement (`validation/disagreements.csv`, column `judged_ad`). I read each one and judged
from the text alone whether it is an ad:

- *Labelled ad, but in a story row:* 26 of 40 are plainly ads (`ask your doctor about cialis for daily use and
  a 30-tablet free trial.`, `visit progressive.com today.`). 3 are programme. 11 cannot be told from the text
  (`you make a great team.`).
- *Labelled program, but in a commercial row:* 32 of 40 are plainly news or teases (`reporter: today's target,
  two stores openly selling stolen clothes and perfume.`). 0 are plainly ads. 8 are unclear.

Most measured errors are therefore misalignment in the reference, not mistakes by the detector, so the table
understates accuracy. The 2013 dip points the same way. The detector-independent check (what share of
`reporter:` sentences land in story rows) is 0.96–0.98 for ABC in 2010–2012 and 2014, but 0.89 in 2013. The
reference for those items is off, not just the detector. These judgments are mine, from text alone. The CSV
is there to be re-read.

**Not yet covered.** The pages present when this was run were mostly ABC 2010–2013. CBS and NBC test
estimates rest on 17 and 25 newscasts. Rerun `make validate` when the full sample is in. Stanford's commercial
intervals for cable (tvnews.stanford.edu/export/commercial) could not be used as a second benchmark: the
site's TLS certificate has expired, and verification was not bypassed. The moment-level LLM codes that
notwork_news is producing (`captions_codes.csv.gz`) can be scored with
`tv-ads score-codes --sentences data/flagged/sentences.csv.gz --sample captions_sample.csv.gz --codes captions_codes.csv.gz --out codes.json`.
That command finds the sentence whose interpolated span contains each coded moment.

## Examples

From `KGO_20120103_013000_ABC_World_News_With_Diane_Sawyer`, with the index built from all 1,724 pages
(`repeat_share` in brackets):

| minute | label | sentence | why |
|---|---|---|---|
| 12 | program | reporter: these harrowing meth ads helped drop its use in montana by 50%. | news about ads; no 8-gram recurs (0.0) |
| 13 | boilerplate | steve osunsami, abc news, atlanta. | the reporter's sign-off recurs on other nights, but the run is too short to be an ad (1.0) |
| 13 | program | and still ahead tonight, we have new details on an amazing rescue, ten brave strangers ... | tonight's tease is unique (0.08) |
| 13 | program | sometimes investing opportunities are hard to spot. | **missed**: the first line of the Fidelity spot does not recur in this corpus (0.0) |
| 13 | ad | the etf market tracker is one more innovative reason serious investors are choosing fidelity. | (1.0) |
| 13 | ad | climb nothing freezing waters dy | garbled line inside the break, filled in as a gap (0.0) |
| 14 | ad | when bp made a commitment to the gulf, we knew it would take time, but we were determined to see it through. | BP spot (1.0) |
| 15 | ad | possible side effects include headache, diarrhea and abdominal pain. | Nexium spot (1.0) |
| 16 | program | reporter: it was a new year's eve near-tragedy. | news resumes (0.0) |
| 18 | ad | [ male announcer ] don't have the hops for hoops with your buddies? | low-T spot (1.0) |

## Known failure modes

- **Ads that air once in the corpus.** These include local spots, the first or last airing of a campaign, and
  whole infomercials that replace a pre-empted newscast (`KGO_20190706_230000_...` is 30 minutes of a
  pain-relief device). With nothing to match, they stay `program`. Gap filling recovers a single-airing spot
  only when it is short and sits between two repeating ones.
- **Show furniture and replayed news.** Sign-offs and hand-offs recur and are labelled `boilerplate`, which is
  kept. A news package rerun on a later date, such as a weekend replay or a year-end look back, is long enough
  to be labelled `ad`. A replayed sound bite is labelled `boilerplate`.
- **Teases are programme.** "Still ahead ..." is written for the night, so it is `program`. The ad-free text is
  free of ads, not of everything except stories.
- **Missing ad captions.** Some stations' pages have no captions for commercial minutes. On
  `KNTV_20120306_013000_NBC_Nightly_News`, minutes 15, 19, 20, 24 and 25 are empty. There is nothing to
  remove, and the low NBC recall above is largely reference time with no caption text in it (break time is
  9.8% of NBC words against 26% for ABC).
- **Interleaved streams.** Recordings start with the tail of the previous local show ("i'm cheryl jennings.
  >> from all of us, thanks for watching.") and end with the next one. Local news is unique, so it is
  `program`, not removed.
- **Short sentences** ("geico.") have fewer words than *n*. They are judged by the *n*-grams that run across
  them from the neighbouring sentences, so they follow their neighbours.
- **Corpus density.** The thresholds depend on how many nights are indexed. Tuned on a sparse corpus,
  `min_dates=1` will be too loose on the full sample.

## Development

```bash
make lint    # black (line length 120), isort (black profile), flake8
make test    # pytest; fixtures in tests/fixtures with provenance.json
make validate  # export Vanderbilt rows if needed, tune on validation days, score test days -> data/validate
```

## License

MIT. See `LICENSE`.
