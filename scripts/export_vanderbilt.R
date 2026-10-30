# Writes the Vanderbilt index rows tv-ads validate reads: regular ABC, CBS and NBC evening newscasts, 2010-2021.
#
# Usage: Rscript --no-init-file scripts/export_vanderbilt.R <archive_all.rds> <out.csv.gz>
#
# archive_all.rds is written by notwork_news (R/01_extract_archive.R) from the Vanderbilt Television News
# Archive export (https://doi.org/10.7910/DVN/BP2JXU, access-restricted) and its 2020-2025 extension.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: scripts/export_vanderbilt.R <archive_all.rds> <out.csv.gz>")
archive <- readRDS(args[[1]])
keep <- archive$year >= 2010 & archive$year <= 2021 & archive$channel %in% c("ABC", "CBS", "NBC") & !archive$special
rows <- archive[keep, c("channel", "date", "broadcast_time", "program_title", "broadcast_title", "advertising", "row")]
write.csv(rows, gzfile(args[[2]]), row.names = FALSE)
message(nrow(rows), " rows -> ", args[[2]])
