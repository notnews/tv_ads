PAGES ?= ../notwork_news/data/derived/captions/html
VANDERBILT ?= data/vanderbilt/segments_2010_2021.csv.gz

.PHONY: lint format test ci-docker vanderbilt validate

lint:
	uv run black --check .
	uv run isort --check-only .
	uv run flake8 src tests scripts

format:
	uv run isort .
	uv run black .

test:
	uv run pytest -q

ci-docker:
	@for version in 3.12 3.13; do \
	  COPYFILE_DISABLE=1 tar --exclude=._* --exclude=__pycache__ --exclude=.git --exclude=.venv --exclude=data -cf - . | \
	  docker run --rm -i python:$$version-slim sh -ec 'mkdir /work; tar -xf - -C /work; cd /work; pip install -q uv; \
	    uv sync --frozen --group dev; uv run black --check .; uv run isort --check-only .; uv run flake8 src tests scripts; \
	    uv run pytest -q' || exit $$?; \
	done

vanderbilt: $(VANDERBILT)

$(VANDERBILT):
	mkdir -p $(dir $@)
	Rscript --no-init-file scripts/export_vanderbilt.R ../notwork_news/data/derived/archive_all.rds $@

validate: $(VANDERBILT)
	uv run tv-ads validate $(PAGES) --vanderbilt $(VANDERBILT) --out data/validate
