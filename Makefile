PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest

.PHONY: refresh build test all clean

refresh:
	$(PYTHON) pipeline/refresh_live_sources.py

build:
	$(PYTHON) pipeline/build_duckdb.py

test:
	$(PYTEST) -v

all: refresh build test

clean:
	rm -f dist/*.duckdb dist/*.duckdb.wal
