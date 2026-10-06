PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
WORKERS ?= 4
export PYTHONPATH := src

.PHONY: test data audit previews lint format check build public-check clean

test:
	$(PY) -m pytest -q

data:
	$(PY) scripts/generate_data.py --config configs/dataset/tetris2d.json --out data/tetris2d --workers $(WORKERS)
	$(PY) scripts/generate_data.py --config configs/dataset/tetris3d.json --out data/tetris3d --workers $(WORKERS)

audit:
	$(PY) scripts/dataset_stats.py --config configs/dataset/tetris2d.json --data data/tetris2d --out docs/dataset_stats_tetris2d.md
	$(PY) scripts/dataset_stats.py --config configs/dataset/tetris3d.json --data data/tetris3d --out docs/dataset_stats_tetris3d.md

previews:
	$(PY) scripts/preview_samples.py --config configs/dataset/tetris2d.json --data data/tetris2d --split id_test --count 8 --out previews/tetris2d
	$(PY) scripts/preview_samples.py --config configs/dataset/tetris3d.json --data data/tetris3d --split sc_id_test --count 8 --out previews/tetris3d

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

format:
	$(PY) -m ruff format src tests scripts

check: lint test

build:
	$(PY) -m build

public-check:
	$(PY) scripts/check_publication.py

clean:
	rm -rf data previews .pytest_cache
