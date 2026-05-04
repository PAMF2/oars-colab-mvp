.PHONY: install test pipeline ablation plots

install:
	python -m pip install --upgrade pip
	pip install -e .[dev]

test:
	pytest

pipeline:
	python scripts/run_full_pipeline.py

ablation:
	python scripts/run_ablation.py --epochs 3 --samples 1200 --seeds 3

plots:
	python scripts/plot_results.py --csv outputs/ablation_summary.csv
