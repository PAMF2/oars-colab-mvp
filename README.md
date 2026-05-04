# OARS Colab MVP

Minimal, reproducible MVP for testing ARLA-style allocation in a theorem-proving-inspired setup, optimized for Google Colab.

## What this MVP does
- Runs a small training loop with 4 ablation modes:
  - `baseline`
  - `hierarchical_only`
  - `arla_block`
  - `arla_full`
- Uses a synthetic proof-task dataset (fast and deterministic) to validate pipeline and metrics.
- Logs key metrics per run:
  - accuracy
  - reward
  - allocator entropy (for ARLA modes)
  - runtime and config metadata
- Supports multi-seed ablation sweeps.

## Quickstart (Colab)
1. Open `notebooks/oars_colab_mvp.ipynb` in Colab.
2. Run all cells.
3. Results are written to `outputs/`.

## Local quickstart
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python scripts/run_ablation.py --epochs 3 --samples 1200 --seeds 3
```

## Project structure
- `src/oars_mvp/dataset.py`: synthetic dataset generator
- `src/oars_mvp/model.py`: encoder + policy + ARLA allocator stubs
- `src/oars_mvp/train.py`: single-run training/eval entrypoint
- `scripts/run_ablation.py`: multi-config, multi-seed runner
- `configs/default.yaml`: baseline config
- `notebooks/oars_colab_mvp.ipynb`: Colab notebook

## Notes
This is an MVP scaffold for protocol validation, not a final theorem prover.
