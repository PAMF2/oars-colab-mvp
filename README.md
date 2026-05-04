# OARS Colab MVP

Minimal, reproducible MVP for testing ARLA-style allocation in a theorem-proving-inspired setup, optimized for Google Colab.

## What this MVP does
- Runs a training loop with 4 ablation modes:
  - `baseline`
  - `hierarchical_only`
  - `arla_block`
  - `arla_full`
- Supports two dataset modes:
  - `synthetic` (default, deterministic and fast)
  - `minif2f_like` (prepared JSONL with `features`, `label`, optional `block_id`)
- Converts raw miniF2F-like JSONL into trainable dense features.
- Validates prepared datasets before training.
- Produces ablation CSV + plots.
- Supports optional W&B tracking.
- Includes tests and CI workflow.

## Quickstart (one command)
```bash
python scripts/run_full_pipeline.py
```

This runs:
1. `prepare_minif2f.py`
2. `validate_dataset.py`
3. `run_ablation.py`
4. `plot_results.py`

## Local setup
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Raw miniF2F preparation
```bash
python scripts/prepare_minif2f.py \
  --input data/minif2f_raw_sample.jsonl \
  --output data/minif2f_prepared.jsonl \
  --input-dim 24 \
  --num-blocks 4 \
  --write-split
```

## Dataset validation
```bash
python scripts/validate_dataset.py --input data/minif2f_prepared.jsonl --input-dim 24
```

## Ablation
```bash
python scripts/run_ablation.py --epochs 3 --samples 1200 --seeds 3
python scripts/plot_results.py --csv outputs/ablation_summary.csv
```

## Use prepared miniF2F-like in training
Set in `configs/default.yaml`:
```yaml
dataset:
  type: minif2f_like
  path: data/minif2f_prepared.jsonl
```

## Enable W&B
Set in `configs/default.yaml`:
```yaml
wandb:
  enabled: true
  project: oars-colab-mvp
```
Then run `wandb login`.

## Tests
```bash
pytest -q
```

## CI
GitHub Actions workflow at `.github/workflows/ci.yml` runs tests and a full smoke pipeline on push/PR.

## Project structure
- `src/oars_mvp/dataset.py`: loaders + miniF2F preparation utilities
- `src/oars_mvp/model.py`: encoder + policy + allocator
- `src/oars_mvp/train.py`: single-run training/eval
- `scripts/prepare_minif2f.py`: raw JSONL -> prepared JSONL
- `scripts/validate_dataset.py`: schema/statistics validator
- `scripts/run_ablation.py`: multi-config, multi-seed runner
- `scripts/plot_results.py`: chart generation
- `scripts/run_full_pipeline.py`: end-to-end runner
- `tests/`: smoke tests
- `notebooks/oars_colab_mvp.ipynb`: Colab notebook

## Notes
This is an MVP scaffold for protocol validation, not a final theorem prover.
