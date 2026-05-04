# OARS Colab MVP

Minimal, reproducible MVP for testing ARLA-style allocation in a theorem-proving-inspired setup, optimized for Google Colab.

## What this MVP does
- Runs a small training loop with 4 ablation modes:
  - `baseline`
  - `hierarchical_only`
  - `arla_block`
  - `arla_full`
- Supports two dataset modes:
  - `synthetic` (default, deterministic and fast)
  - `minif2f_like` (prepared JSONL with `features`, `label`, optional `block_id`)
- Includes a miniF2F raw-data preparer:
  - Converts raw JSONL theorem rows into trainable dense features.
- Logs key metrics per run:
  - accuracy
  - reward
  - allocator entropy
  - runtime and config metadata
- Supports optional Weights & Biases tracking.
- Generates automatic comparison plots.

## Quickstart (Colab)
1. Open `notebooks/oars_colab_mvp.ipynb` in Colab.
2. Run all cells.
3. Results are written to `outputs/` and charts to `outputs/plots/`.

## Local quickstart
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python scripts/run_ablation.py --epochs 3 --samples 1200 --seeds 3
python scripts/plot_results.py --csv outputs/ablation_summary.csv
```

## Prepare raw miniF2F-like JSONL
Expected input is JSONL where each row has at least one text field among:
`formal_statement`, `statement`, `informal_statement`, `goal`, `theorem`, `prompt`, `text`.
Optional fields: `proof`/`formal_proof`/`completion` and `label`.

```bash
python scripts/prepare_minif2f.py \
  --input data/minif2f_raw.jsonl \
  --output data/minif2f_prepared.jsonl \
  --input-dim 24 \
  --num-blocks 4 \
  --write-split
```

Then set:
```yaml
dataset:
  type: minif2f_like
  path: data/minif2f_prepared.jsonl
```

And run training.

## Enable W&B
Set in `configs/default.yaml`:
```yaml
wandb:
  enabled: true
  project: oars-colab-mvp
```

Then run `wandb login` in Colab/local before training.

## Project structure
- `src/oars_mvp/dataset.py`: synthetic loader + miniF2F preparation utilities
- `src/oars_mvp/model.py`: encoder + policy + allocator
- `src/oars_mvp/train.py`: single-run training/eval
- `scripts/prepare_minif2f.py`: raw JSONL -> prepared JSONL converter
- `scripts/run_ablation.py`: multi-config, multi-seed runner
- `scripts/plot_results.py`: chart generation
- `configs/default.yaml`: unified config
- `notebooks/oars_colab_mvp.ipynb`: Colab notebook

## Notes
This is an MVP scaffold for protocol validation, not a final theorem prover.
