# OARS Colab MVP

Minimal, reproducible MVP for ARLA ablation experiments with full end-to-end tooling.

## Complete feature set
- 4 ablation modes: `baseline`, `hierarchical_only`, `arla_block`, `arla_full`
- raw miniF2F-like JSONL preparation -> trainable feature JSONL
- prepared dataset validation (schema + distribution stats)
- ablation runner with multi-seed support
- plot generation
- statistical summary with CI95
- one-shot end-to-end pipeline runner
- package CLI (`oars-mvp`)
- tests + CI workflow

## Install
```bash
python -m pip install -e .[dev]
```

## One-shot full run
```bash
python scripts/run_full_pipeline.py --epochs 30 --seeds 7 --samples 5000
```

## CLI usage
```bash
oars-mvp prepare --input data/minif2f_raw_sample.jsonl --output data/minif2f_prepared.jsonl --write-split
oars-mvp stats --csv outputs/ablation_summary.csv
```

## Manual workflow
```bash
python scripts/prepare_minif2f.py --input data/minif2f_raw_sample.jsonl --output data/minif2f_prepared.jsonl --write-split
python scripts/validate_dataset.py --input data/minif2f_prepared.jsonl --input-dim 24
python scripts/run_ablation.py --epochs 30 --samples 5000 --seeds 7
python scripts/plot_results.py --csv outputs/ablation_summary.csv
```

## Make targets
```bash
make install
make test
make pipeline
```

## Colab
Use `notebooks/oars_colab_mvp.ipynb`.

## CI
GitHub Actions runs tests and smoke pipeline on push/PR.

## Outputs
- `outputs/ablation_summary.csv`
- `outputs/ablation_stats.json`
- `outputs/plots/*.png`
- `outputs/plots/stats_summary.json`


## M2 Runner (Roadmap)
Run robust M2 validation (k-fold x multi-seed) and get automatic pass/fail:

```bash
python scripts/run_m2.py --raw minif2f_raw.jsonl --kfold 5 --seeds 5 --epochs 12
```

Outputs:
- `outputs/m2/m2_report.json`
- `outputs/m2/m2_runs.json`

M2 criterion:
- pass if best ARLA (`arla_block` or `arla_full`) beats baseline by at least `0.01` in `macro-F1`.

## Phase Decision
python scripts/decide_phase1.py --m2 outputs/phase1/M2_robust/summary.json --m3 outputs/phase1/M3_robust/summary.json

