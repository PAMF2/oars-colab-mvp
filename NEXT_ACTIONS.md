# Next Actions (Phase I Recovery Plan)

Current status from robust runs:
- M2: FAIL (delta not statistically positive)
- M3: FAIL (delta not statistically positive)

Objective:
- Achieve statistically positive ARLA lift over baseline on M2 and M3.

## Week 1: Data and Signal Integrity
1. Expand benchmark coverage:
- Increase effective dataset size (more miniF2F/Putnam rows) to reduce CI width.
- Ensure benchmark/family labels are explicit in source rows (avoid fallback hashing whenever possible).

2. Strengthen target fidelity:
- Replace weak proxy labels with tactic-family labels derived from proof traces where available.
- Add leakage checks between train/val/test family splits.

3. Diagnostics to add:
- Per-family confusion matrix.
- Entropy-by-family for allocator (detect collapse).

## Week 2: ARLA Efficacy Iteration
1. M2 tuning (hardcoded bank):
- Sweep `arla_aux_weight` in {0.6, 0.8, 1.0, 1.2}.
- Sweep gating strength for `arla_block`.
- Keep baseline frozen for fair delta comparison.

2. M3 tuning (emergent regime):
- Introduce concept-count constraints: min=8, max=64.
- Penalize concept explosion with regularizer.
- Track stability of concept assignments across seeds.

3. Acceptance criteria:
- M2_GO: `delta_pass_mean - delta_pass_ci95 > 0`
- M3_GO: `delta_pass_mean - delta_pass_ci95 > 0`

## Execution commands
Run robust M2:
```bash
python scripts/run_m2.py --raw minif2f_raw.jsonl --seeds 30 --epochs 20 --output-dir outputs/phase1/M2_robust
```

Run robust M3:
```bash
python scripts/run_m3.py --raw minif2f_raw.jsonl --seeds 30 --epochs 20 --output-dir outputs/phase1/M3_robust
```

Decide phase status:
```bash
python scripts/decide_phase1.py --m2 outputs/phase1/M2_robust/summary.json --m3 outputs/phase1/M3_robust/summary.json
```

## Stop conditions
- If two consecutive redesign cycles fail to produce positive M2 or M3, freeze architecture changes and revisit label/spec definition.
