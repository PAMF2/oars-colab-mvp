import argparse
import json
from pathlib import Path

from phase1_common import build_rows, load_cfg, mean_ci95, read_jsonl, run_cfg, split_train_val_test, write_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="minif2f_raw.jsonl")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--output-dir", default="outputs/phase1/M3")
    args = p.parse_args()

    base_cfg = load_cfg()
    raw = read_jsonl(args.raw)
    rows = build_rows(raw, input_dim=base_cfg["input_dim"], num_blocks=base_cfg["num_blocks"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    delta_pass, concept_counts, runs = [], [], []
    for s in range(args.seeds):
        train, val, test = split_train_val_test(rows, seed=base_cfg.get("seed", 42) + s)
        sp = out / f"seed_{s}"
        train_p, val_p, test_p = write_split(sp, train, val, test)

        # hardcoded concept regime (proxy): lower aux
        rh = run_cfg(base_cfg, mode="arla_block", train_p=train_p, val_p=val_p, test_p=test_p, seed=base_cfg.get("seed", 42) + s, epochs=args.epochs, out_dir=out / "raw_runs", extra_task={"arla_aux_weight": 0.8})
        # discovered concept regime (proxy): stronger aux + full allocator
        rd = run_cfg(base_cfg, mode="arla_full", train_p=train_p, val_p=val_p, test_p=test_p, seed=base_cfg.get("seed", 42) + s, epochs=args.epochs, out_dir=out / "raw_runs", extra_task={"arla_aux_weight": 1.2})

        d = rd["metrics"]["acc"] - rh["metrics"]["acc"]
        delta_pass.append(d)
        # proxy concept count from entropy scale
        concept_counts.append(int(10 + rd["metrics"].get("allocator_entropy", 0.0) * 20))
        runs.append({"seed": s, "hardcoded": rh, "discovered": rd, "delta_pass": d})

    d_m, d_ci = mean_ci95(delta_pass)
    c_m, c_ci = mean_ci95(concept_counts)
    summary = {
        "milestone": "M3",
        "primary_metric": "delta_pass_rate (discovered - hardcoded)",
        "diagnostic": "concept_count_proxy",
        "delta_pass_mean": d_m,
        "delta_pass_ci95": d_ci,
        "concept_count_mean": c_m,
        "concept_count_ci95": c_ci,
        "runs": len(runs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
