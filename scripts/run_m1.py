import argparse
import json
from pathlib import Path

from phase1_common import build_rows, load_cfg, mean_ci95, read_jsonl, run_cfg, split_train_val_test, write_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="minif2f_raw.jsonl")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--output-dir", default="outputs/phase1/M1")
    args = p.parse_args()

    base_cfg = load_cfg()
    raw = read_jsonl(args.raw)
    rows = build_rows(raw, input_dim=base_cfg["input_dim"], num_blocks=base_cfg["num_blocks"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    f1s, accs, runs = [], [], []

    for s in range(args.seeds):
        train, val, test = split_train_val_test(rows, seed=base_cfg.get("seed", 42) + s)
        sp = out / f"seed_{s}"
        train_p, val_p, test_p = write_split(sp, train, val, test)
        r = run_cfg(base_cfg, mode="hierarchical_only", train_p=train_p, val_p=val_p, test_p=test_p, seed=base_cfg.get("seed", 42) + s, epochs=args.epochs, out_dir=out / "raw_runs", extra_task={"arla_aux_weight": 0.0})
        f1s.append(r["metrics"]["f1_macro"])
        accs.append(r["metrics"]["acc"])
        runs.append(r)

    f1_m, f1_ci = mean_ci95(f1s)
    acc_m, acc_ci = mean_ci95(accs)
    summary = {
        "milestone": "M1",
        "primary_metric": "miniF2F/Putnam pass rate proxy (test acc)",
        "diagnostic": "f1_macro",
        "pass_rate_mean": acc_m,
        "pass_rate_ci95": acc_ci,
        "f1_mean": f1_m,
        "f1_ci95": f1_ci,
        "runs": len(runs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
