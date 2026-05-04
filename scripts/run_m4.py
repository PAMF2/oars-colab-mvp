import argparse
import json
from pathlib import Path

from phase1_common import build_rows, load_cfg, mean_ci95, read_jsonl, run_cfg, write_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="minif2f_raw.jsonl")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--output-dir", default="outputs/phase1/M4")
    args = p.parse_args()

    base_cfg = load_cfg()
    raw = read_jsonl(args.raw)
    rows = build_rows(raw, input_dim=base_cfg["input_dim"], num_blocks=base_cfg["num_blocks"])

    # Train on miniF2F, test on PutnamBench families (proxy by benchmark field)
    mini = [r for r in rows if r.get("benchmark") == "minif2f"]
    putnam = [r for r in rows if r.get("benchmark") == "putnambench"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transfer_scores, runs = [], []
    for s in range(args.seeds):
        # small val slice from mini
        val_size = max(1, int(len(mini) * 0.15))
        train = mini[val_size:]
        val = mini[:val_size]
        test = putnam if putnam else mini[:max(1, len(mini)//4)]

        sp = out / f"seed_{s}"
        train_p, val_p, test_p = write_split(sp, train, val, test)

        r = run_cfg(base_cfg, mode="arla_full", train_p=train_p, val_p=val_p, test_p=test_p, seed=base_cfg.get("seed", 42) + s, epochs=args.epochs, out_dir=out / "raw_runs", extra_task={"arla_aux_weight": 1.2})
        transfer_scores.append(r["metrics"]["acc"])
        runs.append(r)

    t_m, t_ci = mean_ci95(transfer_scores)
    summary = {
        "milestone": "M4",
        "primary_metric": "transfer_success (miniF2F -> Putnam family holdout)",
        "diagnostic": "I(c;reward) vs I(c;z) proxy pending",
        "transfer_mean": t_m,
        "transfer_ci95": t_ci,
        "runs": len(runs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
