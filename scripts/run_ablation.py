import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oars_mvp.stats import summarize_csv  # noqa: E402
from oars_mvp.train import run_experiment  # noqa: E402


MODES = ["baseline", "hierarchical_only", "arla_block", "arla_full"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/ablation_summary.csv")
    parser.add_argument("--stats-json", type=str, default="outputs/ablation_stats.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    runs = []
    for i in range(args.seeds):
        for mode in MODES:
            cfg = deepcopy(base_cfg)
            cfg["seed"] = base_cfg.get("seed", 42) + i
            cfg["mode"] = mode
            if args.epochs is not None:
                cfg["epochs"] = args.epochs
            if args.samples is not None:
                cfg["samples"] = args.samples
            result = run_experiment(cfg)
            runs.append(result)

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "seed", "acc", "reward", "allocator_entropy", "runtime_sec"])
        for r in runs:
            m = r["metrics"]
            writer.writerow([r["mode"], r["seed"], m["acc"], m["reward"], m["allocator_entropy"], r["runtime_sec"]])

    stats = summarize_csv(args.output)
    with open(args.stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps({"runs": len(runs), "summary_csv": args.output, "stats_json": args.stats_json}, indent=2))


if __name__ == "__main__":
    main()
