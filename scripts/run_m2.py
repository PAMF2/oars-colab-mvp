import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oars_mvp.dataset import prepare_minif2f_rows, read_jsonl, write_jsonl  # noqa: E402
from oars_mvp.train import run_experiment  # noqa: E402


MODES = ["baseline", "hierarchical_only", "arla_block", "arla_full"]


def kfold_indices_by_class(class_labels, k: int, seed: int):
    rng = np.random.default_rng(seed)
    by_cls = {}
    for idx, c in enumerate(class_labels):
        by_cls.setdefault(int(c), []).append(idx)

    folds = [[] for _ in range(k)]
    for idxs in by_cls.values():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            folds[i % k].append(int(idx))

    for f in folds:
        rng.shuffle(f)
    return folds


def split_train_val_test_from_folds(n, folds, test_fold_idx, val_ratio, seed):
    test_idx = set(folds[test_fold_idx])
    train_pool = [i for i in range(n) if i not in test_idx]

    rng = np.random.default_rng(seed)
    train_pool = np.array(train_pool)
    rng.shuffle(train_pool)

    n_val = max(1, int(len(train_pool) * val_ratio))
    val_idx = set(train_pool[:n_val].tolist())
    train_idx = set(train_pool[n_val:].tolist())
    return train_idx, val_idx, test_idx


def mean_ci95(vals):
    arr = np.asarray(vals, dtype=float)
    m = float(arr.mean()) if len(arr) else 0.0
    s = float(arr.std(ddof=0)) if len(arr) else 0.0
    ci = 1.96 * s / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return m, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw", default="minif2f_raw.jsonl")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", default="outputs/m2")
    parser.add_argument("--m2-threshold", type=float, default=0.01, help="minimum ARLA gain in macro-F1 over baseline")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    rows_raw = read_jsonl(args.raw)
    rows = prepare_minif2f_rows(rows_raw, input_dim=base_cfg["input_dim"], num_blocks=base_cfg["num_blocks"])
    class_labels = [r.get("class_label", 0) for r in rows]
    folds = kfold_indices_by_class(class_labels, k=args.kfold, seed=base_cfg.get("seed", 42))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    run_id = 0

    for fold_idx in range(args.kfold):
        train_idx, val_idx, test_idx = split_train_val_test_from_folds(
            n=len(rows),
            folds=folds,
            test_fold_idx=fold_idx,
            val_ratio=args.val_ratio,
            seed=base_cfg.get("seed", 42) + fold_idx,
        )

        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_rows = [rows[i] for i in sorted(train_idx)]
        val_rows = [rows[i] for i in sorted(val_idx)]
        test_rows = [rows[i] for i in sorted(test_idx)]

        train_path = fold_dir / "train.jsonl"
        val_path = fold_dir / "val.jsonl"
        test_path = fold_dir / "test.jsonl"
        write_jsonl(str(train_path), train_rows)
        write_jsonl(str(val_path), val_rows)
        write_jsonl(str(test_path), test_rows)

        for s in range(args.seeds):
            for mode in MODES:
                cfg = deepcopy(base_cfg)
                cfg["mode"] = mode
                cfg["epochs"] = args.epochs
                cfg["seed"] = base_cfg.get("seed", 42) + s
                cfg["dataset"]["type"] = "minif2f_like"
                cfg["dataset"]["train_path"] = str(train_path)
                cfg["dataset"]["val_path"] = str(val_path)
                cfg["dataset"]["test_path"] = str(test_path)
                cfg["output_dir"] = str(out_dir / "raw_runs")

                r = run_experiment(cfg)
                r["fold"] = fold_idx
                r["seed_idx"] = s
                results.append(r)
                run_id += 1
                if run_id % 10 == 0:
                    print(f"completed {run_id} runs")

    # Aggregate per mode
    by_mode = {m: [] for m in MODES}
    for r in results:
        by_mode[r["mode"]].append(r["metrics"]["f1_macro"])

    summary = {}
    for m in MODES:
        mean, ci = mean_ci95(by_mode[m])
        summary[m] = {
            "f1_mean": mean,
            "f1_ci95": ci,
            "n": len(by_mode[m]),
        }

    baseline = summary["baseline"]["f1_mean"]
    arla_block_gain = summary["arla_block"]["f1_mean"] - baseline
    arla_full_gain = summary["arla_full"]["f1_mean"] - baseline
    best_arla_gain = max(arla_block_gain, arla_full_gain)

    # M2 pass criterion
    m2_pass = best_arla_gain >= args.m2_threshold

    report = {
        "config": {
            "kfold": args.kfold,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "val_ratio": args.val_ratio,
            "m2_threshold": args.m2_threshold,
            "raw": args.raw,
        },
        "summary": summary,
        "gains_vs_baseline": {
            "arla_block": arla_block_gain,
            "arla_full": arla_full_gain,
            "best_arla_gain": best_arla_gain,
        },
        "m2_pass": m2_pass,
        "recommendation": "M2 complete" if m2_pass else "M2 not complete: improve concept allocation/labels/features",
        "runs": len(results),
    }

    report_path = out_dir / "m2_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    raw_path = out_dir / "m2_runs.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
