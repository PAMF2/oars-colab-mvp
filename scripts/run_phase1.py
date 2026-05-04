import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oars_mvp.dataset import prepare_minif2f_rows, read_jsonl, write_jsonl  # noqa: E402
from oars_mvp.train import run_experiment  # noqa: E402


MILESTONES = {
    "M1": {
        "label": "No allocator (Pillar I + III)",
        "mode": "hierarchical_only",
        "task_overrides": {"arla_aux_weight": 0.0},
    },
    "M2": {
        "label": "ARLA regime A (hardcoded concepts)",
        "mode": "arla_block",
        "task_overrides": {"arla_aux_weight": 0.8},
    },
    "M3": {
        "label": "ARLA regime B (concept emergence proxy)",
        "mode": "arla_full",
        "task_overrides": {"arla_aux_weight": 1.0},
    },
    "M4": {
        "label": "ARLA regime C (IB proxy)",
        "mode": "arla_full",
        "task_overrides": {"arla_aux_weight": 1.2},
    },
}


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


def split_train_val_test(n, folds, test_fold_idx, val_ratio, seed):
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


def run_milestone(milestone, base_cfg, rows, folds, args, out_root):
    spec = MILESTONES[milestone]
    mode = spec["mode"]
    results = []

    ms_dir = out_root / milestone
    ms_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(args.kfold):
        train_idx, val_idx, test_idx = split_train_val_test(
            n=len(rows), folds=folds, test_fold_idx=fold_idx, val_ratio=args.val_ratio, seed=base_cfg.get("seed", 42) + fold_idx
        )

        fold_dir = ms_dir / f"fold_{fold_idx}"
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
            cfg = deepcopy(base_cfg)
            cfg["mode"] = mode
            cfg["epochs"] = args.epochs
            cfg["seed"] = base_cfg.get("seed", 42) + s
            cfg["dataset"]["type"] = "minif2f_like"
            cfg["dataset"]["train_path"] = str(train_path)
            cfg["dataset"]["val_path"] = str(val_path)
            cfg["dataset"]["test_path"] = str(test_path)
            cfg["output_dir"] = str(ms_dir / "raw_runs")

            cfg.setdefault("task", {})
            for k, v in spec["task_overrides"].items():
                cfg["task"][k] = v

            r = run_experiment(cfg)
            r["fold"] = fold_idx
            r["seed_idx"] = s
            r["milestone"] = milestone
            results.append(r)

    f1_vals = [r["metrics"]["f1_macro"] for r in results]
    acc_vals = [r["metrics"]["acc"] for r in results]
    f1_mean, f1_ci = mean_ci95(f1_vals)
    acc_mean, acc_ci = mean_ci95(acc_vals)

    summary = {
        "milestone": milestone,
        "label": spec["label"],
        "mode": mode,
        "runs": len(results),
        "f1_mean": f1_mean,
        "f1_ci95": f1_ci,
        "acc_mean": acc_mean,
        "acc_ci95": acc_ci,
    }

    with (ms_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (ms_dir / "runs.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw", default="minif2f_raw.jsonl")
    parser.add_argument("--milestones", default="M1,M2,M3,M4")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", default="outputs/phase1")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    rows_raw = read_jsonl(args.raw)
    rows = prepare_minif2f_rows(rows_raw, input_dim=base_cfg["input_dim"], num_blocks=base_cfg["num_blocks"])
    class_labels = [r.get("class_label", 0) for r in rows]
    folds = kfold_indices_by_class(class_labels, k=args.kfold, seed=base_cfg.get("seed", 42))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    mlist = [m.strip() for m in args.milestones.split(",") if m.strip()]
    summaries = []
    for m in mlist:
        if m not in MILESTONES:
            raise ValueError(f"Unknown milestone: {m}")
        summaries.append(run_milestone(m, base_cfg, rows, folds, args, out_root))

    # compare against M1 if present
    report = {"summaries": summaries}
    by = {s["milestone"]: s for s in summaries}
    if "M1" in by:
        base = by["M1"]["f1_mean"]
        gains = {}
        for m, s in by.items():
            gains[m] = s["f1_mean"] - base
        report["gains_vs_m1"] = gains

    with (out_root / "phase1_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"saved: {out_root / 'phase1_report.json'}")


if __name__ == "__main__":
    main()
