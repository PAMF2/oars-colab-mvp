import hashlib
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


def assign_benchmark(row):
    if "benchmark" in row:
        b = str(row["benchmark"]).lower()
        if "putnam" in b:
            return "putnambench"
        if "mini" in b:
            return "minif2f"
    # deterministic fallback split
    key = json.dumps(row, sort_keys=True).encode("utf-8", errors="ignore")
    h = int(hashlib.md5(key).hexdigest()[:8], 16)
    return "putnambench" if (h % 5 == 0) else "minif2f"


def build_rows(raw_rows, input_dim, num_blocks):
    rows = prepare_minif2f_rows(raw_rows, input_dim=input_dim, num_blocks=num_blocks)
    for i, r in enumerate(rows):
        r["benchmark"] = assign_benchmark(raw_rows[i])
        r["problem_family"] = int(r.get("class_label", 0))
    return rows


def split_train_test(rows, seed=42, train_ratio=0.8):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    cut = int(len(rows) * train_ratio)
    train = [rows[i] for i in idx[:cut]]
    test = [rows[i] for i in idx[cut:]]
    return train, test


def split_train_val_test(rows, seed=42, train_ratio=0.7, val_ratio=0.15):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n = len(rows)
    c1 = int(n * train_ratio)
    c2 = int(n * (train_ratio + val_ratio))
    train = [rows[i] for i in idx[:c1]]
    val = [rows[i] for i in idx[c1:c2]]
    test = [rows[i] for i in idx[c2:]]
    if len(val) == 0 and len(train) > 1:
        val.append(train.pop())
    if len(test) == 0 and len(train) > 1:
        test.append(train.pop())
    return train, val, test


def write_split(base_dir: Path, train, val, test):
    base_dir.mkdir(parents=True, exist_ok=True)
    train_p = base_dir / "train.jsonl"
    val_p = base_dir / "val.jsonl"
    test_p = base_dir / "test.jsonl"
    write_jsonl(str(train_p), train)
    write_jsonl(str(val_p), val)
    write_jsonl(str(test_p), test)
    return train_p, val_p, test_p


def run_cfg(base_cfg, mode, train_p, val_p, test_p, seed, epochs, out_dir, extra_task=None):
    cfg = deepcopy(base_cfg)
    cfg["mode"] = mode
    cfg["seed"] = seed
    cfg["epochs"] = epochs
    cfg["dataset"]["type"] = "minif2f_like"
    cfg["dataset"]["train_path"] = str(train_p)
    cfg["dataset"]["val_path"] = str(val_p)
    cfg["dataset"]["test_path"] = str(test_p)
    cfg["output_dir"] = str(out_dir)
    cfg.setdefault("task", {})
    if extra_task:
        cfg["task"].update(extra_task)
    return run_experiment(cfg)


def mean_ci95(vals):
    arr = np.asarray(vals, dtype=float)
    m = float(arr.mean()) if len(arr) else 0.0
    s = float(arr.std(ddof=0)) if len(arr) else 0.0
    ci = 1.96 * s / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return m, ci


def load_cfg(path="configs/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
