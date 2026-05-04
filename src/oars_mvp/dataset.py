import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticProofDataset(Dataset):
    def __init__(self, n_samples: int, input_dim: int, num_blocks: int, seed: int = 42):
        self.n_samples = n_samples
        rng = np.random.default_rng(seed)
        self.x = rng.normal(0, 1, size=(n_samples, input_dim)).astype(np.float32)
        self.block_id = rng.integers(low=0, high=num_blocks, size=n_samples, endpoint=False)
        weights = rng.normal(0, 0.7, size=(num_blocks, input_dim)).astype(np.float32)
        block_bias = rng.normal(0, 0.25, size=(num_blocks,)).astype(np.float32)
        logits = np.array([
            np.dot(self.x[i], weights[self.block_id[i]]) + block_bias[self.block_id[i]]
            for i in range(n_samples)
        ])
        probs = 1 / (1 + np.exp(-logits))
        self.y = (probs > 0.5).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "block_id": torch.tensor(self.block_id[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class MiniF2FLikeDataset(Dataset):
    def __init__(self, path: str, input_dim: int, num_blocks: int):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        rows = []
        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError("Dataset file is empty")

        x, y, b = [], [], []
        for i, row in enumerate(rows):
            feats = row.get("features")
            if not isinstance(feats, list) or len(feats) != input_dim:
                raise ValueError(f"Row {i} has invalid features; expected list length {input_dim}")
            label = float(row.get("label", 0.0))
            block = int(row.get("block_id", i % max(num_blocks, 1)))
            block = max(0, min(block, num_blocks - 1))
            x.append(feats)
            y.append(label)
            b.append(block)

        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.block_id = np.asarray(b, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "block_id": torch.tensor(self.block_id[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def _text_from_row(row: Dict) -> str:
    keys = ["formal_statement", "statement", "informal_statement", "informal_prefix", "goal", "theorem", "prompt", "text"]
    return "\n".join([str(row.get(k, "")) for k in keys if row.get(k)])


def _proof_from_row(row: Dict) -> str:
    keys = ["formal_proof", "proof", "lean_proof", "completion", "solution"]
    return "\n".join([str(row.get(k, "")) for k in keys if row.get(k)])


def _bucket_from_text(text: str, num_blocks: int) -> int:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16) % max(num_blocks, 1)


def _dense_features(text: str, proof: str, input_dim: int) -> List[float]:
    full = (text + "\n" + proof).strip()
    toks = re.findall(r"[A-Za-z0-9_']+|[=+\-*/^(){}\[\]<>]|∀|∃|→|↔|≤|≥", full)
    tok_count = max(len(toks), 1)
    uniq = len(set(toks))
    chars = len(full)

    keyword_set = {
        "induction": ["induction", "nat.rec", "cases", "simp"],
        "algebra": ["ring", "field", "linarith", "nlinarith"],
        "logic": ["forall", "exists", "intro", "exact", "apply"],
        "order": ["le", "lt", "ge", "gt", "monotone"],
        "rewrite": ["rw", "simp", "conv", "change"],
        "arith": ["omega", "norm_num", "zify"],
    }
    low = full.lower()
    key_hits = [sum(1 for k in v if k in low) for v in keyword_set.values()]
    punct = [full.count(ch) for ch in ["(", ")", "[", "]", "{", "}", "=", "+", "-", "*"]]
    ratios = [math.log1p(chars) / 10.0, math.log1p(tok_count) / 10.0, uniq / tok_count, len(proof) / max(chars, 1)]

    # V2 adds hashed token bins for less trivial separability.
    bins = [0.0] * 16
    for t in toks:
        idx = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:2], 16) % len(bins)
        bins[idx] += 1.0
    bins = [b / tok_count for b in bins]

    raw = ratios + [h / 5.0 for h in key_hits] + [p / max(chars, 1) for p in punct] + bins

    vec = np.zeros((input_dim,), dtype=np.float32)
    for i in range(min(len(raw), input_dim)):
        vec[i] = float(raw[i])
    if input_dim > len(raw):
        h = hashlib.md5(full.encode("utf-8", errors="ignore")).digest()
        for i in range(len(raw), input_dim):
            vec[i] = (h[i % len(h)] / 255.0) * 2.0 - 1.0
    return vec.tolist()


def prepare_minif2f_rows(rows: List[Dict], input_dim: int, num_blocks: int) -> List[Dict]:
    out = []
    for row in rows:
        text = _text_from_row(row)
        proof = _proof_from_row(row)
        label = float(row["label"]) if "label" in row else (1.0 if proof.strip() else 0.0)
        block_id = int(row.get("block_id", _bucket_from_text(text, num_blocks)))
        block_id = max(0, min(block_id, num_blocks - 1))
        features = _dense_features(text=text, proof=proof, input_dim=input_dim)
        out.append({"features": features, "label": label, "block_id": block_id})
    return out


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows_three_way(rows: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n = len(rows)
    c1 = int(n * train_ratio)
    c2 = int(n * (train_ratio + val_ratio))
    train = [rows[i] for i in idx[:c1]]
    val = [rows[i] for i in idx[c1:c2]]
    test = [rows[i] for i in idx[c2:]]
    return train, val, test
