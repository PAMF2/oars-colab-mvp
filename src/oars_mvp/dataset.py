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
        rng = np.random.default_rng(seed)
        self.x = rng.normal(0, 1, size=(n_samples, input_dim)).astype(np.float32)
        self.block_id = rng.integers(low=0, high=num_blocks, size=n_samples, endpoint=False)
        w = rng.normal(0, 0.7, size=(num_blocks, input_dim)).astype(np.float32)
        b = rng.normal(0, 0.25, size=(num_blocks,)).astype(np.float32)
        logits = np.array([np.dot(self.x[i], w[self.block_id[i]]) + b[self.block_id[i]] for i in range(n_samples)])
        probs = 1 / (1 + np.exp(-logits))
        self.y = (probs > 0.5).astype(np.float32)
        self.class_label = self.block_id.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "block_id": torch.tensor(self.block_id[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "class_label": torch.tensor(self.class_label[idx], dtype=torch.long),
        }


class MiniF2FLikeDataset(Dataset):
    def __init__(self, path: str, input_dim: int, num_blocks: int):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        rows = read_jsonl(path)
        if not rows:
            raise ValueError("Dataset file is empty")

        x, y, b, c = [], [], [], []
        for i, row in enumerate(rows):
            feats = row.get("features")
            if not isinstance(feats, list) or len(feats) != input_dim:
                raise ValueError(f"Row {i} has invalid features; expected list length {input_dim}")
            label = float(row.get("label", 0.0))
            block = int(row.get("block_id", i % max(num_blocks, 1)))
            block = max(0, min(block, num_blocks - 1))
            cls = int(row.get("class_label", block))
            x.append(feats)
            y.append(label)
            b.append(block)
            c.append(cls)

        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.block_id = np.asarray(b, dtype=np.int64)
        self.class_label = np.asarray(c, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "block_id": torch.tensor(self.block_id[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "class_label": torch.tensor(self.class_label[idx], dtype=torch.long),
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


def _family_class(text: str, proof: str) -> int:
    low = (text + "\n" + proof).lower()
    families = [
        ["induction", "nat.rec", "cases"],
        ["ring", "field", "linarith", "nlinarith", "norm_num"],
        ["simp", "rw", "change", "conv"],
        ["forall", "exists", "intro", "exact", "apply"],
    ]
    scores = [sum(1 for t in fam if t in low) for fam in families]
    best = int(np.argmax(scores))
    if scores[best] == 0:
        return 3
    return best


def _dense_features(text: str, proof: str, input_dim: int) -> List[float]:
    full = (text + "\n" + proof).strip()
    toks = re.findall(r"[A-Za-z0-9_']+|[=+\-*/^(){}\[\]<>]|∀|∃|→|↔|≤|≥", full)
    tok_count = max(len(toks), 1)
    uniq = len(set(toks))
    chars = len(full)

    key_groups = {
        "induction": ["induction", "nat.rec", "cases", "simp"],
        "algebra": ["ring", "field", "linarith", "nlinarith"],
        "logic": ["forall", "exists", "intro", "exact", "apply"],
        "order": ["le", "lt", "ge", "gt", "monotone"],
        "rewrite": ["rw", "simp", "conv", "change"],
        "arith": ["omega", "norm_num", "zify"],
    }
    low = full.lower()
    key_hits = [sum(1 for k in v if k in low) for v in key_groups.values()]
    punct = [full.count(ch) for ch in ["(", ")", "[", "]", "{", "}", "=", "+", "-", "*"]]
    ratios = [math.log1p(chars) / 10.0, math.log1p(tok_count) / 10.0, uniq / tok_count, len(proof) / max(chars, 1)]

    bins = [0.0] * 24
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
        class_label = int(row.get("class_label", _family_class(text, proof)))
        features = _dense_features(text=text, proof=proof, input_dim=input_dim)
        out.append({"features": features, "label": label, "block_id": block_id, "class_label": class_label})
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


def split_rows_three_way_stratified(rows: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios")
    rng = np.random.default_rng(seed)

    by_cls = {}
    for r in rows:
        k = int(r.get("class_label", 0))
        by_cls.setdefault(k, []).append(r)

    train, val, test = [], [], []
    for cls_rows in by_cls.values():
        idx = np.arange(len(cls_rows))
        rng.shuffle(idx)
        n = len(cls_rows)
        c1 = int(n * train_ratio)
        c2 = int(n * (train_ratio + val_ratio))
        # Ensure at least one sample in val/test when class has enough items.
        if n >= 3:
            c1 = max(1, min(c1, n - 2))
            c2 = max(c1 + 1, min(c2, n - 1))
        train.extend([cls_rows[i] for i in idx[:c1]])
        val.extend([cls_rows[i] for i in idx[c1:c2]])
        test.extend([cls_rows[i] for i in idx[c2:]])

    # Global safety for tiny datasets.
    if len(val) == 0 and len(train) > 1:
        val.append(train.pop())
    if len(test) == 0 and len(train) > 1:
        test.append(train.pop())

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test

