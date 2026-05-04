import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticProofDataset(Dataset):
    """Synthetic theorem-like tasks for fast MVP experiments."""

    def __init__(self, n_samples: int, input_dim: int, num_blocks: int, seed: int = 42):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.num_blocks = num_blocks
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
    """Loads local jsonl with {features: [...], label: 0/1, block_id: int?}."""

    def __init__(self, path: str, input_dim: int, num_blocks: int):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
