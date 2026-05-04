import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticProofDataset(Dataset):
    """Synthetic theorem-like tasks for fast MVP experiments.

    Each sample has:
    - x: feature vector
    - block_id: dominant concept block id
    - y: binary success label
    """

    def __init__(self, n_samples: int, input_dim: int, num_blocks: int, seed: int = 42):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        rng = np.random.default_rng(seed)

        self.x = rng.normal(0, 1, size=(n_samples, input_dim)).astype(np.float32)
        self.block_id = rng.integers(low=0, high=num_blocks, size=n_samples, endpoint=False)

        # Label depends on sparse subset and block interaction.
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
