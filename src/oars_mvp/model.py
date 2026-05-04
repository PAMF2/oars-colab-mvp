import torch
import torch.nn as nn
import torch.nn.functional as F


class ARLAAllocator(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int, concepts_per_block: int):
        super().__init__()
        self.block_head = nn.Linear(hidden_dim, num_blocks)
        self.concept_head = nn.Linear(hidden_dim, num_blocks * concepts_per_block)
        self.num_blocks = num_blocks
        self.concepts_per_block = concepts_per_block

    def forward(self, h):
        block_logits = self.block_head(h)
        concept_logits = self.concept_head(h).view(-1, self.num_blocks, self.concepts_per_block)
        block_probs = F.softmax(block_logits, dim=-1)
        concept_probs = F.softmax(concept_logits, dim=-1)
        return block_logits, concept_logits, block_probs, concept_probs


class OARSMVP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, concepts_per_block: int, num_classes: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.allocator = ARLAAllocator(hidden_dim, num_blocks, concepts_per_block)
        self.shared_head = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
        self.num_blocks = num_blocks

        # ARLA expert heads (one per block) for multiclass task.
        self.block_heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_blocks)])

    def _shared_logits(self, z):
        logits = self.shared_head(z)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        return logits

    def _arla_expert_logits(self, z, block_probs, hard: bool):
        # [B, K, C]
        expert = torch.stack([head(z) for head in self.block_heads], dim=1)
        if hard:
            idx = block_probs.argmax(dim=-1)
            mask = F.one_hot(idx, num_classes=self.num_blocks).float()
            w = mask.unsqueeze(-1)
        else:
            w = block_probs.unsqueeze(-1)
        logits = (expert * w).sum(dim=1)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        return logits

    def forward(self, x, mode: str = "baseline"):
        h = self.encoder(x)
        block_logits, concept_logits, block_probs, concept_probs = self.allocator(h)

        if mode == "baseline":
            z = h
            logits = self._shared_logits(z)
        elif mode == "hierarchical_only":
            z = h + 0.05 * torch.tanh(h)
            logits = self._shared_logits(z)
        elif mode == "arla_block":
            block_idx = block_probs.argmax(dim=-1)
            conf = block_probs.gather(1, block_idx.unsqueeze(-1))
            z = h * (0.35 + 1.65 * conf)
            logits = self._arla_expert_logits(z, block_probs, hard=False)
        elif mode == "arla_full":
            block_idx = block_probs.argmax(dim=-1)
            conf = block_probs.gather(1, block_idx.unsqueeze(-1))
            concept_conf = concept_probs.max(dim=-1).values.mean(dim=-1, keepdim=True)
            z = h * (0.25 + 1.25 * conf + 0.9 * concept_conf)
            logits = self._arla_expert_logits(z, block_probs, hard=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return logits, block_logits, concept_logits, block_probs, concept_probs
