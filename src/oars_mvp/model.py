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
        return block_probs, concept_probs


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
        self.policy = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x, mode: str = "baseline"):
        h = self.encoder(x)
        block_probs, concept_probs = self.allocator(h)

        if mode == "baseline":
            z = h
        elif mode == "hierarchical_only":
            z = h + 0.05 * torch.tanh(h)
        elif mode == "arla_block":
            weights = block_probs.max(dim=-1).values.unsqueeze(-1)
            z = h * (1.0 + weights)
        elif mode == "arla_full":
            block_weight = block_probs.max(dim=-1).values.unsqueeze(-1)
            concept_weight = concept_probs.max(dim=-1).values.mean(dim=-1, keepdim=True)
            z = h * (1.0 + block_weight + concept_weight)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        logits = self.policy(z)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        return logits, block_probs, concept_probs
