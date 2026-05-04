import random
from dataclasses import dataclass
from typing import List


@dataclass
class CandidateProof:
    text: str
    score: float


class TacticGenerator:
    """Stochastic candidate generator with stronger/weak pools.

    We mix templates to avoid trivially valid one-liners.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.strong = [
            "by\n  intro h\n  have h1 := h\n  try simp at h1\n  try linarith\n  aesop",
            "by\n  try simp\n  try rw [Nat.add_comm]\n  try omega\n  try linarith\n  aesop",
            "by\n  induction n with\n  | zero => simp\n  | succ n ih =>\n    simp [ih]\n    try omega",
            "by\n  have h1 := by\n    try ring\n    try nlinarith\n  try simpa using h1",
            "by\n  intro x\n  intro y\n  have hxy := by try linarith\n  try simpa using hxy",
        ]
        self.weak = [
            "by simpa",
            "by aesop",
            "by exact?",
            "by trivial",
            "by decide",
        ]

    def generate(self, statement: str, k: int) -> List[CandidateProof]:
        out: List[CandidateProof] = []
        for _ in range(k):
            if self.rng.random() < 0.65:
                t = self.rng.choice(self.strong)
                s = 0.4 + self.rng.random() * 0.6
            else:
                t = self.rng.choice(self.weak)
                s = self.rng.random() * 0.5
            out.append(CandidateProof(text=t, score=s))

        out.sort(key=lambda c: c.score, reverse=True)
        return out
