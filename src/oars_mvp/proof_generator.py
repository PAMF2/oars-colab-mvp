import random
from dataclasses import dataclass
from typing import List


@dataclass
class CandidateProof:
    text: str
    score: float


class TacticGenerator:
    """Simple stochastic generator for candidate Lean proofs.

    This is a bootstrap generator until a stronger prover backend is plugged in.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tactic_bank = [
            "by simpa",
            "by aesop",
            "by omega",
            "by linarith",
            "by ring",
            "by exact?",
            "by first | simp | omega",
            "by intro h; aesop",
            "by nlinarith",
            "by try simp; try omega; aesop",
        ]

    def generate(self, statement: str, k: int) -> List[CandidateProof]:
        out: List[CandidateProof] = []
        for _ in range(k):
            t = self.rng.choice(self.tactic_bank)
            s = self.rng.random()
            out.append(CandidateProof(text=t, score=s))

        out.sort(key=lambda c: c.score, reverse=True)
        return out
