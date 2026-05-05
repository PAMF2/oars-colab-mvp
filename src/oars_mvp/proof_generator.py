import random
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


class ModelProofGenerator:
    """Proof generator backed by a causal LM fine-tuned on statement->proof."""

    def __init__(self, model_path: str, seed: int = 42, max_new_tokens: int = 160):
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _prompt(self, statement: str) -> str:
        return f"### Problem\n{statement}\n\n### Lean proof\nby\n"

    def generate(self, statement: str, k: int) -> List[CandidateProof]:
        prompt = self._prompt(statement)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=k,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        cands: List[CandidateProof] = []
        for i in range(out.size(0)):
            txt = self.tokenizer.decode(out[i], skip_special_tokens=True)
            proof = txt.split("### Lean proof", 1)[-1].strip()
            if not proof.startswith("by"):
                proof = "by\n" + proof
            cands.append(CandidateProof(text=proof, score=max(1e-6, 1.0 - (i / max(k, 1)))))
        return cands


class HybridProofGenerator:
    """Mix LM-generated candidates with tactic templates for diversity."""

    def __init__(self, model_path: str, seed: int = 42):
        self.model_gen = ModelProofGenerator(model_path=model_path, seed=seed)
        self.tactic_gen = TacticGenerator(seed=seed)

    def generate(self, statement: str, k: int) -> List[CandidateProof]:
        k_model = max(1, int(k * 0.7))
        k_tac = max(1, k - k_model)
        cands = self.model_gen.generate(statement, k_model) + self.tactic_gen.generate(statement, k_tac)
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:k]


class AirLLMProofGenerator:
    """Inference-only baseline with AirLLM for low VRAM environments."""

    def __init__(self, model_path: str, seed: int = 42, max_new_tokens: int = 160):
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        try:
            from airllm import AutoModel  # type: ignore
        except Exception as e:
            raise RuntimeError("airllm is not installed. Install with: pip install airllm") from e
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = self.model.tokenizer

    def _prompt(self, statement: str) -> str:
        return f"### Problem\n{statement}\n\n### Lean proof\nby\n"

    def generate(self, statement: str, k: int) -> List[CandidateProof]:
        prompt = self._prompt(statement)
        input_tokens = self.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=1024,
            padding=False,
        )
        out = self.model.generate(
            input_tokens["input_ids"].cuda(),
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            num_return_sequences=max(1, k),
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
        seqs = out.sequences
        if seqs.ndim == 1:
            seqs = seqs.unsqueeze(0)
        cands: List[CandidateProof] = []
        for i in range(min(seqs.size(0), k)):
            txt = self.tokenizer.decode(seqs[i], skip_special_tokens=True)
            proof = txt.split("### Lean proof", 1)[-1].strip()
            if not proof.startswith("by"):
                proof = "by\n" + proof
            cands.append(CandidateProof(text=proof, score=max(1e-6, 1.0 - (i / max(k, 1)))))
        return cands


class AirLLMHybridProofGenerator:
    """Mix AirLLM candidates with tactic templates."""

    def __init__(self, model_path: str, seed: int = 42):
        self.model_gen = AirLLMProofGenerator(model_path=model_path, seed=seed)
        self.tactic_gen = TacticGenerator(seed=seed)

    def generate(self, statement: str, k: int) -> List[CandidateProof]:
        k_model = max(1, int(k * 0.7))
        k_tac = max(1, k - k_model)
        cands = self.model_gen.generate(statement, k_model) + self.tactic_gen.generate(statement, k_tac)
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:k]
