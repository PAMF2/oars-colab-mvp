from dataclasses import dataclass
from typing import Optional
import subprocess


@dataclass
class VerificationResult:
    ok: bool
    message: str
    verifier: str


class BaseVerifier:
    name = "base"

    def verify(self, theorem_statement: str, proof: str) -> VerificationResult:
        raise NotImplementedError


class HeuristicVerifier(BaseVerifier):
    """Fast fallback verifier for smoke tests when Lean is unavailable."""

    name = "heuristic"

    def verify(self, theorem_statement: str, proof: str) -> VerificationResult:
        low = proof.lower().strip()
        if not low:
            return VerificationResult(False, "empty proof", self.name)
        # Very weak proxy accepted patterns.
        good_tokens = ["by", "simpa", "exact", "rw", "ring", "linarith", "omega", "aesop"]
        ok = any(t in low for t in good_tokens)
        return VerificationResult(ok, "heuristic-check", self.name)


class LeanCliVerifier(BaseVerifier):
    """Verifies Lean snippets by calling local `lean` binary.

    Requires Lean toolchain installed in environment.
    """

    name = "lean-cli"

    def __init__(self, lean_cmd: str = "lean"):
        self.lean_cmd = lean_cmd

    def verify(self, theorem_statement: str, proof: str) -> VerificationResult:
        snippet = f"{theorem_statement}\n{proof}\n"
        try:
            proc = subprocess.run(
                [self.lean_cmd, "--stdin"],
                input=snippet,
                text=True,
                capture_output=True,
                timeout=20,
            )
        except FileNotFoundError:
            return VerificationResult(False, "lean binary not found", self.name)
        except subprocess.TimeoutExpired:
            return VerificationResult(False, "verification timeout", self.name)

        ok = proc.returncode == 0
        msg = (proc.stdout + "\n" + proc.stderr).strip()
        if not msg:
            msg = "ok" if ok else "failed"
        return VerificationResult(ok, msg[:800], self.name)


def make_verifier(kind: str) -> BaseVerifier:
    kind = (kind or "heuristic").lower()
    if kind == "lean":
        return LeanCliVerifier()
    return HeuristicVerifier()
