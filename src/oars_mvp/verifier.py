from dataclasses import dataclass
import os
import re
import shutil
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


def default_lean_cmd() -> str:
    elan_lean = os.path.expanduser("~/.elan/bin/lean")
    if os.path.exists(elan_lean):
        return elan_lean
    return "lean"


class StrictHeuristicVerifier(BaseVerifier):
    name = "heuristic_strict"

    def verify(self, theorem_statement: str, proof: str) -> VerificationResult:
        low = proof.lower().strip()
        st = theorem_statement.lower()

        if not low:
            return VerificationResult(False, "empty proof", self.name)
        if not low.startswith("by"):
            return VerificationResult(False, "missing by", self.name)

        tactic_tokens = [
            "simp", "rw", "linarith", "nlinarith", "omega", "ring", "field_simp", "calc", "have", "exact", "apply",
            "induction", "cases", "aesop", "norm_num", "zify",
        ]
        t_count = sum(1 for t in tactic_tokens if t in low)
        if t_count < 2:
            return VerificationResult(False, "too few tactics", self.name)

        bad = ["sorry", "admit", "exact?", "by trivial", "by decide"]
        if any(b in low for b in bad):
            return VerificationResult(False, "placeholder/weak proof", self.name)

        if ("+" in st or "nat.add" in st) and not any(t in low for t in ["ring", "linarith", "omega", "rw", "simp"]):
            return VerificationResult(False, "missing arithmetic reasoning", self.name)

        if ("forall" in st or "∀" in st or "exists" in st or "∃" in st) and not any(t in low for t in ["intro", "have", "apply", "exact"]):
            return VerificationResult(False, "missing quantified reasoning", self.name)

        toks = re.findall(r"[a-zA-Z_]+", low)
        if len(toks) < 6:
            return VerificationResult(False, "proof too short", self.name)

        return VerificationResult(True, "strict-heuristic-pass", self.name)


class LeanCliVerifier(BaseVerifier):
    name = "lean-cli"

    def __init__(self, lean_cmd: str = None):
        self.lean_cmd = lean_cmd or default_lean_cmd()

    @staticmethod
    def _strip_leading_by(proof: str) -> str:
        p = (proof or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not p:
            return ""
        # Remove a single leading `by` token, robust to spacing/newlines.
        p = re.sub(r"^\s*by\b", "", p, count=1).lstrip()
        return p

    @staticmethod
    def _normalize_statement(theorem_statement: str) -> str:
        st = (theorem_statement or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        # If the statement already contains `:= by`, drop any old body and rebuild.
        if re.search(r":=\s*by\b", st):
            head = re.split(r":=\s*by\b", st, maxsplit=1)[0].rstrip()
            return f"{head} := by"
        # If it ends with a bare `:=`, just complete it.
        if re.search(r":=\s*$", st):
            head = re.sub(r":=\s*$", "", st).rstrip()
            return f"{head} := by"
        return st + "\n:= by"

    def _build_snippet(self, theorem_statement: str, proof: str) -> str:
        statement = self._normalize_statement(theorem_statement)
        body = self._strip_leading_by(proof)
        if not body:
            body = "  sorry"
        else:
            body = "\n".join("  " + ln for ln in body.splitlines())
        return f"import Mathlib\n\n{statement}\n{body}\n"

    def verify(self, theorem_statement: str, proof: str) -> VerificationResult:
        snippet = self._build_snippet(theorem_statement, proof)
        try:
            proc = subprocess.run(
                [self.lean_cmd, "--stdin"],
                input=snippet,
                text=True,
                capture_output=True,
                timeout=25,
            )
        except FileNotFoundError:
            return VerificationResult(False, f"lean binary not found: {self.lean_cmd}", self.name)
        except subprocess.TimeoutExpired:
            return VerificationResult(False, "verification timeout", self.name)

        ok = proc.returncode == 0
        msg = (proc.stdout + "\n" + proc.stderr).strip()
        if not msg:
            msg = "ok" if ok else "failed"
        return VerificationResult(ok, msg[:800], self.name)


def lean_available() -> bool:
    if shutil.which("lean") is not None:
        return True
    return os.path.exists(os.path.expanduser("~/.elan/bin/lean"))


def make_verifier(kind: str) -> BaseVerifier:
    kind = (kind or "heuristic").lower()
    if kind == "lean":
        return LeanCliVerifier()
    if kind == "auto":
        return LeanCliVerifier() if lean_available() else StrictHeuristicVerifier()
    return StrictHeuristicVerifier()
