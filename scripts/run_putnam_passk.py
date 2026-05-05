import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oars_mvp.proof_generator import HybridProofGenerator, ModelProofGenerator, TacticGenerator
from oars_mvp.verifier import make_verifier, lean_available


def ensure_lean_path():
    elan_bin = os.path.expanduser("~/.elan/bin")
    if os.path.isdir(elan_bin):
        cur = os.environ.get("PATH", "")
        if elan_bin not in cur.split(":"):
            os.environ["PATH"] = elan_bin + ":" + cur


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_statement(row):
    for k in ["formal_statement", "statement", "goal", "theorem", "text"]:
        if row.get(k):
            return str(row[k])
    return "theorem unknown : True := by trivial"


def statement_seems_valid(st: str) -> bool:
    s = (st or "").strip()
    if not s.startswith("theorem"):
        return False
    # Exclude clearly truncated rows from raw dumps.
    if s.endswith("(h") or s.endswith(":") or s.count("(") < s.count(")"):
        return False
    return True


def pass_at_k(successes: int, n: int, k: int) -> float:
    if n == 0:
        return 0.0
    if n - successes < k:
        return 1.0
    prod = 1.0
    for i in range(n - successes + 1, n + 1):
        prod *= (1.0 - k / i)
    return 1.0 - prod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="minif2f_raw.jsonl")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--verifier", choices=["heuristic", "lean", "auto"], default="auto")
    p.add_argument("--require-lean", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--generator", choices=["tactic", "model", "hybrid"], default="tactic")
    p.add_argument("--model-path", default="outputs/real_model/final")
    p.add_argument("--save-details", action="store_true", default=False)
    p.add_argument("--out", default="outputs/putnam_passk_report.json")
    args = p.parse_args()

    ensure_lean_path()
    have_lean = lean_available()

    if (args.verifier == "lean" or args.require_lean) and not have_lean:
        raise RuntimeError(
            "Lean requested but not available in PATH. "
            "Try: source ~/.elan/env or install via scripts/install_lean.py"
        )

    rows = read_jsonl(args.data)
    usable_rows = []
    for row in rows:
        st = get_statement(row)
        if statement_seems_valid(st):
            usable_rows.append(row)
        if len(usable_rows) >= args.limit:
            break

    if args.generator == "model":
        gen = ModelProofGenerator(model_path=args.model_path, seed=args.seed)
    elif args.generator == "hybrid":
        gen = HybridProofGenerator(model_path=args.model_path, seed=args.seed)
    else:
        gen = TacticGenerator(seed=args.seed)
    verifier = make_verifier(args.verifier)

    solved = 0
    per_problem = []
    first_fail = None

    for i, row in enumerate(usable_rows):
        st = get_statement(row)
        cands = gen.generate(st, args.k)

        ok_any = False
        checked = []
        for c in cands:
            r = verifier.verify(st, c.text)
            checked.append({"proof": c.text, "score": c.score, "ok": r.ok, "msg": r.message[:120]})
            if r.ok:
                ok_any = True
                break

        solved += 1 if ok_any else 0
        row_detail = {"idx": i, "statement": st[:180], "solved": ok_any, "checked": checked}
        per_problem.append(row_detail)
        if first_fail is None and not ok_any:
            first_fail = row_detail

    n = len(usable_rows)
    passk = solved / max(n, 1)
    report = {
        "n_problems": n,
        "k": args.k,
        "verifier_requested": args.verifier,
        "verifier_used": verifier.name,
        "generator": args.generator,
        "model_path": args.model_path if args.generator in {"model", "hybrid"} else None,
        "lean_available": have_lean,
        "solved_count": solved,
        "pass_ratio": passk,
        "pass_at_k_estimate": pass_at_k(solved, n, args.k),
        "first_fail": first_fail,
    }
    if args.save_details:
        report["details"] = per_problem

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_problems": n,
        "solved_count": solved,
        "pass_ratio": passk,
        "pass_at_k_estimate": report["pass_at_k_estimate"],
        "verifier_used": verifier.name,
        "lean_available": have_lean,
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
