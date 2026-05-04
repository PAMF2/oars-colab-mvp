import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/minif2f_raw_sample.jsonl")
    parser.add_argument("--prepared", default="data/minif2f_prepared.jsonl")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=800)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    print("[1/4] preparing dataset")
    print(run([
        sys.executable,
        "scripts/prepare_minif2f.py",
        "--input", args.raw,
        "--output", args.prepared,
        "--write-split",
    ], cwd=root))

    print("[2/4] validating prepared dataset")
    print(run([
        sys.executable,
        "scripts/validate_dataset.py",
        "--input", args.prepared,
        "--input-dim", "24",
    ], cwd=root))

    print("[3/4] running ablation (synthetic fast path)")
    print(run([
        sys.executable,
        "scripts/run_ablation.py",
        "--epochs", str(args.epochs),
        "--samples", str(args.samples),
        "--seeds", str(args.seeds),
    ], cwd=root))

    print("[4/4] plotting")
    print(run([
        sys.executable,
        "scripts/plot_results.py",
        "--csv", "outputs/ablation_summary.csv",
    ], cwd=root))

    print(json.dumps({"status": "ok", "summary": "outputs/ablation_summary.csv"}, indent=2))


if __name__ == "__main__":
    main()
