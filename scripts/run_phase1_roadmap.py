import argparse
import json
import subprocess
from pathlib import Path


def run(cmd, cwd):
    p = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        raise SystemExit(p.returncode)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="minif2f_raw.jsonl")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="outputs/phase1/phase1_report.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    run(f"python scripts/run_m1.py --raw {args.raw} --seeds {args.seeds} --epochs {args.epochs}", cwd=root)
    run(f"python scripts/run_m2.py --raw {args.raw} --seeds {args.seeds} --epochs {args.epochs}", cwd=root)
    run(f"python scripts/run_m3.py --raw {args.raw} --seeds {args.seeds} --epochs {args.epochs}", cwd=root)
    run(f"python scripts/run_m4.py --raw {args.raw} --seeds {args.seeds} --epochs {args.epochs}", cwd=root)

    m1 = read_json(root / "outputs/phase1/M1/summary.json")
    m2 = read_json(root / "outputs/phase1/M2/summary.json")
    m3 = read_json(root / "outputs/phase1/M3/summary.json")
    m4 = read_json(root / "outputs/phase1/M4/summary.json")

    report = {
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "M4": m4,
        "checks": {
            "M2_delta_positive": m2["delta_pass_mean"] > 0,
            "M3_delta_positive": m3["delta_pass_mean"] > 0,
            "M4_transfer_positive": m4["transfer_mean"] > 0,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
