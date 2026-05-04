import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def go_no_go(delta_mean: float, delta_ci95: float) -> bool:
    return (delta_mean > 0.0) and ((delta_mean - delta_ci95) > 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m2", default="outputs/phase1/M2_robust/summary.json")
    p.add_argument("--m3", default="outputs/phase1/M3_robust/summary.json")
    p.add_argument("--out", default="outputs/phase1/decision_phase1.json")
    args = p.parse_args()

    m2 = load_json(Path(args.m2))
    m3 = load_json(Path(args.m3))

    m2_go = go_no_go(m2["delta_pass_mean"], m2["delta_pass_ci95"])
    m3_go = go_no_go(m3["delta_pass_mean"], m3["delta_pass_ci95"])

    decision = {
        "phase": "Phase I",
        "milestones": {
            "M2": {
                "delta_pass_mean": m2["delta_pass_mean"],
                "delta_pass_ci95": m2["delta_pass_ci95"],
                "go": m2_go,
            },
            "M3": {
                "delta_pass_mean": m3["delta_pass_mean"],
                "delta_pass_ci95": m3["delta_pass_ci95"],
                "go": m3_go,
            },
        },
        "overall": "GO" if (m2_go and m3_go) else "NO_GO",
        "recommendation": "Proceed to next phase" if (m2_go and m3_go) else "Run redesign cycle before advancing",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
