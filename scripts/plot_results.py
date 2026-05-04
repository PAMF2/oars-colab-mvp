import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oars_mvp.stats import summarize_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="outputs/ablation_summary.csv")
    parser.add_argument("--outdir", type=str, default="outputs/plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    agg = df.groupby("mode", as_index=False).agg(
        acc_mean=("acc", "mean"),
        reward_mean=("reward", "mean"),
        entropy_mean=("allocator_entropy", "mean"),
        runtime_mean=("runtime_sec", "mean"),
    )

    for col, title in [
        ("acc_mean", "Accuracy by Mode"),
        ("reward_mean", "Reward by Mode"),
        ("entropy_mean", "Allocator Entropy by Mode"),
        ("runtime_mean", "Runtime (s) by Mode"),
    ]:
        plt.figure(figsize=(8, 4))
        plt.bar(agg["mode"], agg[col])
        plt.title(title)
        plt.xticks(rotation=15)
        plt.tight_layout()
        out = outdir / f"{col}.png"
        plt.savefig(out, dpi=140)
        plt.close()

    agg.to_csv(outdir / "aggregated_metrics.csv", index=False)
    stats = summarize_csv(args.csv)
    with (outdir / "stats_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"saved plots to {outdir}")


if __name__ == "__main__":
    main()
