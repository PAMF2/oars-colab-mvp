import math

import pandas as pd


def _ci95(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * (std / math.sqrt(n))


def summarize_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    grouped = df.groupby("mode", as_index=False)
    has_f1 = "f1_macro" in df.columns

    rows = []
    for _, g in grouped:
        mode = g["mode"].iloc[0]
        n = len(g)
        acc_mean = float(g["acc"].mean())
        acc_std = float(g["acc"].std(ddof=0)) if n > 0 else 0.0
        reward_mean = float(g["reward"].mean())
        reward_std = float(g["reward"].std(ddof=0)) if n > 0 else 0.0
        runtime_mean = float(g["runtime_sec"].mean())

        row = {
            "mode": mode,
            "n": n,
            "acc_mean": round(acc_mean, 6),
            "acc_std": round(acc_std, 6),
            "acc_ci95": round(_ci95(acc_std, n), 6),
            "reward_mean": round(reward_mean, 6),
            "reward_std": round(reward_std, 6),
            "reward_ci95": round(_ci95(reward_std, n), 6),
            "runtime_mean": round(runtime_mean, 6),
        }

        if has_f1:
            f1_mean = float(g["f1_macro"].mean())
            f1_std = float(g["f1_macro"].std(ddof=0)) if n > 0 else 0.0
            row["f1_mean"] = round(f1_mean, 6)
            row["f1_std"] = round(f1_std, 6)
            row["f1_ci95"] = round(_ci95(f1_std, n), 6)

        rows.append(row)

    if has_f1:
        best = sorted(rows, key=lambda x: x.get("f1_mean", x["acc_mean"]), reverse=True)[0] if rows else None
    else:
        best = sorted(rows, key=lambda x: x["acc_mean"], reverse=True)[0] if rows else None
    return {"rows": rows, "best_mode": best}
