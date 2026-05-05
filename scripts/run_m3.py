import argparse
import json
from pathlib import Path

from phase1_common import (
    assign_concepts,
    build_rows,
    dpmeans_fit_assign,
    load_cfg,
    mean_ci95,
    read_jsonl,
    run_cfg,
    split_train_val_test,
)
from oars_mvp.dataset import write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="minif2f_raw.jsonl")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--dp-lam", type=float, default=1.25)
    p.add_argument("--max-concepts", type=int, default=16)
    p.add_argument("--output-dir", default="outputs/phase1/M3")
    args = p.parse_args()

    base_cfg = load_cfg()
    raw = read_jsonl(args.raw)
    rows = build_rows(raw, input_dim=base_cfg["input_dim"], num_blocks=base_cfg["num_blocks"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    delta_pass, concept_counts, runs = [], [], []
    for s in range(args.seeds):
        train, val, test = split_train_val_test(rows, seed=base_cfg.get("seed", 42) + s)
        sp = out / f"seed_{s}"

        # Hardcoded concept regime.
        h_dir = sp / "hardcoded"
        h_dir.mkdir(parents=True, exist_ok=True)
        h_train_p = h_dir / "train.jsonl"
        h_val_p = h_dir / "val.jsonl"
        h_test_p = h_dir / "test.jsonl"
        write_jsonl(str(h_train_p), train)
        write_jsonl(str(h_val_p), val)
        write_jsonl(str(h_test_p), test)

        rh = run_cfg(
            base_cfg,
            mode="arla_block",
            train_p=h_train_p,
            val_p=h_val_p,
            test_p=h_test_p,
            seed=base_cfg.get("seed", 42) + s,
            epochs=args.epochs,
            out_dir=out / "raw_runs",
            extra_task={"arla_aux_weight": 0.8, "num_classes": 4},
        )

        # Discovered concept regime via DP-means.
        all_rows = train + val + test
        assign, _ = dpmeans_fit_assign(train, all_rows, lam=args.dp_lam, max_concepts=args.max_concepts)
        d_train = assign_concepts(train, assign[: len(train)])
        d_val = assign_concepts(val, assign[len(train): len(train) + len(val)])
        d_test = assign_concepts(test, assign[len(train) + len(val):])

        d_dir = sp / "discovered"
        d_dir.mkdir(parents=True, exist_ok=True)
        d_train_p = d_dir / "train.jsonl"
        d_val_p = d_dir / "val.jsonl"
        d_test_p = d_dir / "test.jsonl"
        write_jsonl(str(d_train_p), d_train)
        write_jsonl(str(d_val_p), d_val)
        write_jsonl(str(d_test_p), d_test)

        n_concepts = len(set(int(x) for x in assign.tolist()))
        rd = run_cfg(
            base_cfg,
            mode="arla_full",
            train_p=d_train_p,
            val_p=d_val_p,
            test_p=d_test_p,
            seed=base_cfg.get("seed", 42) + s,
            epochs=args.epochs,
            out_dir=out / "raw_runs",
            extra_task={"arla_aux_weight": 1.2, "num_classes": int(max(2, n_concepts))},
        )

        d = rd["metrics"]["acc"] - rh["metrics"]["acc"]
        delta_pass.append(d)
        concept_counts.append(int(max(1, n_concepts)))
        runs.append(
            {
                "seed": s,
                "hardcoded": rh,
                "discovered": rd,
                "delta_pass": d,
                "n_concepts": int(n_concepts),
            }
        )

    d_m, d_ci = mean_ci95(delta_pass)
    c_m, c_ci = mean_ci95(concept_counts)
    summary = {
        "milestone": "M3",
        "primary_metric": "delta_pass_rate (discovered - hardcoded)",
        "diagnostic": "dpmeans_concept_count",
        "delta_pass_mean": d_m,
        "delta_pass_ci95": d_ci,
        "concept_count_mean": c_m,
        "concept_count_ci95": c_ci,
        "runs": len(runs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
