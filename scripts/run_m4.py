import argparse
import json
from pathlib import Path

from phase1_common import (
    assign_concepts,
    build_rows,
    dpmeans_fit_assign,
    ib_keep_concepts,
    load_cfg,
    mean_ci95,
    read_jsonl,
    run_cfg,
)
from oars_mvp.dataset import write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="minif2f_raw.jsonl")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--dp-lam", type=float, default=1.25)
    p.add_argument("--max-concepts", type=int, default=16)
    p.add_argument("--keep-concepts", type=int, default=8)
    p.add_argument("--output-dir", default="outputs/phase1/M4")
    args = p.parse_args()

    base_cfg = load_cfg()
    raw = read_jsonl(args.raw)
    rows = build_rows(raw, input_dim=base_cfg["input_dim"], num_blocks=base_cfg["num_blocks"])

    mini = [r for r in rows if r.get("benchmark") == "minif2f"]
    putnam = [r for r in rows if r.get("benchmark") == "putnambench"]
    if not mini:
        mini = rows
    if not putnam:
        putnam = rows[: max(1, len(rows) // 4)]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transfer_scores, kept_counts, runs = [], [], []
    for s in range(args.seeds):
        val_size = max(1, int(len(mini) * 0.15))
        train = mini[val_size:]
        val = mini[:val_size]

        # Discover concepts on mini train and assign across train/val/test.
        all_rows = train + val + putnam
        assign, _ = dpmeans_fit_assign(train, all_rows, lam=args.dp_lam, max_concepts=args.max_concepts)

        t_train = assign_concepts(train, assign[: len(train)])
        t_val = assign_concepts(val, assign[len(train): len(train) + len(val)])
        t_put = assign_concepts(putnam, assign[len(train) + len(val):])

        # IB proxy filter: keep only concepts with high reward*compression score on train+val.
        keep = ib_keep_concepts(t_train + t_val, max_keep=args.keep_concepts)
        if keep:
            t_put_f = [r for r in t_put if int(r.get("concept_id", -1)) in keep]
        else:
            t_put_f = t_put
        if not t_put_f:
            t_put_f = t_put[: max(1, len(t_put) // 4)]

        sp = out / f"seed_{s}"
        sp.mkdir(parents=True, exist_ok=True)
        train_p = sp / "train.jsonl"
        val_p = sp / "val.jsonl"
        test_p = sp / "test.jsonl"
        write_jsonl(str(train_p), t_train)
        write_jsonl(str(val_p), t_val)
        write_jsonl(str(test_p), t_put_f)

        n_concepts = len({int(x.get("concept_id", 0)) for x in t_train + t_val + t_put_f})
        r = run_cfg(
            base_cfg,
            mode="arla_full",
            train_p=train_p,
            val_p=val_p,
            test_p=test_p,
            seed=base_cfg.get("seed", 42) + s,
            epochs=args.epochs,
            out_dir=out / "raw_runs",
            extra_task={"arla_aux_weight": 1.2, "num_classes": int(max(2, n_concepts))},
        )

        transfer_scores.append(r["metrics"]["acc"])
        kept_counts.append(len(keep))
        runs.append({"seed": s, "kept_concepts": sorted(list(keep)), "result": r})

    t_m, t_ci = mean_ci95(transfer_scores)
    k_m, k_ci = mean_ci95(kept_counts)
    summary = {
        "milestone": "M4",
        "primary_metric": "transfer_success (miniF2F -> Putnam family holdout)",
        "diagnostic": "IB proxy: reward-compression concept filter",
        "transfer_mean": t_m,
        "transfer_ci95": t_ci,
        "kept_concepts_mean": k_m,
        "kept_concepts_ci95": k_ci,
        "runs": len(runs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
