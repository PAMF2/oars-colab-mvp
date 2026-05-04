import argparse
import json
from pathlib import Path

from .dataset import prepare_minif2f_rows, read_jsonl, split_rows, write_jsonl
from .stats import summarize_csv


def cmd_prepare(args):
    rows = read_jsonl(args.input)
    prepared = prepare_minif2f_rows(rows, input_dim=args.input_dim, num_blocks=args.num_blocks)
    write_jsonl(args.output, prepared)

    meta = {
        "input": args.input,
        "output": args.output,
        "rows": len(prepared),
        "input_dim": args.input_dim,
        "num_blocks": args.num_blocks,
    }

    if args.write_split:
        train_rows, val_rows = split_rows(prepared, train_ratio=args.train_ratio, seed=args.seed)
        out = Path(args.output)
        train_path = str(out.with_name(out.stem + "_train.jsonl"))
        val_path = str(out.with_name(out.stem + "_val.jsonl"))
        write_jsonl(train_path, train_rows)
        write_jsonl(val_path, val_rows)
        meta["train_path"] = train_path
        meta["val_path"] = val_path
        meta["train_rows"] = len(train_rows)
        meta["val_rows"] = len(val_rows)

    print(json.dumps(meta, indent=2))


def cmd_stats(args):
    out = summarize_csv(args.csv)
    print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="oars-mvp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Prepare miniF2F-like raw jsonl")
    p_prepare.add_argument("--input", required=True)
    p_prepare.add_argument("--output", default="data/minif2f_prepared.jsonl")
    p_prepare.add_argument("--input-dim", type=int, default=24)
    p_prepare.add_argument("--num-blocks", type=int, default=4)
    p_prepare.add_argument("--train-ratio", type=float, default=0.8)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--write-split", action="store_true")
    p_prepare.set_defaults(func=cmd_prepare)

    p_stats = sub.add_parser("stats", help="Summarize ablation CSV with CI95")
    p_stats.add_argument("--csv", default="outputs/ablation_summary.csv")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
