import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oars_mvp.dataset import prepare_minif2f_rows, read_jsonl, split_rows_three_way_stratified, write_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/minif2f_prepared.jsonl")
    parser.add_argument("--input-dim", type=int, default=80)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-split", action="store_true")
    args = parser.parse_args()

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
        train_rows, val_rows, test_rows = split_rows_three_way_stratified(
            prepared, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
        )
        out = Path(args.output)
        train_path = str(out.with_name(out.stem + "_train.jsonl"))
        val_path = str(out.with_name(out.stem + "_val.jsonl"))
        test_path = str(out.with_name(out.stem + "_test.jsonl"))
        write_jsonl(train_path, train_rows)
        write_jsonl(val_path, val_rows)
        write_jsonl(test_path, test_rows)
        meta.update(
            {
                "train_path": train_path,
                "val_path": val_path,
                "test_path": test_path,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "test_rows": len(test_rows),
            }
        )

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
