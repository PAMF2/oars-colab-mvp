import argparse
import json
import math
import statistics
from pathlib import Path


def read_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-dim", type=int, default=24)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("dataset is empty")

    labels, blocks, norms = [], [], []
    for i, row in enumerate(rows):
        feats = row.get("features")
        if not isinstance(feats, list) or len(feats) != args.input_dim:
            raise SystemExit(f"invalid features at row {i}: expected length {args.input_dim}")
        y = float(row.get("label", 0.0))
        if y not in (0.0, 1.0):
            raise SystemExit(f"invalid label at row {i}: {y}")
        labels.append(y)
        blocks.append(int(row.get("block_id", 0)))
        norms.append(math.sqrt(sum(float(v) * float(v) for v in feats)))

    pos_rate = sum(labels) / len(labels)
    out = {
        "rows": len(rows),
        "input_dim": args.input_dim,
        "positive_rate": round(pos_rate, 4),
        "block_count": len(set(blocks)),
        "feature_norm_mean": round(statistics.mean(norms), 4),
        "feature_norm_std": round(statistics.pstdev(norms), 4),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
