import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="Tonic/MiniF2F")
    p.add_argument("--filename", default="minif2f.jsonl")
    p.add_argument("--out", default="minif2f_raw.jsonl")
    args = p.parse_args()

    src = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=args.filename)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
