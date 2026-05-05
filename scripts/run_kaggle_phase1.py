import argparse
import json
import subprocess
from pathlib import Path

import yaml


def run(cmd: str, cwd: Path, allow_fail: bool = False):
    p = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if p.stdout:
        print(p.stdout)
    if p.returncode != 0:
        if p.stderr:
            print(p.stderr)
        if not allow_fail:
            raise SystemExit(p.returncode)
    return p.returncode


def ensure_raw(root: Path, raw_path: Path):
    if raw_path.exists():
        print(f"raw dataset exists: {raw_path}")
        return
    run(f"python scripts/download_minif2f.py --out {raw_path}", cwd=root)


def patch_cfg(root: Path):
    cfg_path = root / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["input_dim"] = 80
    cfg["device"] = "auto"
    cfg["dataset"]["type"] = "minif2f_like"
    cfg["dataset"]["path"] = "data/minif2f_prepared.jsonl"
    cfg["dataset"]["train_path"] = "data/minif2f_prepared_train.jsonl"
    cfg["dataset"]["val_path"] = "data/minif2f_prepared_val.jsonl"
    cfg["dataset"]["test_path"] = "data/minif2f_prepared_test.jsonl"
    cfg["encoder"]["use_pretrained"] = True
    cfg["encoder"]["checkpoint"] = "outputs/autoencoder/best_autoencoder.pt"
    cfg["encoder"]["freeze"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("config patched for phase1")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="minif2f_raw.jsonl")
    p.add_argument("--quick-seeds", type=int, default=3)
    p.add_argument("--quick-epochs", type=int, default=8)
    p.add_argument("--robust-seeds", type=int, default=20)
    p.add_argument("--robust-epochs", type=int, default=12)
    p.add_argument("--m4-robust-seeds", type=int, default=None)
    p.add_argument("--m4-robust-epochs", type=int, default=None)
    p.add_argument("--run-passk", action="store_true", default=False)
    p.add_argument("--passk-k", type=int, default=32)
    p.add_argument("--passk-limit", type=int, default=200)
    p.add_argument("--skip-real-model", action="store_true", default=False)
    p.add_argument("--real-model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--real-model-limit", type=int, default=2000)
    p.add_argument("--real-model-epochs", type=float, default=1.0)
    p.add_argument("--run-airllm-baseline", action="store_true", default=False)
    p.add_argument("--airllm-model", default="deepseek-ai/DeepSeek-Prover-V2-7B")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_path = root / args.raw

    ensure_raw(root, raw_path)

    run(
        f"python scripts/prepare_minif2f.py --input {raw_path} --output data/minif2f_prepared.jsonl --input-dim 80 --write-split",
        cwd=root,
    )
    run("python scripts/validate_dataset.py --input data/minif2f_prepared.jsonl --input-dim 80", cwd=root)
    run(
        "python scripts/train_autoencoder.py --data data/minif2f_prepared.jsonl --epochs 30 --latent-dim 32 --hidden-dim 128 --out-dir outputs/autoencoder",
        cwd=root,
    )
    real_model_ready = False
    if not args.skip_real_model:
        rc = run(
            f"python scripts/train_real_model.py --data {raw_path} --model {args.real_model} "
            f"--out-dir outputs/real_model --epochs {args.real_model_epochs} --batch-size 1 --grad-accum 16 --limit {args.real_model_limit}",
            cwd=root,
            allow_fail=True,
        )
        if rc == 0 and (root / "outputs/real_model/final").exists():
            real_model_ready = True
        else:
            print("warning: real-model training skipped/fail; continuing Phase I with fallback generators.")
    patch_cfg(root)

    run(
        f"python scripts/run_phase1_roadmap.py --raw {raw_path} --seeds {args.quick_seeds} --epochs {args.quick_epochs}",
        cwd=root,
    )
    run(
        f"python scripts/run_m2.py --raw {raw_path} --seeds {args.robust_seeds} --epochs {args.robust_epochs} --output-dir outputs/phase1/M2_robust",
        cwd=root,
    )
    run(
        f"python scripts/run_m3.py --raw {raw_path} --seeds {args.robust_seeds} --epochs {args.robust_epochs} --output-dir outputs/phase1/M3_robust",
        cwd=root,
    )
    m4_seeds = args.m4_robust_seeds if args.m4_robust_seeds is not None else args.robust_seeds
    m4_epochs = args.m4_robust_epochs if args.m4_robust_epochs is not None else args.robust_epochs
    run(
        f"python scripts/run_m4.py --raw {raw_path} --seeds {m4_seeds} --epochs {m4_epochs} --output-dir outputs/phase1/M4_robust",
        cwd=root,
    )
    run(
        "python scripts/decide_phase1.py --m2 outputs/phase1/M2_robust/summary.json --m3 outputs/phase1/M3_robust/summary.json",
        cwd=root,
    )

    if args.run_passk:
        gen = "hybrid" if real_model_ready else "tactic"
        model_arg = "--model-path outputs/real_model/final" if real_model_ready else ""
        run(
            f"python scripts/run_putnam_passk.py --data {raw_path} --k {args.passk_k} --verifier lean --require-lean "
            f"--generator {gen} {model_arg} --limit {args.passk_limit} --out outputs/putnam_passk_report_lean.json",
            cwd=root,
        )
    if args.run_airllm_baseline:
        run("pip -q install airllm", cwd=root)
        run(
            f"python scripts/run_putnam_passk.py --data {raw_path} --k {args.passk_k} --verifier lean --require-lean "
            f"--generator airllm_hybrid --model-path {args.airllm_model} --limit {args.passk_limit} --out outputs/putnam_passk_airllm_report_lean.json",
            cwd=root,
        )

    report = {
        "phase1_report": read_json(root / "outputs/phase1/phase1_report.json"),
        "m2_robust": read_json(root / "outputs/phase1/M2_robust/summary.json"),
        "m3_robust": read_json(root / "outputs/phase1/M3_robust/summary.json"),
        "m4_robust": read_json(root / "outputs/phase1/M4_robust/summary.json"),
        "decision": read_json(root / "outputs/phase1/decision_phase1.json"),
        "real_model_ready": real_model_ready,
    }
    if args.run_passk and (root / "outputs/putnam_passk_report_lean.json").exists():
        report["passk"] = read_json(root / "outputs/putnam_passk_report_lean.json")
    if args.run_airllm_baseline and (root / "outputs/putnam_passk_airllm_report_lean.json").exists():
        report["passk_airllm_baseline"] = read_json(root / "outputs/putnam_passk_airllm_report_lean.json")

    out = root / "outputs/phase1/kaggle_phase1_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
