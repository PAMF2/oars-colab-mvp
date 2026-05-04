import csv
import subprocess
import sys
from pathlib import Path


def test_smoke_ablation_generates_csv(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    out_csv = tmp_path / "summary.csv"
    cmd = [
        sys.executable,
        "scripts/run_ablation.py",
        "--epochs", "1",
        "--samples", "200",
        "--seeds", "1",
        "--output", str(out_csv),
    ]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
