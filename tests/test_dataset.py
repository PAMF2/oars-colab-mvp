import json
from pathlib import Path

from oars_mvp.dataset import prepare_minif2f_rows, read_jsonl, write_jsonl


def test_prepare_rows_shape_and_values(tmp_path: Path):
    rows = [
        {"formal_statement": "theorem t1 : a + b = b + a", "formal_proof": "by simpa", "label": 1},
        {"statement": "theorem t2 : False", "proof": "", "label": 0},
    ]
    out = prepare_minif2f_rows(rows, input_dim=24, num_blocks=4)
    assert len(out) == 2
    assert len(out[0]["features"]) == 24
    assert out[0]["label"] in (0.0, 1.0)
    assert 0 <= out[0]["block_id"] < 4

    p = tmp_path / "prepared.jsonl"
    write_jsonl(str(p), out)
    loaded = read_jsonl(str(p))
    assert len(loaded) == 2


def test_sample_raw_file_exists_and_is_readable():
    p = Path("data/minif2f_raw_sample.jsonl")
    assert p.exists()
    rows = read_jsonl(str(p))
    assert len(rows) >= 3
    assert isinstance(rows[0], dict)
