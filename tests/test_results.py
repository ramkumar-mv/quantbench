import json
import math

import pytest

from quantbench.results import build_record, decode_tokens_per_s, write_record


def test_decode_tokens_per_s_subtracts_prefill():
    # 129 new tokens -> 128 decode steps in (1100 - 100) ms = 1 s -> 128 tok/s per sequence
    assert math.isclose(decode_tokens_per_s(1100.0, 100.0, 129, 1), 128.0)
    assert math.isclose(decode_tokens_per_s(1100.0, 100.0, 129, 4), 512.0)


@pytest.mark.parametrize("args", [(100.0, 100.0, 10, 1), (100.0, 50.0, 1, 1)])
def test_decode_tokens_per_s_rejects_bad_input(args):
    with pytest.raises(ValueError):
        decode_tokens_per_s(*args)


def test_build_and_write_record(tmp_path):
    rec = build_record({"model": "m"}, {"ttft_ms": [1.0, 2.0, 3.0], "empty": []}, {"x": 1}, env={"gpu": None})
    assert rec["schema"] == "quantbench.result/v1"
    assert rec["summary"]["ttft_ms"]["p50"] == 2.0
    assert "empty" not in rec["summary"]
    path = write_record(rec, tmp_path / "out", "run1")
    assert json.loads(path.read_text(encoding="utf-8"))["config"] == {"model": "m"}
