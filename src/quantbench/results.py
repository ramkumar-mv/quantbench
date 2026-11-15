"""Result records: one JSON file per benchmark run, with raw samples and environment."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quantbench.env import capture_env
from quantbench.stats import summarize


def decode_tokens_per_s(total_ms: float, ttft_ms: float, new_tokens: int, batch: int) -> float:
    """Aggregate decode throughput from an end-to-end generate() timing.

    generate() = prefill (produces the first token) + (new_tokens - 1) decode steps. Subtracting the
    separately measured prefill time isolates the decode phase.
    """
    if new_tokens < 2:
        raise ValueError("need at least 2 new tokens to measure decode")
    decode_ms = total_ms - ttft_ms
    if decode_ms <= 0:
        raise ValueError(f"decode time must be positive (total={total_ms}, ttft={ttft_ms})")
    return batch * (new_tokens - 1) / (decode_ms / 1000.0)


def build_record(
    config: dict[str, Any],
    raw: dict[str, Sequence[float]],
    derived: dict[str, Any] | None = None,
    env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "quantbench.result/v1",
        "created_unix": int(time.time()),
        "config": config,
        "env": env if env is not None else capture_env(),
        "raw": {k: list(v) for k, v in raw.items()},
        "summary": {k: summarize(v).to_dict() for k, v in raw.items() if len(v) > 0},
        "derived": derived or {},
    }


def write_record(record: dict[str, Any], out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
