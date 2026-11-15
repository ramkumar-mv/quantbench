"""Latency statistics. Pure Python so it is testable without a GPU."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


def percentile(samples: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation between closest ranks (numpy's default 'linear' method).

    ``q`` is in [0, 100].
    """
    if not samples:
        raise ValueError("percentile() requires at least one sample")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    xs = sorted(samples)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class LatencySummary:
    n: int
    mean: float
    std: float  # sample std (ddof=1); 0.0 when n == 1
    min: float
    p50: float
    p90: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize(samples: Sequence[float]) -> LatencySummary:
    """Summarize latency samples (any unit; the caller keeps units consistent)."""
    if not samples:
        raise ValueError("summarize() requires at least one sample")
    n = len(samples)
    mean = math.fsum(samples) / n
    var = math.fsum((x - mean) ** 2 for x in samples) / (n - 1) if n > 1 else 0.0
    return LatencySummary(
        n=n,
        mean=mean,
        std=math.sqrt(var),
        min=float(min(samples)),
        p50=percentile(samples, 50),
        p90=percentile(samples, 90),
        p99=percentile(samples, 99),
        max=float(max(samples)),
    )
