"""Wall-clock measurement with warmup and an optional device-sync hook.

GPU work is asynchronous: without synchronizing before reading the clock you time the kernel *launch*, not the
kernel. Pass ``sync=torch.cuda.synchronize`` (or equivalent) when measuring device code.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def measure(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    repeats: int = 10,
    sync: Callable[[], None] | None = None,
) -> list[float]:
    """Run ``fn`` ``warmup`` times (discarded), then ``repeats`` timed times. Returns latencies in ms."""
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    _sync = sync or (lambda: None)

    for _ in range(warmup):
        fn()
    _sync()

    out: list[float] = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter_ns()
        fn()
        _sync()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out
