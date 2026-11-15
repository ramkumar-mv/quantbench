"""Capture the environment a result was produced in. Every result file embeds this."""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from typing import Any

_TRACKED = ("torch", "transformers", "vllm", "bitsandbytes", "autoawq", "gptqmodel", "llama_cpp_python")


def _pkg_versions() -> dict[str, str]:
    out = {}
    for name in _TRACKED:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    return out


def _gpu_info() -> dict[str, Any] | None:
    try:
        import torch  # optional dependency
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_mib": props.total_memory // (1024 * 1024),
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_runtime": torch.version.cuda,
    }


def capture_env() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _pkg_versions(),
        "gpu": _gpu_info(),
    }
