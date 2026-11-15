# quantbench

**Which quantization should you actually serve on a 16 GB consumer GPU?** quantbench measures FP16, INT8,
INT4 (AWQ / GPTQ / GGUF), FP8 and NVFP4 across HF Transformers, vLLM and llama.cpp. It reports latency,
throughput, peak memory **and quality loss** in one table, with raw numbers and the exact environment for every run.

## Results

> No results yet. The first FP16 baseline lands in week 1. Every number here will link to a JSON file in
> `benchmarks/results/` with raw samples and the hardware/software environment it was measured on.

| Model | Backend | Format | TTFT p50 (ms) | Decode tok/s (bs=1) | Peak VRAM (GiB) | ΔPPL vs FP16 |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

**Reference hardware:** all numbers are measured on my personal workstation: NVIDIA RTX 5070 Ti 16 GB
(Blackwell, sm_120), CUDA 12.8. No cloud or employer hardware is used.

## Install

```bash
git clone https://github.com/ramkumar-mv/quantbench && cd quantbench
uv sync            # dependencies are pinned in uv.lock
uv run pytest -q
```

## Methodology
- Warmup runs are discarded; GPU timings synchronize the device before reading the clock.
- Reported: n, mean, sample std, min, p50/p90/p99, max. Raw samples are always saved.
- Quality is measured alongside speed (perplexity + task evals, including Indic languages).

See [`benchmarks/README.md`](benchmarks/README.md) for the result file format.

## Roadmap
- [x] Metrics core: latency stats, sync-aware timer, environment capture
- [ ] HF Transformers runner: FP16 / INT8 / INT4 (AWQ, GPTQ)
- [ ] vLLM runner: FP16 / AWQ / GPTQ / FP8, batch sweep
- [ ] llama.cpp runner: GGUF Q8_0 / Q4_K_M
- [ ] NVFP4 on Blackwell

## License
Apache-2.0
