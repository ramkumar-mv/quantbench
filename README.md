# quantbench

**Which quantization should you actually serve on a 16 GB consumer GPU?** quantbench measures FP16, INT8,
INT4 (AWQ / GPTQ / GGUF), FP8 and NVFP4 across HF Transformers, vLLM and llama.cpp. It reports latency,
throughput, peak memory **and quality loss** in one table, with raw numbers and the exact environment for every run.

## Results

Baseline: Qwen2.5-1.5B-Instruct, HF Transformers `generate()` (greedy), BF16, prompt 512 tokens, 128 new tokens,
5 timed repeats after warmup. Raw samples + environment: [`benchmarks/results/`](benchmarks/results/).

| Model | Backend | Format | Batch | TTFT p50 (ms) | Decode tok/s p50 (min–max) | Peak VRAM (GiB) | ΔPPL vs BF16 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | HF Transformers | BF16 | 1 | 33.5 | 43.3 (38.8–44.3) | 3.04 | 0 (ref) |
| Qwen2.5-1.5B | HF Transformers | BF16 | 4 | 119.7 | 171.9 (165.0–176.3) | 3.52 | 0 (ref) |
| Qwen2.5-1.5B | HF Transformers | BF16 | 16 | 490.4 | 591.8 (576.8–601.4) | 5.44 | 0 (ref) |

**What this shows:** at batch 1, HF eager decode reaches **~15% of the bandwidth ceiling** (~290 tok/s ideal for
3.1 GB of weights at 896 GB/s). Throughput scales almost linearly with batch, so decode here is **overhead-bound**
(Python + kernel launches), not memory-bound. Prefill at batch 16 achieves ~51 TFLOPS. Next: torch.compile / CUDA
graphs, vLLM, then quantized formats.

Peak VRAM = `torch.cuda.max_memory_allocated()` (excludes the CUDA context). Decode tok/s = batch × 127 decode steps ÷
(generate time − prefill p50).

**Reference hardware:** all numbers are measured on my personal workstation: NVIDIA RTX 5070 Ti 16 GB
(Blackwell, sm_120), CUDA 12.8. No cloud or employer hardware is used.

## Install

```bash
git clone https://github.com/ramkumar-mv/quantbench && cd quantbench
uv sync            # dependencies are pinned in uv.lock
uv run pytest -q
```

Reproduce the baseline (needs an NVIDIA GPU; installs PyTorch 2.8.0 + CUDA 12.8):

```bash
uv run --extra hf python benchmarks/run_hf.py --model Qwen/Qwen2.5-1.5B-Instruct --batches 1 4 16
```

## Methodology
- Warmup runs are discarded; GPU timings synchronize the device before reading the clock.
- Reported: n, mean, sample std, min, p50/p90/p99, max. Raw samples are always saved.
- Quality is measured alongside speed (perplexity + task evals).

See [`benchmarks/README.md`](benchmarks/README.md) for the result file format.

## Roadmap
- [x] Metrics core: latency stats, sync-aware timer, environment capture
- [x] HF Transformers runner: BF16 baseline (TTFT, decode tok/s, peak VRAM, batch sweep)
- [ ] HF Transformers: INT8 / INT4 (AWQ, GPTQ) + perplexity
- [ ] vLLM runner: FP16 / AWQ / GPTQ / FP8, batch sweep
- [ ] llama.cpp runner: GGUF Q8_0 / Q4_K_M
- [ ] NVFP4 on Blackwell

## License
Apache-2.0
