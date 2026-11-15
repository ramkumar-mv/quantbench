# benchmarks

Each runner writes one JSON file per run to `benchmarks/results/` containing:

- `config`: model, backend, quantization format, batch size, prompt/gen lengths, warmup, repeats
- `env`: output of `quantbench.capture_env()` (GPU, driver/CUDA, package versions)
- `raw`: every individual latency sample in ms (not just aggregates)
- `summary`: `quantbench.summarize()` of the raw samples

Results tables in the top-level README are generated from these files. No numbers are typed by hand.
