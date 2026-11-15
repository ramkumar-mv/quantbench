"""HF Transformers baseline: TTFT, decode throughput and peak VRAM for one model/precision.

Usage:
    uv run --extra hf python benchmarks/run_hf.py --model Qwen/Qwen2.5-1.5B-Instruct --batches 1 4 16

Writes one JSON per batch size to benchmarks/results/ (raw samples + environment).
"""

from __future__ import annotations

import argparse
import re
from functools import partial

import torch
from transformers import AutoModelForCausalLM

from quantbench.results import build_record, decode_tokens_per_s, write_record
from quantbench.stats import summarize
from quantbench.timer import measure

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype]).to(dev).eval()
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    vocab = model.config.vocab_size
    sync = torch.cuda.synchronize

    for batch in args.batches:
        g = torch.Generator().manual_seed(0)  # fixed synthetic prompt: timing doesn't depend on content
        ids = torch.randint(100, min(vocab, 30000), (batch, args.prompt_len), generator=g).to(dev)
        mask = torch.ones_like(ids)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            prefill = partial(model, input_ids=ids, attention_mask=mask, use_cache=True)
            generate = partial(model.generate, input_ids=ids, attention_mask=mask, do_sample=False,
                               max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, pad_token_id=0)
            ttft = measure(prefill, warmup=args.warmup, repeats=args.repeats, sync=sync)
            gen = measure(generate, warmup=1, repeats=args.repeats, sync=sync)
        peak = torch.cuda.max_memory_allocated()

        ttft_p50 = summarize(ttft).p50
        decode = [decode_tokens_per_s(t, ttft_p50, args.new_tokens, batch) for t in gen]
        config = {"backend": "hf-transformers", "model": args.model, "format": args.dtype, "batch": batch,
                  "prompt_len": args.prompt_len, "new_tokens": args.new_tokens, "warmup": args.warmup,
                  "repeats": args.repeats, "decode": "greedy generate(), min_new_tokens=max_new_tokens"}
        derived = {"weight_bytes": weight_bytes, "peak_vram_bytes": peak,
                   "decode_tok_s_p50": summarize(decode).p50, "ttft_ms_p50": ttft_p50}
        rec = build_record(config, {"ttft_ms": ttft, "generate_ms": gen, "decode_tok_s": decode}, derived)
        name = f"hf_{re.sub(r'[^A-Za-z0-9.]+', '-', args.model)}_{args.dtype}_b{batch}"
        path = write_record(rec, args.out, name)
        tps = derived["decode_tok_s_p50"]
        print(f"batch={batch:>3}  TTFT p50={ttft_p50:7.1f} ms  decode={tps:8.1f} tok/s"
              f"  peak VRAM={peak / 2**30:5.2f} GiB  -> {path}")


if __name__ == "__main__":
    main()
