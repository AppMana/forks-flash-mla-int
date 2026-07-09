#!/usr/bin/env python3
"""Stage-split timing for the native sparse-MLA prefill paths (sm_86).

Times each CUDA kernel of the staged prefill separately (gather vs attention)
via CUDA-event bracketing of the single fused op with env-selected variants,
and prints a per-kernel breakdown from torch profiler. Production 16k shape.

  CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_prefill_stages.py --swa-topk 256 --extra-topk 256
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from bench_sparse_mla_16k_matrix import make_indices  # noqa: E402
from bench_sparse_mla_sm86_shapes import HEAD_DIM, H, build_cache, time_us  # noqa: E402
from flash_mla import flash_sparse_mla_prefill  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--context", type=int, default=16_384)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--swa-topk", type=int, default=256)
    ap.add_argument("--extra-topk", type=int, default=256)
    ap.add_argument("--extra-block", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--staged", action="store_true", default=True)
    ap.add_argument("--fused", dest="staged", action="store_false")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)
    device = f"cuda:{args.device}"
    scale = 1.0 / math.sqrt(HEAD_DIM)

    swa_cache, _ = build_cache(args.context, 256, device)
    extra_cache, _ = build_cache(args.context, args.extra_block, device)
    q = torch.randn(args.tokens, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    swa_idx = make_indices(args.tokens, args.swa_topk, args.context, device)
    extra_idx = make_indices(args.tokens, args.extra_topk, args.context, device)
    swa_lens = torch.full((args.tokens,), args.swa_topk, dtype=torch.int32, device=device)
    extra_lens = torch.full((args.tokens,), args.extra_topk, dtype=torch.int32, device=device)
    sink = torch.randn(H, device=device, dtype=torch.float32) * 0.1

    def run():
        return flash_sparse_mla_prefill(
            q,
            swa_cache,
            swa_idx.unsqueeze(1),
            swa_lens,
            scale=scale,
            attn_sink=sink,
            extra_cache=extra_cache,
            extra_indices=extra_idx,
            extra_lens=extra_lens,
            use_staged_prefill=args.staged,
        )

    total = time_us(run, args.warmup, args.iters)
    print(
        f"total: {total:.1f} us  ({total / args.tokens:.3f} us/tok)  "
        f"tokens={args.tokens} width={args.swa_topk + args.extra_topk} staged={args.staged}"
    )

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            run()
        torch.cuda.synchronize()
    rows = sorted(
        prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True
    )
    for e in rows[:8]:
        if e.self_device_time_total <= 0:
            continue
        print(
            f"{e.key[:80]:80s} calls={e.count:4d} "
            f"self_gpu={e.self_device_time_total / args.iters:9.1f} us/iter"
        )


if __name__ == "__main__":
    main()
