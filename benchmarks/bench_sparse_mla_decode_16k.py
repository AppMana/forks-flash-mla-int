#!/usr/bin/env python3
"""TRUE-footprint sparse-MLA decode benchmark (sm_86).

The historical decode numbers (49.4 us @ width 512, 74.4 us @ width 1024) were
measured on the topk+64-slot microbench cache from bench_sparse_mla_sm86_shapes,
which is L2-resident (192 slots ~ 112 KB). This harness builds REAL >=16k-slot
fp8_ds_mla caches for both streams (swa block 256, extra block 2 -- the c128a
serving layout; --extra-block 64 for c4a), T=1 decode, width = swa_topk +
extra_topk in {512, 1024}, CUDA-event timing.

Also checks parity (cos vs fp32 oracle over the expected bf16 rows) on every
run so a perf number is never reported for a wrong kernel.

Run:
  CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_sparse_mla_decode_16k.py
"""
from __future__ import annotations

import argparse
import math

import torch

FP8_DIM = 448
ROPE_DIM = 64
SCALE_DIM = 8
TOKEN_DATA_SIZE = FP8_DIM + ROPE_DIM * 2  # 576
SCALE_GROUP = 64
HEAD_DIM = 512
H = 64


def build_cache_fast(num_slots: int, block_size: int, device: str, seed: int):
    """Vectorized fp8_ds_mla cache builder. Returns (cache uint8 [nb,bs,584], K bf16 [nb*bs,512])."""
    g = torch.Generator(device=device).manual_seed(seed)
    nb = (num_slots + block_size - 1) // block_size
    total = nb * block_size
    # random per-group UE8M0 exponents in [-2, 2]; values scaled to match so fp8 quant is benign
    exps = torch.randint(-2, 3, (total, FP8_DIM // SCALE_GROUP), generator=g, device=device)
    scales = torch.exp2(exps.float())                            # [total, 7]
    scale_per_dim = scales.repeat_interleave(SCALE_GROUP, dim=1)  # [total, 448]
    nope_raw = torch.randn(total, FP8_DIM, generator=g, device=device) * scale_per_dim
    fp8 = (nope_raw / scale_per_dim).to(torch.float8_e4m3fn)
    expected_nope = fp8.float() * scale_per_dim
    rope = torch.randn(total, ROPE_DIM, generator=g, device=device).to(torch.bfloat16)

    data = torch.empty(total, TOKEN_DATA_SIZE, dtype=torch.uint8, device=device)
    data[:, :FP8_DIM] = fp8.view(torch.uint8)
    data[:, FP8_DIM:] = rope.view(torch.uint8).view(total, ROPE_DIM * 2)
    sc = torch.full((total, SCALE_DIM), 127, dtype=torch.uint8, device=device)
    sc[:, : FP8_DIM // SCALE_GROUP] = (exps + 127).to(torch.uint8)

    cache = torch.empty(nb, block_size * (TOKEN_DATA_SIZE + SCALE_DIM), dtype=torch.uint8, device=device)
    cache[:, : block_size * TOKEN_DATA_SIZE] = data.view(nb, block_size * TOKEN_DATA_SIZE)
    cache[:, block_size * TOKEN_DATA_SIZE :] = sc.view(nb, block_size * SCALE_DIM)
    cache = cache.view(nb, block_size, TOKEN_DATA_SIZE + SCALE_DIM)
    K = torch.cat([expected_nope, rope.float()], dim=1).to(torch.bfloat16)
    return cache, K


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def fp32_ref(q, swa_K, swa_idx, swa_lens, extra_K, extra_idx, extra_lens, scale, sink):
    T = q.shape[0]
    out = torch.zeros_like(q, dtype=torch.float32)
    for t in range(T):
        ns, ne = int(swa_lens[t]), int(extra_lens[t])
        K = torch.cat([swa_K[swa_idx[t, :ns].long()], extra_K[extra_idx[t, :ne].long()]]).float()
        sc = (q[t].float() @ K.t()) * scale
        sk = sink[:, None].float()
        m = torch.maximum(sc.max(-1, keepdim=True).values, sk)
        ex = torch.exp(sc - m)
        out[t] = (ex @ K) / (ex.sum(-1, keepdim=True) + torch.exp(sk - m))
    return out


def time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--context", type=int, default=16_384)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--widths", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--extra-block", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--triton", action="store_true", help="also time production Triton decode")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--profile-loop", type=int, default=0,
                    help="just run the native decode N times at the first width (for ncu) and exit")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}"
    scale = 1.0 / math.sqrt(HEAD_DIM)

    from flash_mla import sparse_mla_decode_fp8

    swa_cache, swa_K = build_cache_fast(args.context, 256, device, args.seed)
    extra_cache, extra_K = build_cache_fast(args.context, args.extra_block, device, args.seed + 1)
    swa_bytes = swa_cache.numel()
    extra_bytes = extra_cache.numel()
    print(
        f"# context={args.context} slots/stream; swa cache {swa_bytes/1e6:.1f} MB, "
        f"extra cache {extra_bytes/1e6:.1f} MB (extra_block={args.extra_block}); "
        f"T={args.tokens} H={H} D={HEAD_DIM}",
        flush=True,
    )
    sink = torch.randn(H, device=device, dtype=torch.float32) * 0.1

    print("width,swa_topk,extra_topk,impl,us,cos_vs_fp32", flush=True)
    for width in args.widths:
        swa_topk = width // 2
        extra_topk = width - swa_topk
        q = torch.randn(args.tokens, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
        swa_idx = torch.randint(0, args.context, (args.tokens, swa_topk), dtype=torch.int32, device=device)
        extra_idx = torch.randint(0, args.context, (args.tokens, extra_topk), dtype=torch.int32, device=device)
        swa_lens = torch.full((args.tokens,), swa_topk, dtype=torch.int32, device=device)
        extra_lens = torch.full((args.tokens,), extra_topk, dtype=torch.int32, device=device)

        def run_native():
            return sparse_mla_decode_fp8(
                q, swa_cache, swa_idx, swa_lens, scale=scale, attn_sink=sink,
                extra_cache=extra_cache, extra_indices=extra_idx, extra_lens=extra_lens,
            )

        if args.profile_loop:
            for _ in range(args.profile_loop):
                run_native()
            torch.cuda.synchronize()
            return

        cd = float("nan")
        if not args.no_check:
            out = run_native()
            ref = fp32_ref(q, swa_K, swa_idx, swa_lens, extra_K, extra_idx, extra_lens, scale, sink)
            cd = cos_diff(out.float(), ref)
            assert cd < 8e-5, f"native decode parity FAILED at width={width}: cos={cd:.2e}"
            del out, ref
            # the oracle's big fp32 temporaries fragment the caching allocator; a
            # polluted cache slows the PER-OP host allocations (out/oaccum/scratch)
            # and can inflate launch gaps by ~10 us at this op size
            torch.cuda.empty_cache()

        us = time_us(run_native, args.warmup, args.iters)
        print(f"{width},{swa_topk},{extra_topk},native,{us:.2f},{cd:.2e}", flush=True)

        if args.triton:
            from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
                decode_sparse_attention_triton,
            )
            out_t = torch.empty_like(q)

            def run_triton():
                decode_sparse_attention_triton(
                    q=q, swa_cache=swa_cache, swa_indices=swa_idx, swa_lens=swa_lens,
                    scale=scale, attn_sink=sink, out=out_t,
                    extra_cache=extra_cache, extra_indices=extra_idx, extra_lens=extra_lens,
                )

            cd_t = float("nan")
            if not args.no_check:
                run_triton()
                ref = fp32_ref(q, swa_K, swa_idx, swa_lens, extra_K, extra_idx, extra_lens, scale, sink)
                cd_t = cos_diff(out_t.float(), ref)
            us_t = time_us(run_triton, args.warmup, args.iters)
            print(f"{width},{swa_topk},{extra_topk},triton,{us_t:.2f},{cd_t:.2e}", flush=True)


if __name__ == "__main__":
    main()
