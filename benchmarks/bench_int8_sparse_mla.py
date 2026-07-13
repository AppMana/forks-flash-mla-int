#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

from flash_mla import flash_sparse_mla_decode
from flash_mla.int8_sparse_mla import (
    quantize_int8_ds_mla_rows,
    sparse_int8_mla_decode,
    triton_sparse_int8_mla_decode,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_sparse_mla_sm86_shapes import (  # noqa: E402
    HEAD_DIM,
    build_cache as build_fp8_cache,
)


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


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


def build_int8_cache(rows: torch.Tensor, block_size: int):
    nblocks = (rows.shape[0] + block_size - 1) // block_size
    padded = torch.zeros(nblocks * block_size, HEAD_DIM, device=rows.device, dtype=rows.dtype)
    padded[: rows.shape[0]] = rows
    return quantize_int8_ds_mla_rows(padded.view(nblocks, block_size, HEAD_DIM))


def fp32_ref(q, rows, indices, lens, scale, sink):
    out = torch.empty_like(q, dtype=torch.float32)
    for t in range(q.shape[0]):
        k = rows[indices[t, : int(lens[t])].long()].float()
        scores = (q[t].float() @ k.t()) * scale
        m = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink[:, None].float())
        p = torch.exp(scores - m)
        out[t] = (p @ k) / (p.sum(-1, keepdim=True) + torch.exp(sink[:, None].float() - m))
    return out


def run(args):
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    dev = f"cuda:{args.device}"
    scale = 1.0 / math.sqrt(HEAD_DIM)
    print("tokens,topk,impl,us,us_per_token,cos_vs_fp32")

    for tokens in args.tokens:
        for topk in args.topk:
            rows = topk + 64
            fp8_cache, fp8_rows = build_fp8_cache(rows, args.block_size, dev)
            int8_cache, int8_scale = build_int8_cache(fp8_rows[:rows], args.block_size)
            q = torch.randn(tokens, args.heads, HEAD_DIM, device=dev, dtype=torch.bfloat16)
            indices = torch.stack([
                torch.randperm(rows, device=dev)[:topk].to(torch.int32)
                for _ in range(tokens)
            ])
            lens = torch.full((tokens,), topk, device=dev, dtype=torch.int32)
            sink = torch.randn(args.heads, device=dev, dtype=torch.float32) * 0.1

            ref = fp32_ref(q, fp8_rows, indices, lens, scale, sink)
            out_fp8 = flash_sparse_mla_decode(q, fp8_cache, indices, lens, scale=scale, attn_sink=sink)
            out_i8 = triton_sparse_int8_mla_decode(q, int8_cache, int8_scale, indices, lens, scale=scale, attn_sink=sink)
            out_i8_native = sparse_int8_mla_decode(q, int8_cache, int8_scale, indices, lens, scale=scale, attn_sink=sink)

            rows_to_print = [
                (
                    "fp8_flashmla",
                    time_us(lambda: flash_sparse_mla_decode(q, fp8_cache, indices, lens, scale=scale, attn_sink=sink),
                            args.warmup, args.iters),
                    cos_diff(out_fp8.float(), ref),
                ),
                (
                    "int8_triton_imma",
                    time_us(lambda: triton_sparse_int8_mla_decode(q, int8_cache, int8_scale, indices, lens, scale=scale, attn_sink=sink),
                            args.warmup, args.iters),
                    cos_diff(out_i8.float(), ref),
                ),
                (
                    "int8_flashmla",
                    time_us(lambda: sparse_int8_mla_decode(q, int8_cache, int8_scale, indices, lens, scale=scale, attn_sink=sink),
                            args.warmup, args.iters),
                    cos_diff(out_i8_native.float(), ref),
                ),
            ]
            for name, us, cd in rows_to_print:
                print(f"{tokens},{topk},{name},{us:.3f},{us / max(tokens, 1):.3f},{cd:.3e}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 16, 256])
    p.add_argument("--topk", type=int, nargs="+", default=[128, 512])
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    run(p.parse_args())


if __name__ == "__main__":
    main()
