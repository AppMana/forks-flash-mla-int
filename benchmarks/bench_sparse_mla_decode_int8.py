#!/usr/bin/env python3
"""Benchmark native INT8 sparse-MLA decode at serving geometries.

The benchmark uses the packed 528-byte cache layout consumed by vLLM and
times the complete public native operation, including its temporary workspace
allocations.  It reports repeated-sample median and p95 CUDA-event latency so
kernel changes can be compared without relying on a single timing sample.

Run:
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_sparse_mla_decode_int8.py
"""
from __future__ import annotations

import argparse
import math
import os
import statistics

import torch

HEAD_DIM = 512
TOKEN_BYTES = 528
SWA_TOPK = 128


def build_inline_cache(num_slots: int, block_size: int, device: str, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    num_blocks = (num_slots + block_size - 1) // block_size
    rows = (
        torch.randn(
            num_blocks * block_size,
            HEAD_DIM,
            generator=generator,
            device=device,
        )
        * 2.0
    ).to(torch.bfloat16)
    scales = (rows.float().abs().amax(dim=-1) / 127.0).clamp_min(1.0e-12)
    quantized = (
        torch.round(rows.float() / scales.unsqueeze(-1))
        .clamp(-127, 127)
        .to(torch.int8)
    )
    packed = torch.zeros(
        num_blocks,
        block_size,
        TOKEN_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    packed[:, :, :HEAD_DIM] = quantized.view(
        num_blocks, block_size, HEAD_DIM
    ).view(torch.uint8)
    packed[:, :, HEAD_DIM : HEAD_DIM + 4] = scales.view(
        num_blocks, block_size, 1
    ).view(torch.uint8)
    cache = packed[:, :, :HEAD_DIM].view(torch.int8)
    cache_scale = packed[:, :, HEAD_DIM : HEAD_DIM + 4].view(torch.float32).squeeze(-1)
    return cache, cache_scale


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def timed_sample(operation, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--context", type=int, default=16_384)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--widths", type=int, nargs="+", default=[640, 1024, 4224])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--slots-per-split", type=int, default=96)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}"
    os.environ["FLASH_MLA_SLOTS_PER_SPLIT"] = str(args.slots_per_split)

    from flash_mla import sparse_mla_decode_int8

    max_extra = max(args.widths) - SWA_TOPK
    if max_extra <= 0:
        raise ValueError(f"all widths must be greater than SWA_TOPK={SWA_TOPK}")
    swa_cache, swa_scale = build_inline_cache(args.context, 256, device, args.seed)
    extra_cache, extra_scale = build_inline_cache(
        args.context, 2, device, args.seed + 1
    )
    sink = torch.randn(64, dtype=torch.float32, device=device) * 0.1
    scale = 1.0 / math.sqrt(HEAD_DIM)

    print("tokens,width,splits,median_us,p95_us,query_rows_per_s", flush=True)
    sm_count = torch.cuda.get_device_properties(args.device).multi_processor_count
    for tokens in args.tokens:
        for width in args.widths:
            extra_topk = width - SWA_TOPK
            q = torch.randn(
                tokens, 64, HEAD_DIM, dtype=torch.bfloat16, device=device
            )
            swa_indices = torch.randint(
                0,
                args.context,
                (tokens, SWA_TOPK),
                dtype=torch.int32,
                device=device,
            )
            extra_indices = torch.randint(
                0,
                args.context,
                (tokens, extra_topk),
                dtype=torch.int32,
                device=device,
            )
            swa_lens = torch.full(
                (tokens,), SWA_TOPK, dtype=torch.int32, device=device
            )
            extra_lens = torch.full(
                (tokens,), extra_topk, dtype=torch.int32, device=device
            )

            def operation():
                return sparse_mla_decode_int8(
                    q,
                    swa_cache,
                    swa_scale,
                    swa_indices,
                    swa_lens,
                    scale=scale,
                    attn_sink=sink,
                    extra_cache=extra_cache,
                    extra_scale=extra_scale,
                    extra_indices=extra_indices,
                    extra_lens=extra_lens,
                )

            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize()
            samples = [
                timed_sample(operation, args.iterations)
                for _ in range(args.samples)
            ]
            median_us = statistics.median(samples)
            p95_us = percentile(samples, 0.95)
            rows_per_second = tokens * 1.0e6 / median_us
            occupancy_splits = max(1, sm_count // (tokens * math.ceil(64 / 32)))
            slot_cap = math.ceil(width / args.slots_per_split)
            splits = min(64, occupancy_splits, slot_cap)
            print(
                f"{tokens},{width},{splits},{median_us:.2f},"
                f"{p95_us:.2f},{rows_per_second:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
