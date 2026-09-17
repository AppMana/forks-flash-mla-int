#!/usr/bin/env python3
"""Compute-Sanitizer reproducer for the DSV4 INT8 sparse-MLA cache on GB10.

This deliberately uses the production cache geometry rather than the small
sm86-only parity-test geometry:

* 528-byte inline token rows (512 int8 bytes, fp32 scale, 12 bytes padding)
* SWA width 128
* compressed-cache widths 384 and 512
* both C4 (block size 64) and C128 (block size 2) cache views
* indices at the final valid physical slot
* T=6 decode, matching one target plus five DSpark speculative positions

Run inside the serving image:

  CUDA_LAUNCH_BLOCKING=1 compute-sanitizer --tool memcheck \
    python3 sm121_int8_cache_memcheck.py
"""

from __future__ import annotations

import math

import torch

from flash_mla.int8_sparse_mla import (
    sparse_mla_decode_int8,
    sparse_mla_prefill_int8,
)


HEAD_DIM = 512
TOKEN_BYTES = 528
HEADS = 64
SWA_TOPK = 128
NUM_SLOTS = 32_770


def inline_cache(num_slots: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = (num_slots + block_size - 1) // block_size
    raw = torch.zeros(
        num_blocks,
        block_size,
        TOKEN_BYTES,
        dtype=torch.uint8,
        device="cuda",
    )
    data = raw[:, :, :HEAD_DIM].view(torch.int8)
    data.random_(-127, 128)
    scales = (
        raw[:, :, HEAD_DIM : HEAD_DIM + 4]
        .view(torch.float32)
        .squeeze(-1)
    )
    scales.fill_(1.0 / 127.0)
    assert data.stride(1) == TOKEN_BYTES
    assert scales.stride(1) == TOKEN_BYTES // 4
    return data, scales


def end_indices(tokens: int, width: int, num_slots: int) -> torch.Tensor:
    base = torch.arange(width, dtype=torch.int32, device="cuda")
    indices = (num_slots - 1 - base).clamp_min(0).repeat(tokens, 1)
    indices[:, 0] = num_slots - 1
    indices[:, -1] = 0
    return indices


def exercise(extra_block_size: int, extra_topk: int) -> None:
    swa, swa_scales = inline_cache(NUM_SLOTS, 256)
    extra, extra_scales = inline_cache(NUM_SLOTS, extra_block_size)
    sink = torch.randn(HEADS, dtype=torch.float32, device="cuda") * 0.1
    scale = 1.0 / math.sqrt(HEAD_DIM)

    decode_tokens = 6
    q_decode = torch.randn(
        decode_tokens,
        HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    swa_indices = end_indices(decode_tokens, SWA_TOPK, NUM_SLOTS)
    extra_indices = end_indices(decode_tokens, extra_topk, NUM_SLOTS)
    swa_lens = torch.full(
        (decode_tokens,), SWA_TOPK, dtype=torch.int32, device="cuda"
    )
    extra_lens = torch.full(
        (decode_tokens,), extra_topk, dtype=torch.int32, device="cuda"
    )
    decoded = sparse_mla_decode_int8(
        q_decode,
        swa,
        swa_scales,
        swa_indices,
        swa_lens,
        scale=scale,
        attn_sink=sink,
        extra_cache=extra,
        extra_scale=extra_scales,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(decoded).all()

    prefill_tokens = 67
    q_prefill = torch.randn(
        prefill_tokens,
        HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    prefill_swa_indices = end_indices(prefill_tokens, SWA_TOPK, NUM_SLOTS)
    prefill_extra_indices = end_indices(prefill_tokens, extra_topk, NUM_SLOTS)
    prefill_swa_lens = torch.full(
        (prefill_tokens,), SWA_TOPK, dtype=torch.int32, device="cuda"
    )
    prefill_extra_lens = torch.full(
        (prefill_tokens,), extra_topk, dtype=torch.int32, device="cuda"
    )
    prefilled = sparse_mla_prefill_int8(
        q_prefill,
        swa,
        swa_scales,
        prefill_swa_indices,
        prefill_swa_lens,
        scale=scale,
        attn_sink=sink,
        extra_cache=extra,
        extra_scale=extra_scales,
        extra_indices=prefill_extra_indices,
        extra_lens=prefill_extra_lens,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(prefilled).all()
    print(
        f"PASS sm={torch.cuda.get_device_capability()} "
        f"extra_block_size={extra_block_size} extra_topk={extra_topk}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise SystemExit(f"expected GB10 sm121, got {capability}")
    for block_size in (64, 2):
        for extra_topk in (384, 512):
            exercise(block_size, extra_topk)


if __name__ == "__main__":
    main()
