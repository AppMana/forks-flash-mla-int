#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from bench_sparse_mla_sm86_shapes import (  # noqa: E402
    HEAD_DIM,
    H,
    build_cache,
    time_us,
)
from flash_mla import sparse_mla_decode_fp8, sparse_mla_prefill
from flash_mla.int8_sparse_mla import (
    quantize_int8_ds_mla_rows,
    sparse_mla_decode_int8_triton,
    sparse_mla_prefill_int8,
)
from vllm.triton_utils import LOG2E, triton
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_global_slots_k_cache,
)
from vllm.models.deepseek_v4.nvidia_sm86 import triton_kernels as tk
from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
    decode_sparse_attention_triton,
)


def make_indices(tokens: int, topk: int, num_slots: int, device: str) -> torch.Tensor:
    return torch.randint(0, num_slots, (tokens, topk), dtype=torch.int32, device=device)


def build_int8_cache(rows: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    pad = (-rows.shape[0]) % block_size
    if pad:
        rows = torch.cat([rows, torch.zeros(pad, HEAD_DIM, dtype=rows.dtype, device=rows.device)])
    return quantize_int8_ds_mla_rows(rows.view(-1, block_size, HEAD_DIM))


def build_gathered(case, swa_idx, extra_idx):
    tokens = swa_idx.shape[0]
    width = swa_idx.shape[1] + extra_idx.shape[1]
    kv = torch.empty(tokens * width, 1, HEAD_DIM, dtype=torch.bfloat16, device=swa_idx.device)
    indices = torch.empty(tokens, 1, width, dtype=torch.int32, device=swa_idx.device)
    lengths = torch.full((tokens,), width, dtype=torch.int32, device=swa_idx.device)
    for start in range(0, tokens, 256):
        end = min(start + 256, tokens)
        rows = torch.cat(
            [
                case["swa_rows"][swa_idx[start:end].long()],
                case["extra_rows"][extra_idx[start:end].long()],
            ],
            dim=1,
        )
        flat_start = start * width
        flat_end = end * width
        kv[flat_start:flat_end, 0, :] = rows.reshape(-1, HEAD_DIM)
        indices[start:end, 0, :] = torch.arange(
            flat_start, flat_end, dtype=torch.int32, device=swa_idx.device
        ).view(end - start, width)
    return kv, indices, lengths


def sparse_attention_triton_const(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    block_h: int,
    block_n: int,
) -> torch.Tensor:
    if indices.ndim == 3:
        indices = indices.squeeze(1)
    if kv.ndim == 3:
        kv = kv.squeeze(1)
    out = torch.empty_like(q)
    grid = (q.shape[0], triton.cdiv(q.shape[1], block_h))
    tk._sparse_attention_bf16_kernel[grid](
        q,
        kv,
        indices,
        lengths,
        sink,
        out,
        q.shape[0],
        q.shape[1],
        kv.shape[0],
        indices.shape[-1],
        scale * LOG2E,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        indices.stride(0),
        indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_D=HEAD_DIM,
        HAS_SINK=True,
        LOG2E_CONST=LOG2E,
        num_warps=8,
    )
    return out


def decode_sparse_attention_triton_const(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    extra_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_lens: torch.Tensor,
    block_h: int,
    block_n: int,
) -> None:
    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)
    num_tokens, num_heads, _ = q.shape
    grid = (num_tokens, triton.cdiv(num_heads, block_h))
    tk._decode_sparse_attention_fp8_kernel[grid](
        q,
        swa_cache,
        swa_cache.view(torch.bfloat16),
        swa_cache,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_cache.view(torch.bfloat16),
        extra_cache,
        extra_indices,
        extra_lens,
        attn_sink if attn_sink is not None else q,
        out,
        num_tokens,
        num_heads,
        swa_indices.shape[-1],
        extra_indices.shape[-1],
        swa_cache.shape[0],
        extra_cache.shape[0],
        swa_cache.shape[1],
        extra_cache.shape[1],
        swa_cache.stride(0),
        extra_cache.stride(0),
        scale * LOG2E,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        swa_indices.stride(0),
        swa_indices.stride(1),
        extra_indices.stride(0),
        extra_indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_D=HEAD_DIM,
        FP8_DIM=tk.FP8_DS_MLA_FP8_DIM,
        SCALE_GROUP=tk.FP8_DS_MLA_SCALE_GROUP,
        SCALE_BYTES=tk.FP8_DS_MLA_SCALE_BYTES,
        TOKEN_BYTES=tk.FP8_DS_MLA_TOKEN_BYTES,
        HAS_EXTRA=True,
        HAS_SINK=attn_sink is not None,
        LOG2E_CONST=LOG2E,
        num_warps=8,
    )


def dequant_gather(case, swa_idx, extra_idx):
    tokens = swa_idx.shape[0]
    width = swa_idx.shape[1] + extra_idx.shape[1]
    kv = torch.empty(tokens, width, HEAD_DIM, dtype=torch.bfloat16, device=swa_idx.device)
    dequantize_global_slots_k_cache(kv[:, : swa_idx.shape[1]], case["swa_cache"], swa_idx, 256)
    dequantize_global_slots_k_cache(
        kv[:, swa_idx.shape[1] :],
        case["extra_cache"],
        extra_idx,
        case["extra_cache"].shape[1],
    )
    indices = torch.arange(tokens * width, dtype=torch.int32, device=swa_idx.device).view(
        tokens, 1, width
    )
    lengths = torch.full((tokens,), width, dtype=torch.int32, device=swa_idx.device)
    return kv.view(tokens * width, 1, HEAD_DIM), indices, lengths


def row(mode: str, cache: str, backend: str, tokens: int, us: float) -> None:
    ms = us / 1000.0
    tps = tokens * 1_000_000.0 / us
    print(f"{mode},{cache},{backend},{tokens},{ms:.3f},{tps:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--context", type=int, default=16_384)
    parser.add_argument("--prefill-tokens", type=int, default=16_384)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--swa-topk", type=int, default=128)
    parser.add_argument("--extra-topk", type=int, default=128)
    parser.add_argument("--extra-block", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--only",
        choices=[
            "prefill-fp8-native",
            "prefill-fp8-native-fused",
            "prefill-fp8-triton-dequant",
            "prefill-fp8-triton-gathered",
            "prefill-int8-native",
            "prefill-int8-triton",
            "decode-fp8-native",
            "decode-fp8-triton",
            "decode-fp8-triton-bh8-bn32",
            "decode-fp8-triton-bh16",
            "decode-fp8-triton-bh16-bn8",
            "decode-fp8-triton-bh16-bn32",
            "decode-fp8-triton-bh32",
            "decode-int8-triton",
        ],
        default=None,
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}"
    scale = 1.0 / math.sqrt(HEAD_DIM)

    print("# mode,cache,backend,tokens,ms,tok_per_s", flush=True)
    print(
        "# shape="
        f"context={args.context},prefill={args.prefill_tokens},decode={args.decode_tokens},"
        f"H={H},D={HEAD_DIM},swa_topk={args.swa_topk},extra_topk={args.extra_topk},"
        f"extra_block={args.extra_block}",
        flush=True,
    )

    swa_cache, swa_rows = build_cache(args.context, 256, device)
    extra_cache, extra_rows = build_cache(args.context, args.extra_block, device)
    int8_swa, int8_swa_scale = build_int8_cache(swa_rows, 256)
    int8_extra, int8_extra_scale = build_int8_cache(extra_rows, args.extra_block)
    case = {
        "swa_cache": swa_cache,
        "extra_cache": extra_cache,
        "swa_rows": swa_rows,
        "extra_rows": extra_rows,
    }

    sink = torch.randn(H, device=device, dtype=torch.float32) * 0.1

    # Prefill.
    q_prefill = torch.randn(args.prefill_tokens, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    pre_swa_idx = make_indices(args.prefill_tokens, args.swa_topk, args.context, device)
    pre_extra_idx = make_indices(args.prefill_tokens, args.extra_topk, args.context, device)
    pre_swa_lens = torch.full((args.prefill_tokens,), args.swa_topk, dtype=torch.int32, device=device)
    pre_extra_lens = torch.full((args.prefill_tokens,), args.extra_topk, dtype=torch.int32, device=device)

    if args.only in (None, "prefill-fp8-native", "prefill-fp8-native-fused"):
        row(
        "prefill",
        "fp8",
        "native_fused",
        args.prefill_tokens,
        time_us(
            lambda: sparse_mla_prefill(
                q_prefill,
                swa_cache,
                pre_swa_idx.unsqueeze(1),
                pre_swa_lens,
                scale=scale,
                attn_sink=sink,
                extra_cache=extra_cache,
                extra_indices=pre_extra_idx,
                extra_lens=pre_extra_lens,
            ),
            args.warmup,
            args.iters,
        ),
        )
        if args.only is not None:
            return

    kv, indices, lengths = dequant_gather(case, pre_swa_idx, pre_extra_idx)
    if args.only in (None, "prefill-fp8-triton-dequant"):
        row(
        "prefill",
        "fp8",
        "triton_bh32_bn16_dequant_gather",
        args.prefill_tokens,
        time_us(
            lambda: (
                dequant_gather(case, pre_swa_idx, pre_extra_idx),
                sparse_attention_triton_const(q_prefill, kv, indices, lengths, sink, scale, 32, 16),
            ),
            args.warmup,
            args.iters,
        ),
        )
        if args.only is not None:
            return
    if args.only in (None, "prefill-fp8-triton-gathered"):
        row(
        "prefill",
        "fp8",
        "triton_bh32_bn16_gathered_only",
        args.prefill_tokens,
        time_us(
            lambda: sparse_attention_triton_const(q_prefill, kv, indices, lengths, sink, scale, 32, 16),
            args.warmup,
            args.iters,
        ),
        )
        if args.only is not None:
            return

    if args.only in (None, "prefill-int8-native"):
        row(
        "prefill",
        "int8",
        "native_fused",
        args.prefill_tokens,
        time_us(
            lambda: sparse_mla_prefill_int8(
                q_prefill,
                int8_swa,
                int8_swa_scale,
                pre_swa_idx,
                pre_swa_lens,
                scale=scale,
                attn_sink=sink,
                extra_cache=int8_extra,
                extra_scale=int8_extra_scale,
                extra_indices=pre_extra_idx,
                extra_lens=pre_extra_lens,
            ),
            args.warmup,
            args.iters,
        ),
        )
        if args.only is not None:
            return
    if args.only in (None, "prefill-int8-triton"):
        row(
        "prefill",
        "int8",
        "triton_imma_bh32_bn32",
        args.prefill_tokens,
        time_us(
            lambda: sparse_mla_decode_int8_triton(
                q_prefill,
                int8_swa,
                int8_swa_scale,
                pre_swa_idx,
                pre_swa_lens,
                scale=scale,
                attn_sink=sink,
                extra_cache=int8_extra,
                extra_scale=int8_extra_scale,
                extra_indices=pre_extra_idx,
                extra_lens=pre_extra_lens,
                block_h=32,
                block_n=32,
            ),
            args.warmup,
            args.iters,
        ),
        )
        if args.only is not None:
            return

    del q_prefill, kv, indices, lengths
    torch.cuda.empty_cache()

    # Decode.
    q_decode = torch.randn(args.decode_tokens, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dec_swa_idx = make_indices(args.decode_tokens, args.swa_topk, args.context, device)
    dec_extra_idx = make_indices(args.decode_tokens, args.extra_topk, args.context, device)
    dec_swa_lens = torch.full((args.decode_tokens,), args.swa_topk, dtype=torch.int32, device=device)
    dec_extra_lens = torch.full((args.decode_tokens,), args.extra_topk, dtype=torch.int32, device=device)

    if args.only in (None, "decode-fp8-native"):
        row(
        "decode",
        "fp8",
        "native",
        args.decode_tokens,
        time_us(
            lambda: sparse_mla_decode_fp8(
                q_decode,
                swa_cache,
                dec_swa_idx,
                dec_swa_lens,
                scale=scale,
                attn_sink=sink,
                extra_cache=extra_cache,
                extra_indices=dec_extra_idx,
                extra_lens=dec_extra_lens,
            ),
            args.warmup,
            args.iters,
        ),
        )
        if args.only is not None:
            return
    out = torch.empty_like(q_decode)
    if args.only in (None, "decode-fp8-triton"):
        row(
        "decode",
        "fp8",
        "triton_default",
        args.decode_tokens,
        time_us(
            lambda: decode_sparse_attention_triton(
                q=q_decode,
                swa_cache=swa_cache,
                swa_indices=dec_swa_idx,
                swa_lens=dec_swa_lens,
                scale=scale,
                attn_sink=sink,
                out=out,
                extra_cache=extra_cache,
                extra_indices=dec_extra_idx,
                extra_lens=dec_extra_lens,
            ),
            args.warmup,
            args.iters,
        ),
        )
        if args.only is not None:
            return
    if args.only in (None, "decode-fp8-triton-bh8-bn32"):
        row(
            "decode",
            "fp8",
            "triton_bh8_bn32",
            args.decode_tokens,
            time_us(
                lambda: decode_sparse_attention_triton_const(
                    q=q_decode,
                    swa_cache=swa_cache,
                    swa_indices=dec_swa_idx,
                    swa_lens=dec_swa_lens,
                    scale=scale,
                    attn_sink=sink,
                    out=out,
                    extra_cache=extra_cache,
                    extra_indices=dec_extra_idx,
                    extra_lens=dec_extra_lens,
                    block_h=8,
                    block_n=32,
                ),
                args.warmup,
                args.iters,
            ),
        )
        if args.only is not None:
            return
    if args.only in (None, "decode-fp8-triton-bh16"):
        row(
            "decode",
            "fp8",
            "triton_bh16_bn16",
            args.decode_tokens,
            time_us(
                lambda: decode_sparse_attention_triton_const(
                    q=q_decode,
                    swa_cache=swa_cache,
                    swa_indices=dec_swa_idx,
                    swa_lens=dec_swa_lens,
                    scale=scale,
                    attn_sink=sink,
                    out=out,
                    extra_cache=extra_cache,
                    extra_indices=dec_extra_idx,
                    extra_lens=dec_extra_lens,
                    block_h=16,
                    block_n=16,
                ),
                args.warmup,
                args.iters,
            ),
        )
        if args.only is not None:
            return
    if args.only in (None, "decode-fp8-triton-bh16-bn8"):
        row(
            "decode",
            "fp8",
            "triton_bh16_bn8",
            args.decode_tokens,
            time_us(
                lambda: decode_sparse_attention_triton_const(
                    q=q_decode,
                    swa_cache=swa_cache,
                    swa_indices=dec_swa_idx,
                    swa_lens=dec_swa_lens,
                    scale=scale,
                    attn_sink=sink,
                    out=out,
                    extra_cache=extra_cache,
                    extra_indices=dec_extra_idx,
                    extra_lens=dec_extra_lens,
                    block_h=16,
                    block_n=8,
                ),
                args.warmup,
                args.iters,
            ),
        )
        if args.only is not None:
            return
    if args.only in (None, "decode-fp8-triton-bh16-bn32"):
        row(
            "decode",
            "fp8",
            "triton_bh16_bn32",
            args.decode_tokens,
            time_us(
                lambda: decode_sparse_attention_triton_const(
                    q=q_decode,
                    swa_cache=swa_cache,
                    swa_indices=dec_swa_idx,
                    swa_lens=dec_swa_lens,
                    scale=scale,
                    attn_sink=sink,
                    out=out,
                    extra_cache=extra_cache,
                    extra_indices=dec_extra_idx,
                    extra_lens=dec_extra_lens,
                    block_h=16,
                    block_n=32,
                ),
                args.warmup,
                args.iters,
            ),
        )
        if args.only is not None:
            return
    if args.only in (None, "decode-fp8-triton-bh32"):
        row(
            "decode",
            "fp8",
            "triton_bh32_bn16",
            args.decode_tokens,
            time_us(
                lambda: decode_sparse_attention_triton_const(
                    q=q_decode,
                    swa_cache=swa_cache,
                    swa_indices=dec_swa_idx,
                    swa_lens=dec_swa_lens,
                    scale=scale,
                    attn_sink=sink,
                    out=out,
                    extra_cache=extra_cache,
                    extra_indices=dec_extra_idx,
                    extra_lens=dec_extra_lens,
                    block_h=32,
                    block_n=16,
                ),
                args.warmup,
                args.iters,
            ),
        )
        if args.only is not None:
            return
    if args.only in (None, "decode-int8-triton"):
        row(
        "decode",
        "int8",
        "triton_imma_bh32_bn32",
        args.decode_tokens,
        time_us(
            lambda: sparse_mla_decode_int8_triton(
                q_decode,
                int8_swa,
                int8_swa_scale,
                dec_swa_idx,
                dec_swa_lens,
                scale=scale,
                attn_sink=sink,
                extra_cache=int8_extra,
                extra_scale=int8_extra_scale,
                extra_indices=dec_extra_idx,
                extra_lens=dec_extra_lens,
                block_h=32,
                block_n=32,
            ),
            args.warmup,
            args.iters,
        ),
        )
        if args.only is not None:
            return


if __name__ == "__main__":
    main()
