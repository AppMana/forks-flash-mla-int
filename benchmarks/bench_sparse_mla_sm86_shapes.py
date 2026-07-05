#!/usr/bin/env python3
"""Microbench AppMana DeepSeek V4 sm86 sparse MLA shapes.

Reports microseconds. Baselines:
- flashmla: direct fp8_ds_mla paged-cache CUDA kernel from this fork.
- triton: vLLM sm86 Triton kernels.
- torch_ref: fp32 torch oracle over the selected bf16 K rows.

Run on this workstation's second GPU:
  CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_sparse_mla_sm86_shapes.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

from flash_mla import (
    flash_sparse_mla_decode,
    flash_sparse_mla_prefill,
)

try:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        dequantize_global_slots_k_cache,
    )
    from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import (
        decode_sparse_attention_triton,
        sparse_attention_triton,
    )
except Exception as exc:  # pragma: no cover - bench-only import guard
    decode_sparse_attention_triton = None
    sparse_attention_triton = None
    dequantize_global_slots_k_cache = None
    TRITON_IMPORT_ERROR = exc
else:
    TRITON_IMPORT_ERROR = None


FP8_DIM = 448
ROPE_DIM = 64
SCALE_DIM = 8
TOKEN_DATA_SIZE = FP8_DIM + ROPE_DIM * 2
SCALE_GROUP = 64
HEAD_DIM = 512
H = 64
SWA_TOPK = 128


@dataclass
class Case:
    q: torch.Tensor
    swa_cache: torch.Tensor
    swa_k: torch.Tensor
    swa_idx: torch.Tensor
    swa_lens: torch.Tensor
    extra_cache: torch.Tensor
    extra_k: torch.Tensor
    extra_idx: torch.Tensor
    extra_lens: torch.Tensor
    sink: torch.Tensor
    scale: float


def write_token(cache: torch.Tensor, slot: int, block_size: int) -> torch.Tensor:
    block_idx = slot // block_size
    block_offset = slot % block_size
    values = (
        (torch.arange(FP8_DIM, device=cache.device, dtype=torch.float32) % 17) - 8
    ) / 16.0
    values = values + float(slot) / 32.0
    exponents = torch.tensor(
        [-2, -1, 0, 1, 2, -2, 1], device=cache.device, dtype=torch.float32
    )
    scales = torch.exp2(exponents).repeat_interleave(SCALE_GROUP)
    fp8_values = (values / scales).to(torch.float8_e4m3fn)
    expected_nope = fp8_values.float() * scales
    rope = (
        torch.linspace(-1.0, 1.0, ROPE_DIM, device=cache.device)
        + float(slot) / 16.0
    ).to(torch.bfloat16)

    flat = cache[block_idx].view(-1)
    data_start = block_offset * TOKEN_DATA_SIZE
    scale_start = block_size * TOKEN_DATA_SIZE + block_offset * SCALE_DIM
    flat[data_start : data_start + FP8_DIM] = fp8_values.view(torch.uint8)
    flat[data_start + FP8_DIM : data_start + TOKEN_DATA_SIZE] = rope.view(torch.uint8)
    enc = (exponents.to(torch.int32) + 127).to(torch.uint8)
    flat[scale_start : scale_start + enc.numel()] = enc
    flat[scale_start + enc.numel() : scale_start + SCALE_DIM] = 127
    return torch.cat([expected_nope, rope.float()]).to(torch.bfloat16)


def build_cache(num_slots: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = (num_slots + block_size - 1) // block_size
    cache = torch.zeros(
        num_blocks,
        block_size,
        TOKEN_DATA_SIZE + SCALE_DIM,
        dtype=torch.uint8,
        device=device,
    )
    k = torch.zeros(num_blocks * block_size, HEAD_DIM, dtype=torch.bfloat16, device=device)
    for slot in range(num_slots):
        k[slot] = write_token(cache, slot, block_size)
    return cache, k


def make_case(tokens: int, extra_topk: int, extra_block_size: int, device: str) -> Case:
    swa_slots = SWA_TOPK + 64
    extra_slots = extra_topk + 64
    swa_cache, swa_k = build_cache(swa_slots, 256, device)
    extra_cache, extra_k = build_cache(extra_slots, extra_block_size, device)
    q = torch.randn(tokens, H, HEAD_DIM, device=device, dtype=torch.bfloat16)
    swa_idx = torch.stack(
        [torch.randperm(swa_k.shape[0], device=device)[:SWA_TOPK].to(torch.int32) for _ in range(tokens)]
    )
    extra_idx = torch.stack(
        [torch.randperm(extra_k.shape[0], device=device)[:extra_topk].to(torch.int32) for _ in range(tokens)]
    )
    return Case(
        q=q,
        swa_cache=swa_cache,
        swa_k=swa_k,
        swa_idx=swa_idx,
        swa_lens=torch.full((tokens,), SWA_TOPK, dtype=torch.int32, device=device),
        extra_cache=extra_cache,
        extra_k=extra_k,
        extra_idx=extra_idx,
        extra_lens=torch.full((tokens,), extra_topk, dtype=torch.int32, device=device),
        sink=torch.randn(H, device=device, dtype=torch.float32) * 0.1,
        scale=1.0 / math.sqrt(HEAD_DIM),
    )


def build_gathered_prefill(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = case.q.shape[0]
    width = int(case.swa_idx.shape[1] + case.extra_idx.shape[1])
    kv = torch.empty(tokens * width, 1, HEAD_DIM, dtype=torch.bfloat16, device=case.q.device)
    indices = torch.empty(tokens, 1, width, dtype=torch.int32, device=case.q.device)
    lengths = torch.full((tokens,), width, dtype=torch.int32, device=case.q.device)
    for t in range(tokens):
        rows = torch.cat(
            [
                case.swa_k[case.swa_idx[t].long()],
                case.extra_k[case.extra_idx[t].long()],
            ],
            dim=0,
        )
        start = t * width
        kv[start : start + width, 0, :] = rows
        indices[t, 0, :] = torch.arange(start, start + width, dtype=torch.int32, device=case.q.device)
    return kv, indices, lengths


def build_empty_prefill(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = case.q.shape[0]
    width = int(case.swa_idx.shape[1] + case.extra_idx.shape[1])
    kv = torch.empty(tokens * width, 1, HEAD_DIM, dtype=torch.bfloat16, device=case.q.device)
    indices = torch.arange(tokens * width, dtype=torch.int32, device=case.q.device).view(tokens, 1, width)
    lengths = torch.full((tokens,), width, dtype=torch.int32, device=case.q.device)
    return kv, indices, lengths


def dequant_gather_prefill(case: Case, kv: torch.Tensor) -> None:
    tokens = case.q.shape[0]
    swa_width = case.swa_idx.shape[1]
    extra_width = case.extra_idx.shape[1]
    view = kv.view(tokens, swa_width + extra_width, HEAD_DIM)
    dequantize_global_slots_k_cache(view[:, :swa_width], case.swa_cache, case.swa_idx, 256)
    dequantize_global_slots_k_cache(
        view[:, swa_width : swa_width + extra_width],
        case.extra_cache,
        case.extra_idx,
        case.extra_cache.shape[1],
    )


def torch_ref(case: Case) -> torch.Tensor:
    out = torch.empty_like(case.q, dtype=torch.float32)
    for t in range(case.q.shape[0]):
        k = torch.cat(
            [
                case.swa_k[case.swa_idx[t].long()],
                case.extra_k[case.extra_idx[t].long()],
            ],
            dim=0,
        ).float()
        scores = (case.q[t].float() @ k.t()) * case.scale
        sink = case.sink[:, None].float()
        m = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - m)
        out[t] = (exp_scores @ k) / (
            exp_scores.sum(-1, keepdim=True) + torch.exp(sink - m)
        )
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


def print_row(mode: str, layer: str, tokens: int, extra_topk: int, extra_block: int, impl: str, us: float) -> None:
    print(
        f"{mode},{layer},{tokens},{extra_topk},{extra_block},{impl},"
        f"{us:.3f},{us / max(tokens, 1):.3f}",
        flush=True,
    )


def bench_decode(case: Case, layer: str, extra_topk: int, extra_block: int, args: argparse.Namespace) -> None:
    tokens = case.q.shape[0]
    out_flash = flash_sparse_mla_decode(
        q=case.q,
        swa_cache=case.swa_cache,
        swa_indices=case.swa_idx,
        swa_lens=case.swa_lens,
        scale=case.scale,
        attn_sink=case.sink,
        extra_cache=case.extra_cache,
        extra_indices=case.extra_idx,
        extra_lens=case.extra_lens,
    )
    if decode_sparse_attention_triton is not None:
        out_triton = torch.empty_like(case.q)
        decode_sparse_attention_triton(
            q=case.q,
            swa_cache=case.swa_cache,
            swa_indices=case.swa_idx,
            swa_lens=case.swa_lens,
            scale=case.scale,
            attn_sink=case.sink,
            out=out_triton,
            extra_cache=case.extra_cache,
            extra_indices=case.extra_idx,
            extra_lens=case.extra_lens,
        )
        cos = 1 - 2 * (out_flash.float() * out_triton.float()).sum().item() / max(
            (out_flash.float().square() + out_triton.float().square()).sum().item(),
            1e-12,
        )
        if cos > 8e-5:
            raise RuntimeError(f"decode FlashMLA/Triton mismatch: cos_diff={cos:.2e}")

    print_row(
        "decode",
        layer,
        tokens,
        extra_topk,
        extra_block,
        "flashmla",
        time_us(
            lambda: flash_sparse_mla_decode(
                q=case.q,
                swa_cache=case.swa_cache,
                swa_indices=case.swa_idx,
                swa_lens=case.swa_lens,
                scale=case.scale,
                attn_sink=case.sink,
                extra_cache=case.extra_cache,
                extra_indices=case.extra_idx,
                extra_lens=case.extra_lens,
            ),
            args.warmup,
            args.iters,
        ),
    )
    if decode_sparse_attention_triton is not None:
        out = torch.empty_like(case.q)
        print_row(
            "decode",
            layer,
            tokens,
            extra_topk,
            extra_block,
            "triton",
            time_us(
                lambda: decode_sparse_attention_triton(
                    q=case.q,
                    swa_cache=case.swa_cache,
                    swa_indices=case.swa_idx,
                    swa_lens=case.swa_lens,
                    scale=case.scale,
                    attn_sink=case.sink,
                    out=out,
                    extra_cache=case.extra_cache,
                    extra_indices=case.extra_idx,
                    extra_lens=case.extra_lens,
                ),
                args.warmup,
                args.iters,
            ),
        )
    if tokens <= args.max_torch_tokens and extra_topk <= args.max_torch_topk:
        print_row(
            "decode",
            layer,
            tokens,
            extra_topk,
            extra_block,
            "torch_ref",
            time_us(lambda: torch_ref(case), max(1, args.warmup // 5), max(1, args.torch_iters)),
        )


def bench_prefill(case: Case, layer: str, extra_topk: int, extra_block: int, args: argparse.Namespace) -> None:
    tokens = case.q.shape[0]
    out_flash = flash_sparse_mla_prefill(
        q=case.q,
        swa_cache=case.swa_cache,
        swa_indices=case.swa_idx.unsqueeze(1),
        swa_lens=case.swa_lens,
        scale=case.scale,
        attn_sink=case.sink,
        extra_cache=case.extra_cache,
        extra_indices=case.extra_idx,
        extra_lens=case.extra_lens,
        use_staged_prefill=False,
    )
    if sparse_attention_triton is not None:
        kv, indices, lengths = build_gathered_prefill(case)
        out_triton = torch.empty_like(case.q)
        sparse_attention_triton(
            q=case.q,
            kv=kv,
            indices=indices,
            lengths=lengths,
            scale=case.scale,
            attn_sink=case.sink,
            out=out_triton,
        )
        cos = 1 - 2 * (out_flash.float() * out_triton.float()).sum().item() / max(
            (out_flash.float().square() + out_triton.float().square()).sum().item(),
            1e-12,
        )
        if cos > 8e-5:
            raise RuntimeError(f"prefill FlashMLA/Triton mismatch: cos_diff={cos:.2e}")

    print_row(
        "prefill",
        layer,
        tokens,
        extra_topk,
        extra_block,
        "flashmla_fused",
        time_us(
            lambda: flash_sparse_mla_prefill(
                q=case.q,
                swa_cache=case.swa_cache,
                swa_indices=case.swa_idx.unsqueeze(1),
                swa_lens=case.swa_lens,
                scale=case.scale,
                attn_sink=case.sink,
                extra_cache=case.extra_cache,
                extra_indices=case.extra_idx,
                extra_lens=case.extra_lens,
                use_staged_prefill=False,
            ),
            args.warmup,
            args.iters,
        ),
    )
    print_row(
        "prefill",
        layer,
        tokens,
        extra_topk,
        extra_block,
        "flashmla_staged",
        time_us(
            lambda: flash_sparse_mla_prefill(
                q=case.q,
                swa_cache=case.swa_cache,
                swa_indices=case.swa_idx.unsqueeze(1),
                swa_lens=case.swa_lens,
                scale=case.scale,
                attn_sink=case.sink,
                extra_cache=case.extra_cache,
                extra_indices=case.extra_idx,
                extra_lens=case.extra_lens,
                use_staged_prefill=True,
            ),
            args.warmup,
            args.iters,
        ),
    )
    if sparse_attention_triton is not None:
        kv, indices, lengths = build_gathered_prefill(case)
        out = torch.empty_like(case.q)
        print_row(
            "prefill",
            layer,
            tokens,
            extra_topk,
            extra_block,
            "triton_gathered_bf16",
            time_us(
                lambda: sparse_attention_triton(
                    q=case.q,
                    kv=kv,
                    indices=indices,
                    lengths=lengths,
                    scale=case.scale,
                    attn_sink=case.sink,
                    out=out,
                ),
                args.warmup,
                args.iters,
            ),
        )
        if dequantize_global_slots_k_cache is not None:
            kv, indices, lengths = build_empty_prefill(case)
            out = torch.empty_like(case.q)
            print_row(
                "prefill",
                layer,
                tokens,
                extra_topk,
                extra_block,
                "triton_dequant_gather_bf16",
                time_us(
                    lambda: (
                        dequant_gather_prefill(case, kv),
                        sparse_attention_triton(
                            q=case.q,
                            kv=kv,
                            indices=indices,
                            lengths=lengths,
                            scale=case.scale,
                            attn_sink=case.sink,
                            out=out,
                        ),
                    ),
                    args.warmup,
                    args.iters,
                ),
            )
    if tokens <= args.max_torch_tokens and extra_topk <= args.max_torch_topk:
        print_row(
            "prefill",
            layer,
            tokens,
            extra_topk,
            extra_block,
            "torch_ref",
            time_us(lambda: torch_ref(case), max(1, args.warmup // 5), max(1, args.torch_iters)),
        )


def run(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability(args.device)[0] != 8:
        raise SystemExit("This benchmark requires an Ampere CUDA device.")
    if TRITON_IMPORT_ERROR is not None:
        print(f"# Triton baseline unavailable: {TRITON_IMPORT_ERROR!r}")

    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}"
    print("mode,layer,tokens,extra_topk,extra_block,impl,us,us_per_token", flush=True)
    decode_layers = (
        ("c4a", 512, 64),
        ("c128a_16k", 128, 2),
        ("c128a_64k", 512, 2),
        ("c128a_200k", 1664, 2),
    )
    if not args.skip_decode:
        for batch in args.decode_batches:
            for layer, extra_topk, extra_block in decode_layers:
                bench_decode(make_case(batch, extra_topk, extra_block, device), layer, extra_topk, extra_block, args)

    prefill_layers = (("c4a", 512, 64), ("c128a_64k", 512, 2))
    if not args.skip_prefill:
        for tokens in args.prefill_tokens:
            for layer, extra_topk, extra_block in prefill_layers:
                bench_prefill(make_case(tokens, extra_topk, extra_block, device), layer, extra_topk, extra_block, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--torch-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--decode-batches", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--prefill-tokens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--max-torch-tokens", type=int, default=256)
    parser.add_argument("--max-torch-topk", type=int, default=512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
