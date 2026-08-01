"""Direct INT8 SparseMLA benchmark at the DSV4 TP=2 target-prefill shape."""

from __future__ import annotations

import math
import os

import pytest
import torch


HEAD_DIM = 512
TOKEN_BYTES = 528
WINDOW = 128
INDEX_TOPK = 512
HEADS_PER_TP2_RANK = 32

pytestmark = pytest.mark.skipif(
    os.environ.get("DSV4_RUN_PERF_TESTS") != "1",
    reason="set DSV4_RUN_PERF_TESTS=1 for the allocation-heavy performance probe",
)


def _inline_cache(num_slots: int, block_size: int):
    blocks = (num_slots + block_size - 1) // block_size
    rows = torch.randn(
        blocks * block_size,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    scales = (rows.float().abs().amax(dim=-1) / 127.0).clamp_min(1.0e-12)
    quantized = torch.round(rows.float() / scales[:, None]).clamp(-127, 127)
    quantized = quantized.to(torch.int8)
    cache = torch.zeros(
        blocks,
        block_size,
        TOKEN_BYTES,
        dtype=torch.uint8,
        device="cuda",
    )
    cache[:, :, :HEAD_DIM] = quantized.view(blocks, block_size, HEAD_DIM).view(
        torch.uint8
    )
    cache[:, :, HEAD_DIM : HEAD_DIM + 4] = scales.view(
        blocks, block_size, 1
    ).view(torch.uint8)
    row_view = cache[:, :, :HEAD_DIM].view(torch.int8)
    scale_view = cache[:, :, HEAD_DIM : HEAD_DIM + 4].view(torch.float32).squeeze(-1)
    assert row_view.stride(1) == TOKEN_BYTES
    return row_view, scale_view


def _elapsed_ms(call, iterations: int) -> float:
    call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1024, 8192])
def test_dsv4_tp2_target_prefill_runtime(tokens: int) -> None:
    from flash_mla import sparse_mla_prefill_int8

    torch.manual_seed(121 + tokens)
    swa_cache, swa_scale = _inline_cache(384, 256)
    extra_cache, extra_scale = _inline_cache(640, 64)
    q = torch.randn(
        tokens,
        HEADS_PER_TP2_RANK,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    swa_indices = torch.randint(
        0, 384, (tokens, WINDOW), dtype=torch.int32, device="cuda"
    )
    extra_indices = torch.randint(
        0, 640, (tokens, INDEX_TOPK), dtype=torch.int32, device="cuda"
    )
    swa_lens = torch.full((tokens,), WINDOW, dtype=torch.int32, device="cuda")
    extra_lens = torch.full(
        (tokens,), INDEX_TOPK, dtype=torch.int32, device="cuda"
    )
    sink = torch.zeros(HEADS_PER_TP2_RANK, dtype=torch.float32, device="cuda")

    def call():
        return sparse_mla_prefill_int8(
            q,
            swa_cache,
            swa_scale,
            swa_indices,
            swa_lens,
            scale=1.0 / math.sqrt(HEAD_DIM),
            attn_sink=sink,
            extra_cache=extra_cache,
            extra_scale=extra_scale,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
        )

    output = call()
    assert output.shape == q.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    elapsed = _elapsed_ms(call, iterations=3 if tokens == 1024 else 1)
    print(
        f"tokens={tokens} heads={HEADS_PER_TP2_RANK} "
        f"swa={WINDOW} extra={INDEX_TOPK} elapsed_ms={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    for token_count in (1024, 8192):
        test_dsv4_tp2_target_prefill_runtime(token_count)
