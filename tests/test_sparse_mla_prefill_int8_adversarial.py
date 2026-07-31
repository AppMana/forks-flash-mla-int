"""Adversarial parity for the fused int8_ds_mla sparse-MLA prefill (sm_86).

Mirrors tests/test_sparse_mla_prefill_adversarial.py for the int8 cache. The row
format ground truth is the vLLM fork's int8 writer (cache_utils.py, int8kv-fix):
512 signed-int8 payload bytes (the 64 RoPE dims are quantized like the rest) +
one little-endian fp32 rowwise scale at byte offset 512, token stride 528
(512 + 4 + 12 pad, 16B-aligned). The kernel must take the stride at runtime, so
both layouts are exercised:
  - inline 528B byte cache consumed through strided (int8_rows, fp32_scales)
    views, exactly like get_int8_ds_mla_cache_views produces, and
  - the legacy separate-tensor layout (contiguous int8 [nb, bs, 512] + fp32
    scale [nb, bs]).
Oracle: fp32 torch over the DEQUANTIZED (int8 * scale) rows.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_sparse_mla_prefill_int8_adversarial.py -x -q
"""
import math

import pytest
import torch

from test_sparse_mla_prefill_adversarial import _ref

HEAD_DIM = 512
INT8_DIM = 512
SCALE_BYTES = 4
PAD_BYTES = 12
TOKEN_BYTES = INT8_DIM + SCALE_BYTES + PAD_BYTES  # 528


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def _quantize_rows(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rowwise symmetric int8 like the vLLM writer (_int8_ds_mla_quantize_rows)."""
    k_fp32 = k.to(torch.float32)
    scale = (k_fp32.abs().amax(dim=-1) / 127.0).clamp_min(1.0e-12)
    q = torch.round(k_fp32 / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def _build_inline_cache(num_slots: int, block_size: int, dev: str):
    """528B-stride inline-scale cache + strided views (get_int8_ds_mla_cache_views)."""
    nb = (num_slots + block_size - 1) // block_size
    rows_bf16 = (torch.randn(nb * block_size, HEAD_DIM, device=dev) * 2.0).to(torch.bfloat16)
    q_i8, scales = _quantize_rows(rows_bf16)
    dequant = (q_i8.float() * scales.unsqueeze(-1)).to(torch.bfloat16)

    cache = torch.zeros(nb, block_size, TOKEN_BYTES, dtype=torch.uint8, device=dev)
    cache[:, :, :INT8_DIM] = q_i8.view(nb, block_size, INT8_DIM).view(torch.uint8)
    cache[:, :, INT8_DIM:INT8_DIM + SCALE_BYTES] = (
        scales.view(nb, block_size, 1).view(torch.uint8)
    )
    data_view = cache[:, :, :INT8_DIM].view(torch.int8)          # stride(1) = 528
    scale_view = (
        cache[:, :, INT8_DIM:INT8_DIM + SCALE_BYTES].view(torch.float32).squeeze(-1)
    )
    assert data_view.stride(1) == TOKEN_BYTES
    return data_view, scale_view, dequant


def _build_separate_cache(num_slots: int, block_size: int, dev: str):
    """Legacy layout: contiguous int8 [nb, bs, 512] + separate fp32 scale [nb, bs]."""
    nb = (num_slots + block_size - 1) // block_size
    rows_bf16 = (torch.randn(nb * block_size, HEAD_DIM, device=dev) * 2.0).to(torch.bfloat16)
    q_i8, scales = _quantize_rows(rows_bf16)
    dequant = (q_i8.float() * scales.unsqueeze(-1)).to(torch.bfloat16)
    return (
        q_i8.view(nb, block_size, INT8_DIM).contiguous(),
        scales.view(nb, block_size).contiguous(),
        dequant,
    )


@pytest.mark.parametrize("layout", ["inline528", "separate"])
@pytest.mark.parametrize("swa_topk,extra_topk", [(256, 256), (512, 512)])
def test_int8_prefill_adversarial_parity(layout, swa_topk, extra_topk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    from flash_mla.int8_sparse_mla import sparse_mla_prefill_int8

    torch.manual_seed(21)
    dev = "cuda"
    T = 67
    H = 64
    scale = 1.0 / math.sqrt(HEAD_DIM)
    build = _build_inline_cache if layout == "inline528" else _build_separate_cache
    swa_cache, swa_scale, swa_K = build(swa_topk + 40, 64, dev)
    extra_cache, extra_scale, extra_K = build(extra_topk + 40, 16, dev)

    q = torch.randn(T, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    specials = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, swa_topk]
    swa_lens = torch.randint(0, swa_topk + 1, (T,), dtype=torch.int32, device=dev)
    extra_lens = torch.randint(0, extra_topk + 1, (T,), dtype=torch.int32, device=dev)
    for i, v in enumerate(specials):
        swa_lens[i] = min(v, swa_topk)
        extra_lens[-1 - i] = min(v, extra_topk)
    swa_idx = torch.randint(0, swa_K.shape[0], (T, swa_topk), dtype=torch.int32, device=dev)
    extra_idx = torch.randint(0, extra_K.shape[0], (T, extra_topk), dtype=torch.int32, device=dev)
    swa_idx[3, 2] = -1
    swa_idx[4, 0] = swa_K.shape[0] + 5
    extra_idx[5, 1] = -1
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _ref(q, swa_K, swa_idx, swa_lens, extra_K, extra_idx, extra_lens, scale, sink)
    out = sparse_mla_prefill_int8(
        q,
        swa_cache,
        swa_scale,
        swa_idx,
        swa_lens,
        scale=scale,
        attn_sink=sink,
        extra_cache=extra_cache,
        extra_scale=extra_scale,
        extra_indices=extra_idx,
        extra_lens=extra_lens,
    )
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, (
        f"int8 prefill adversarial cos_diff={cd:.2e} "
        f"(layout={layout} topk={swa_topk}+{extra_topk})"
    )
    torch.testing.assert_close(out.float(), O_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("layout", ["inline528", "separate"])
def test_int8_prefill_swa_only_parity(layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    from flash_mla.int8_sparse_mla import sparse_mla_prefill_int8

    torch.manual_seed(23)
    dev = "cuda"
    T, H, topk = 33, 64, 512
    scale = 1.0 / math.sqrt(HEAD_DIM)
    build = _build_inline_cache if layout == "inline528" else _build_separate_cache
    cache, cache_scale, K = build(topk + 32, 64, dev)
    q = torch.randn(T, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.randint(0, topk + 1, (T,), dtype=torch.int32, device=dev)
    lens[0] = 0
    lens[1] = topk
    idx = torch.randint(0, K.shape[0], (T, topk), dtype=torch.int32, device=dev)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _ref(q, K, idx, lens, None, None, None, scale, sink)
    out = sparse_mla_prefill_int8(
        q, cache, cache_scale, idx, lens, scale=scale, attn_sink=sink,
    )
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"int8 swa-only prefill cos_diff={cd:.2e} (layout={layout})"


@pytest.mark.parametrize("concurrency", [1, 2])
def test_int8_prefill_tp2_dspark_serving_shape(concurrency):
    """Exact TP2 DSpark draft-prefill shape used by DSV4-Flash.

    Five speculative tokens plus the anchor produce six rows per request.
    Each TP2 rank retains 32 query heads and attends over the 128-token SWA
    stream plus 512 C4 compressed-cache rows.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    from flash_mla.int8_sparse_mla import sparse_mla_prefill_int8

    torch.manual_seed(31 + concurrency)
    dev = "cuda"
    rows = concurrency * 6
    heads = 32
    swa_topk = 128
    extra_topk = 512
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # These are the live vLLM packed-cache page sizes at block_size=256.
    swa_cache, swa_scale, swa_k = _build_inline_cache(384, 256, dev)
    extra_cache, extra_scale, extra_k = _build_inline_cache(640, 64, dev)
    q = torch.randn(rows, heads, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    swa_lens = torch.full(
        (rows,), swa_topk, dtype=torch.int32, device=dev
    )
    extra_lens = torch.full(
        (rows,), extra_topk, dtype=torch.int32, device=dev
    )
    swa_idx = torch.stack(
        [torch.randperm(swa_k.shape[0], device=dev)[:swa_topk] for _ in range(rows)]
    ).to(torch.int32)
    extra_idx = torch.stack(
        [
            torch.randperm(extra_k.shape[0], device=dev)[:extra_topk]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    sink = torch.randn(heads, device=dev, dtype=torch.float32) * 0.1

    expected = _ref(
        q,
        swa_k,
        swa_idx,
        swa_lens,
        extra_k,
        extra_idx,
        extra_lens,
        scale,
        sink,
    )
    out = sparse_mla_prefill_int8(
        q,
        swa_cache,
        swa_scale,
        swa_idx,
        swa_lens,
        scale=scale,
        attn_sink=sink,
        extra_cache=extra_cache,
        extra_scale=extra_scale,
        extra_indices=extra_idx,
        extra_lens=extra_lens,
    )

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    assert cos_diff(out.float(), expected) < 8e-5
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
