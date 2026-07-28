"""Adversarial parity for the native int8_ds_mla sparse-MLA DECODE (sm_86),
``flash_mla.sparse_mla_decode_int8``. Row format is the vLLM fork's int8
writer: 512 int8 payload bytes + fp32 rowwise scale at byte offset 512, token
stride 528. The op takes the stride at runtime, so both the inline-528B
strided views and the legacy separate-tensor layout are exercised. Oracle:
fp32 torch over the dequantized rows; cross-checked vs the Triton int8 decode.
"""
import math

import pytest
import torch

from test_sparse_mla_prefill_adversarial import _ref
from test_sparse_mla_prefill_int8_adversarial import (
    _build_inline_cache,
    _build_separate_cache,
    cos_diff,
)

HEAD_DIM = 512


def _native_int8_decode():
    from flash_mla.int8_sparse_mla import sparse_mla_decode_int8

    return sparse_mla_decode_int8


@pytest.mark.parametrize("layout", ["inline528", "separate"])
@pytest.mark.parametrize("swa_topk,extra_topk", [(128, 384), (128, 896)])
@pytest.mark.parametrize("T", [1, 4])
def test_int8_decode_adversarial_parity(layout, swa_topk, extra_topk, T):
    """Small-T decode regime (split-KV engaged): native int8 decode vs fp32
    oracle and vs the Triton int8 decode, topk totals 512 and 1024."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    from flash_mla import sparse_mla_decode_int8_triton

    decode = _native_int8_decode()

    torch.manual_seed(31)
    dev = "cuda"
    H = 64
    scale = 1.0 / math.sqrt(HEAD_DIM)
    build = _build_inline_cache if layout == "inline528" else _build_separate_cache
    swa_cache, swa_scale, swa_K = build(swa_topk + 40, 64, dev)
    extra_cache, extra_scale, extra_K = build(extra_topk + 40, 16, dev)

    q = torch.randn(T, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    swa_lens = torch.randint(0, swa_topk + 1, (T,), dtype=torch.int32, device=dev)
    extra_lens = torch.randint(0, extra_topk + 1, (T,), dtype=torch.int32, device=dev)
    swa_lens[0] = swa_topk
    extra_lens[0] = extra_topk
    if T > 1:
        swa_lens[1] = 0
        extra_lens[1] = 1
    swa_idx = torch.randint(0, swa_K.shape[0], (T, swa_topk), dtype=torch.int32, device=dev)
    extra_idx = torch.randint(0, extra_K.shape[0], (T, extra_topk), dtype=torch.int32, device=dev)
    swa_idx[0, 2] = -1
    swa_idx[0, 5] = swa_K.shape[0] + 5
    extra_idx[0, 1] = -1
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _ref(q, swa_K, swa_idx, swa_lens, extra_K, extra_idx, extra_lens, scale, sink)
    out = decode(
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
        f"native int8 decode cos_diff={cd:.2e} "
        f"(layout={layout} topk={swa_topk}+{extra_topk} T={T})"
    )
    torch.testing.assert_close(out.float(), O_ref, rtol=2e-2, atol=2e-2)

    tri = sparse_mla_decode_int8_triton(
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
    # Looser cross-kernel bound: the Triton decode ALSO quantizes Q to int8
    # (s8 x s8 IMMA), while the native path keeps Q bf16; both match the fp32
    # oracle < 8e-5 individually, but their errors are independent.
    cd_tri = cos_diff(out.float(), tri.float())
    assert cd_tri < 2e-4, f"native vs Triton int8 decode cos_diff={cd_tri:.2e}"


@pytest.mark.parametrize("layout", ["inline528", "separate"])
def test_int8_decode_swa_only_parity(layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    decode = _native_int8_decode()

    torch.manual_seed(33)
    dev = "cuda"
    T, H, topk = 2, 64, 512
    scale = 1.0 / math.sqrt(HEAD_DIM)
    build = _build_inline_cache if layout == "inline528" else _build_separate_cache
    cache, cache_scale, K = build(topk + 32, 64, dev)
    q = torch.randn(T, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.tensor([0, topk], dtype=torch.int32, device=dev)
    idx = torch.randint(0, K.shape[0], (T, topk), dtype=torch.int32, device=dev)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _ref(q, K, idx, lens, None, None, None, scale, sink)
    out = decode(q, cache, cache_scale, idx, lens, scale=scale, attn_sink=sink)
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"int8 swa-only decode cos_diff={cd:.2e} (layout={layout})"
    torch.testing.assert_close(out.float(), O_ref, rtol=2e-2, atol=2e-2)


def test_int8_decode_large_T_single_split():
    """Large decode batches drive num_splits to 1; the int8 path must still
    route through the (forced) selection-scratch pre-pass and stay correct."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    decode = _native_int8_decode()

    torch.manual_seed(37)
    dev = "cuda"
    T, H, topk = 64, 64, 512
    scale = 1.0 / math.sqrt(HEAD_DIM)
    cache, cache_scale, K = _build_inline_cache(topk + 32, 64, dev)
    q = torch.randn(T, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.randint(1, topk + 1, (T,), dtype=torch.int32, device=dev)
    idx = torch.randint(0, K.shape[0], (T, topk), dtype=torch.int32, device=dev)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _ref(q, K, idx, lens, None, None, None, scale, sink)
    out = decode(q, cache, cache_scale, idx, lens, scale=scale, attn_sink=sink)
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"int8 decode large-T cos_diff={cd:.2e}"
    torch.testing.assert_close(out.float(), O_ref, rtol=2e-2, atol=2e-2)


def test_int8_decode_real_vllm_index_shapes():
    """vLLM hands [T, 1, topk] indices (singleton head dim) and int32 lens;
    the wrapper must flatten exactly like the fp8 decode wrapper does."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    decode = _native_int8_decode()

    torch.manual_seed(41)
    dev = "cuda"
    T, H, topk = 3, 64, 256
    scale = 1.0 / math.sqrt(HEAD_DIM)
    cache, cache_scale, K = _build_inline_cache(topk + 16, 64, dev)
    q = torch.randn(T, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.full((T,), topk, dtype=torch.int32, device=dev)
    idx = torch.randint(0, K.shape[0], (T, topk), dtype=torch.int32, device=dev)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    expected = decode(q, cache, cache_scale, idx, lens, scale=scale, attn_sink=sink)
    actual = decode(
        q, cache, cache_scale, idx.unsqueeze(1), lens, scale=scale, attn_sink=sink
    )
    cd = cos_diff(actual.float(), expected.float())
    assert cd < 8e-5, f"[T,1,topk] index shape changed int8 decode output: cos_diff={cd:.2e}"
