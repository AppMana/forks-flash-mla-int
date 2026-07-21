"""Peak-memory regression guard for the fused int8_ds_mla sparse-MLA prefill (sm_86).

The int8 prefill dequantizes in-kernel (int8 rows + fp32 rowwise scale straight
into the smem gather ring), so the op must NOT allocate anything sized by the
KV pool. The historical failure mode was a whole-cache bf16 dequant buffer
[total_slots, 512] (2 KiB/slot ~= 2.23 GiB at the production 2.3M-slot pool),
which OOM'd 24GB ranks. This test runs the op against a large pool with a tiny
selection and asserts the op's peak allocation stays bounded by the selection,
not the pool.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_sparse_mla_prefill_int8_memory.py -x -q
"""
import math

import pytest
import torch

HEAD_DIM = 512
INT8_DIM = 512
TOKEN_BYTES = 528  # 512 payload + 4 fp32 scale + 12 pad


def _build_inline_cache(num_slots: int, block_size: int, dev: str):
    """Production 528B-stride inline-scale layout, filled without fp32 temps."""
    nb = (num_slots + block_size - 1) // block_size
    cache = torch.zeros(nb, block_size, TOKEN_BYTES, dtype=torch.uint8, device=dev)
    cache[:, :, :INT8_DIM] = torch.randint(
        0, 256, (nb, block_size, INT8_DIM), dtype=torch.uint8, device=dev
    )
    scales = (torch.rand(nb, block_size, device=dev) * 0.02 + 1e-4).to(torch.float32)
    cache[:, :, INT8_DIM:INT8_DIM + 4] = scales.view(nb, block_size, 1).view(torch.uint8)
    data_view = cache[:, :, :INT8_DIM].view(torch.int8)
    scale_view = cache[:, :, INT8_DIM:INT8_DIM + 4].view(torch.float32).squeeze(-1)
    return cache, data_view, scale_view


def test_int8_prefill_peak_memory_independent_of_pool_size():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    from flash_mla.int8_sparse_mla import sparse_mla_prefill_int8

    torch.manual_seed(7)
    dev = "cuda"
    T, H, topk = 8, 64, 64
    block_size = 64
    num_slots = 600_000  # pool >> selection; whole-pool bf16 buffer would be ~614 MB

    keepalive, data_view, scale_view = _build_inline_cache(num_slots, block_size, dev)
    q = torch.randn(T, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.full((T,), topk, dtype=torch.int32, device=dev)
    idx = torch.randint(0, num_slots, (T, topk), dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # warmup (op output caching allocator blocks etc.)
    sparse_mla_prefill_int8(q, data_view, scale_view, idx, lens, scale=scale)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    out = sparse_mla_prefill_int8(q, data_view, scale_view, idx, lens, scale=scale)
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline

    selection_bytes = T * topk * HEAD_DIM * 2  # every selected row as bf16
    out_bytes = out.numel() * out.element_size()
    bound = out_bytes + selection_bytes + (16 << 20)  # generous op-scratch slack
    pool_bytes = num_slots * HEAD_DIM * 2
    assert peak_delta < bound, (
        f"int8 prefill peak alloc {peak_delta / 2**20:.1f} MiB exceeds the "
        f"selection-bounded budget {bound / 2**20:.1f} MiB "
        f"(whole-pool bf16 buffer would be {pool_bytes / 2**20:.1f} MiB -- "
        f"pool-sized allocation regression)"
    )
    del keepalive
