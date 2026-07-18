"""Adversarial parity for the native sparse-MLA prefill paths (sm_86).

Stresses what the basic prefill tests do not: ragged per-token lens including 0,
lens that straddle BLOCK_N tile boundaries, T that is not a multiple of any
internal chunking, topk 512/1024 production widths, and out-of-range indices.
Oracle: fp32 torch over the expected bf16 K rows (cache written with the exact
fp8_ds_mla byte layout).

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_sparse_mla_prefill_adversarial.py -x -q
"""
import math

import pytest
import torch

from test_sparse_mla_decode_sm86 import (
    _HEAD_DIM,
    _SCALE_DIM,
    _TOKEN_DATA_SIZE,
    _build_cache,
    cos_diff,
)


def _ref(q, swa_K, swa_idx, swa_lens, extra_K, extra_idx, extra_lens, scale, sink):
    T, H, D = q.shape
    out = torch.zeros(T, H, D, device=q.device, dtype=torch.float32)
    for t in range(T):
        ns = int(swa_lens[t])
        ne = int(extra_lens[t]) if extra_lens is not None else 0
        rows = []
        if ns:
            sl = swa_idx[t, :ns].long()
            ok = (sl >= 0) & (sl < swa_K.shape[0])
            r = torch.zeros(ns, D, device=q.device, dtype=torch.float32)
            r[ok] = swa_K[sl[ok]].float()
            rows.append(r)
        if ne:
            el = extra_idx[t, :ne].long()
            ok = (el >= 0) & (el < extra_K.shape[0])
            r = torch.zeros(ne, D, device=q.device, dtype=torch.float32)
            r[ok] = extra_K[el[ok]].float()
            rows.append(r)
        sk = sink[:, None].float()
        if not rows:
            out[t] = 0.0
            continue
        K = torch.cat(rows)
        sc = (q[t].float() @ K.t()) * scale
        m = torch.maximum(sc.max(-1, keepdim=True).values, sk)
        ex = torch.exp(sc - m)
        out[t] = (ex @ K) / (ex.sum(-1, keepdim=True) + torch.exp(sk - m))
    return out


@pytest.mark.parametrize("swa_topk,extra_topk", [(256, 256), (512, 512)])
def test_prefill_adversarial_parity(swa_topk, extra_topk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    from flash_mla import sparse_mla_prefill

    torch.manual_seed(7)
    dev = "cuda"
    T = 67  # not a multiple of anything internal
    H = 64
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    swa_cache, swa_K = _build_cache(swa_topk + 40, 32, dev)
    extra_cache, extra_K = _build_cache(extra_topk + 40, 16, dev)

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    # ragged lens: include 0, 1, exact tile multiples, off-by-one around 16/32/64
    specials = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, swa_topk]
    swa_lens = torch.randint(0, swa_topk + 1, (T,), dtype=torch.int32, device=dev)
    extra_lens = torch.randint(0, extra_topk + 1, (T,), dtype=torch.int32, device=dev)
    for i, v in enumerate(specials):
        swa_lens[i] = min(v, swa_topk)
        extra_lens[-1 - i] = min(v, extra_topk)
    swa_idx = torch.randint(0, swa_K.shape[0], (T, swa_topk), dtype=torch.int32, device=dev)
    extra_idx = torch.randint(0, extra_K.shape[0], (T, extra_topk), dtype=torch.int32, device=dev)
    # sprinkle invalid indices inside the valid range of lens
    swa_idx[3, 2] = -1
    swa_idx[4, 0] = swa_K.shape[0] + 5
    extra_idx[5, 1] = -1
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _ref(q, swa_K, swa_idx, swa_lens, extra_K, extra_idx, extra_lens, scale, sink)
    out = sparse_mla_prefill(
        q=q, swa_cache=swa_cache, swa_indices=swa_idx, swa_lens=swa_lens,
        scale=scale, attn_sink=sink,
        extra_cache=extra_cache, extra_indices=extra_idx, extra_lens=extra_lens,
    )
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"prefill adversarial cos_diff={cd:.2e} (topk={swa_topk}+{extra_topk})"
    # elementwise check at the production tolerance convention
    torch.testing.assert_close(out.float(), O_ref, rtol=2e-2, atol=2e-2)


def test_prefill_swa_only_parity():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    from flash_mla import sparse_mla_prefill

    torch.manual_seed(11)
    dev = "cuda"
    T, H, topk = 33, 64, 512
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    cache, K = _build_cache(topk + 32, 64, dev)
    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.randint(0, topk + 1, (T,), dtype=torch.int32, device=dev)
    lens[0] = 0
    lens[1] = topk
    idx = torch.randint(0, K.shape[0], (T, topk), dtype=torch.int32, device=dev)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _ref(q, K, idx, lens, None, None, None, scale, sink)
    out = sparse_mla_prefill(
        q=q, swa_cache=cache, swa_indices=idx, swa_lens=lens,
        scale=scale, attn_sink=sink,
    )
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"swa-only prefill cos_diff={cd:.2e}"
