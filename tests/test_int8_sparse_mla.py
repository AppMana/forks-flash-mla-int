import math

import pytest
import torch

from flash_mla.int8_sparse_mla import (
    quantize_int8_ds_mla_rows,
    triton_sparse_int8_mla_decode,
)


HEAD_DIM = 512


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def _build_cache(rows: torch.Tensor, block_size: int):
    nblocks = (rows.shape[0] + block_size - 1) // block_size
    padded = torch.zeros(nblocks * block_size, HEAD_DIM, device=rows.device, dtype=rows.dtype)
    padded[: rows.shape[0]] = rows
    cache, scale = quantize_int8_ds_mla_rows(padded.view(nblocks, block_size, HEAD_DIM))
    return cache, scale


def _quant_q(q: torch.Tensor):
    amax = q.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    qi = torch.round(q.float() / scale).clamp(-127, 127).to(torch.int8)
    return qi, scale.squeeze(-1)


def _int8_sparse_ref(q, swa_cache, swa_scale, swa_indices, swa_lens, scale, sink,
                     extra_cache=None, extra_scale=None, extra_indices=None, extra_lens=None):
    tcount, heads, dim = q.shape
    out = torch.zeros(tcount, heads, dim, device=q.device, dtype=torch.float32)
    q_i8, q_scale = _quant_q(q)
    for t in range(tcount):
        rows = []
        scales = []
        ns = int(swa_lens[t].item())
        for slot in swa_indices[t, :ns].tolist():
            if slot < 0:
                continue
            b = slot // swa_cache.shape[1]
            p = slot % swa_cache.shape[1]
            rows.append(swa_cache[b, p].float())
            scales.append(swa_scale[b, p])
        if extra_cache is not None:
            ne = int(extra_lens[t].item())
            for slot in extra_indices[t, :ne].tolist():
                if slot < 0:
                    continue
                b = slot // extra_cache.shape[1]
                p = slot % extra_cache.shape[1]
                rows.append(extra_cache[b, p].float())
                scales.append(extra_scale[b, p])
        k_i8 = torch.stack(rows, dim=0)
        k_scale = torch.stack(scales, dim=0)
        qk_i32 = torch.einsum("hd,kd->hk", q_i8[t].float(), k_i8)
        scores = qk_i32 * q_scale[t, :, None] * k_scale[None, :] * scale
        if sink is not None:
            m = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink[:, None].float())
            p = torch.exp(scores - m)
            denom = p.sum(-1, keepdim=True) + torch.exp(sink[:, None].float() - m)
            probs = p / denom
        else:
            probs = torch.softmax(scores, dim=-1)
        v = (k_i8 * k_scale[:, None]).bfloat16().float()
        out[t] = probs.bfloat16().float() @ v
    return out


@pytest.mark.parametrize("has_extra", [False, True])
@pytest.mark.parametrize("tokens", [1, 4])
def test_triton_sparse_int8_mla_decode_matches_int8_oracle(has_extra, tokens):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere")
    torch.manual_seed(101 + tokens + int(has_extra))
    dev = "cuda"
    heads = 64
    swa_topk = 64
    extra_topk = 96
    scale = 1.0 / math.sqrt(HEAD_DIM)

    swa_rows = torch.randn(swa_topk + 32, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    swa_cache, swa_scale = _build_cache(swa_rows, block_size=32)
    q = torch.randn(tokens, heads, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    swa_lens = torch.randint(swa_topk // 2, swa_topk + 1, (tokens,), device=dev, dtype=torch.int32)
    swa_lens[0] = swa_topk
    swa_indices = torch.stack([
        torch.randperm(swa_rows.shape[0], device=dev)[:swa_topk].to(torch.int32)
        for _ in range(tokens)
    ])
    sink = torch.randn(heads, device=dev, dtype=torch.float32) * 0.1

    extra_cache = extra_scale = extra_indices = extra_lens = None
    if has_extra:
        extra_rows = torch.randn(extra_topk + 32, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        extra_cache, extra_scale = _build_cache(extra_rows, block_size=16)
        extra_lens = torch.randint(extra_topk // 2, extra_topk + 1, (tokens,), device=dev, dtype=torch.int32)
        extra_lens[0] = extra_topk
        extra_indices = torch.stack([
            torch.randperm(extra_rows.shape[0], device=dev)[:extra_topk].to(torch.int32)
            for _ in range(tokens)
        ])

    out = triton_sparse_int8_mla_decode(
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
    ref = _int8_sparse_ref(
        q, swa_cache, swa_scale, swa_indices, swa_lens, scale, sink,
        extra_cache, extra_scale, extra_indices, extra_lens,
    )
    cd = cos_diff(out.float(), ref)
    assert cd < 8e-5, f"int8 sparse MLA cos_diff={cd:.2e}"
