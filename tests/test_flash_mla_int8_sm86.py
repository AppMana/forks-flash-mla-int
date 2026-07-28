"""Parity for the fused int8 FlashMLA sm_86 decode kernel:
flash_mla_with_kvcache_int8 must match the exact fused math (q_i8 @ k_i8 ->
s32 -> *q_scale*k_scale -> softmax(bf16) -> p @ dequant(v_i8)) to near-machine
precision, and track fp32 within cos_diff < 8e-5 (the bf16 kernel's bound).
"""
import math
import torch
import triton
import pytest

from flash_mla import get_mla_metadata, flash_mla_with_kvcache_int8


def quant_rowwise_sym_int8(x, dim):
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(dim)


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def _int8_oracle(q, blocked_k, k_scale, v_scale, block_table, cache_seqlens, dv, softmax_scale):
    """Exact math the fused int8 kernel implements, in plain torch (per batch row)."""
    b, s_q, h_q, d = q.shape
    block_size = blocked_k.shape[1]
    out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=q.device)
    q_i8, q_scale = quant_rowwise_sym_int8(q.float(), dim=-1)        # (b,s_q,h), scale (b,s_q,h)
    for i in range(b):
        sk = cache_seqlens[i].item()
        # gather this row's K/V int8 + scales in seq order via block_table
        nblk = (sk + block_size - 1) // block_size
        blk = block_table[i, :nblk]
        kg = blocked_k[blk].reshape(-1, blocked_k.shape[2], d)[:sk, 0, :]      # (sk, d) int8
        ks = k_scale[blk].reshape(-1)[:sk]                                     # (sk,)
        vs = v_scale[blk].reshape(-1)[:sk]                                     # (sk,)
        qk_int = torch.einsum("qhd,kd->hqk", q_i8[i].float(), kg.float())      # s32 accum
        qsc = q_scale[i].permute(1, 0).unsqueeze(-1)                           # (h,s_q,1)
        score = qk_int * qsc * ks.view(1, 1, -1) * softmax_scale
        p = torch.softmax(score, dim=-1).bfloat16().float()                    # PV in bf16
        v_deq = (kg[:, :dv].float() * vs.view(-1, 1)).bfloat16().float()       # int8->bf16
        out[i] = torch.einsum("hqk,kv->qhv", p, v_deq)
    return out


@pytest.mark.parametrize("h_q", [64])
@pytest.mark.parametrize("seqlen", [512, 4096])
@pytest.mark.parametrize("s_q", [1])
def test_int8_decode_parity(h_q, seqlen, s_q):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    device = "cuda"
    b, h_kv, d, dv = 1, 1, 576, 512
    block_size = 32
    softmax_scale = 1.0 / math.sqrt(d)
    max_pad = triton.cdiv(seqlen, 256) * 256

    cache_seqlens = torch.full((b,), seqlen, dtype=torch.int32, device=device)
    q = torch.randn(b, s_q, h_q, d, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(b * max_pad // block_size, dtype=torch.int32,
                               device=device).view(b, max_pad // block_size)
    k_f = torch.randn(block_table.numel(), block_size, h_kv, d, device=device, dtype=torch.float32)
    # quantize cache rowwise per kv token over d (K) and over dv (V)
    k_i8, k_scale = quant_rowwise_sym_int8(k_f, dim=-1)                       # (...,h_kv)
    _, v_scale = quant_rowwise_sym_int8(k_f[..., :dv], dim=-1)
    k_scale = k_scale.squeeze(-1)                                             # (blk, block_size)
    v_scale = v_scale.squeeze(-1)

    O_ref = _int8_oracle(q, k_i8, k_scale, v_scale, block_table, cache_seqlens, dv, softmax_scale)

    meta, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)
    out, _ = flash_mla_with_kvcache_int8(
        q, k_i8, k_scale, v_scale, block_table, cache_seqlens, dv, meta, num_splits,
        softmax_scale=softmax_scale, causal=True,
    )
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"int8 kernel vs int8 oracle cos_diff={cd:.2e} (h={h_q} sk={seqlen})"
