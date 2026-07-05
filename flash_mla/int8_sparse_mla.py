from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634
HEAD_DIM = 512


def quantize_int8_ds_mla_rows(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rowwise symmetric INT8 cache rows for absorbed sparse MLA.

    This is the experimental ``int8_ds_mla`` row format for the sparse path:
    ``int8[512]`` plus one fp32 scale per token. Sparse MLA has ``V == K`` in
    the absorbed form, so the same row and scale are used for QK and PV.
    """
    if k.shape[-1] != HEAD_DIM:
        raise ValueError(f"expected last dim {HEAD_DIM}, got {k.shape[-1]}")
    scale = k.float().abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(k.float() / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


@triton.jit
def _sparse_int8_mla_kernel(
    q_ptr,
    swa_cache_ptr,
    swa_scale_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_ptr,
    extra_scale_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    sink_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    swa_topk: tl.constexpr,
    extra_topk: tl.constexpr,
    swa_num_blocks: tl.constexpr,
    extra_num_blocks: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_swa_cb: tl.constexpr,
    stride_swa_cp: tl.constexpr,
    stride_swa_cd: tl.constexpr,
    stride_swa_sb: tl.constexpr,
    stride_swa_sp: tl.constexpr,
    stride_swa_it: tl.constexpr,
    stride_swa_ik: tl.constexpr,
    stride_extra_cb: tl.constexpr,
    stride_extra_cp: tl.constexpr,
    stride_extra_cd: tl.constexpr,
    stride_extra_sb: tl.constexpr,
    stride_extra_sp: tl.constexpr,
    stride_extra_it: tl.constexpr,
    stride_extra_ik: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    sm_scale_log2: tl.constexpr,
    log2e: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    token = tl.program_id(0)
    hblk = tl.program_id(1)
    heads = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    mask_h = heads < num_heads

    q = tl.load(
        q_ptr + token * stride_qt + heads[:, None] * stride_qh + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.float32)
    q_amax = tl.maximum(tl.max(tl.abs(q), 1), 1e-12)
    q_scale = q_amax / 127.0
    qn = q / q_scale[:, None]
    q_i8 = (qn + tl.where(qn >= 0, 0.5, -0.5)).to(tl.int8)

    if HAS_SINK:
        sink = tl.load(sink_ptr + heads, mask=mask_h, other=-float("inf"))
        e_max = sink * log2e
        e_sum = tl.where(mask_h, 1.0, 0.0)
    else:
        e_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    swa_len = tl.load(swa_lens_ptr + token)
    extra_len = tl.load(extra_lens_ptr + token) if HAS_EXTRA else 0
    total_len = swa_len + extra_len

    for start in range(0, swa_topk + extra_topk, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        use_swa = offs_n < swa_len
        use_extra = HAS_EXTRA & (offs_n >= swa_len) & (offs_n < total_len)
        swa_cols = offs_n
        extra_cols = offs_n - swa_len

        swa_idx = tl.load(
            swa_indices_ptr + token * stride_swa_it + swa_cols * stride_swa_ik,
            mask=swa_cols < swa_topk,
            other=-1,
        )
        extra_idx = tl.load(
            extra_indices_ptr + token * stride_extra_it + extra_cols * stride_extra_ik,
            mask=HAS_EXTRA & (extra_cols >= 0) & (extra_cols < extra_topk),
            other=-1,
        )
        idx = tl.where(use_extra, extra_idx, swa_idx)

        swa_block = idx // swa_block_size
        swa_pos = idx - swa_block * swa_block_size
        extra_block = idx // extra_block_size
        extra_pos = idx - extra_block * extra_block_size
        valid_swa = use_swa & (idx >= 0) & (swa_block < swa_num_blocks)
        valid_extra = use_extra & (idx >= 0) & (extra_block < extra_num_blocks)
        valid = valid_swa | valid_extra

        k_swa = tl.load(
            swa_cache_ptr
            + swa_block[:, None] * stride_swa_cb
            + swa_pos[:, None] * stride_swa_cp
            + offs_d[None, :] * stride_swa_cd,
            mask=valid_swa[:, None],
            other=0,
        )
        if HAS_EXTRA:
            k_extra = tl.load(
                extra_cache_ptr
                + extra_block[:, None] * stride_extra_cb
                + extra_pos[:, None] * stride_extra_cp
                + offs_d[None, :] * stride_extra_cd,
                mask=valid_extra[:, None],
                other=0,
            )
            k_i8 = tl.where(use_extra[:, None], k_extra, k_swa).to(tl.int8)
            kscale_extra = tl.load(
                extra_scale_ptr + extra_block * stride_extra_sb + extra_pos * stride_extra_sp,
                mask=valid_extra,
                other=0.0,
            )
        else:
            k_i8 = k_swa.to(tl.int8)
            kscale_extra = tl.zeros((BLOCK_N,), dtype=tl.float32)

        kscale_swa = tl.load(
            swa_scale_ptr + swa_block * stride_swa_sb + swa_pos * stride_swa_sp,
            mask=valid_swa,
            other=0.0,
        )
        kscale = tl.where(use_extra, kscale_extra, kscale_swa)

        qk_i32 = tl.dot(q_i8, tl.trans(k_i8), out_dtype=tl.int32)
        qk = qk_i32.to(tl.float32) * (q_scale[:, None] * kscale[None, :]) * sm_scale_log2
        qk = tl.where(mask_h[:, None] & valid[None, :], qk, -float("inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        p = tl.where(mask_h[:, None] & valid[None, :], p, 0.0)
        v = (k_i8.to(tl.float32) * kscale[:, None]).to(tl.bfloat16)
        acc = acc * re[:, None] + tl.dot(p.to(tl.bfloat16), v)
        e_sum = e_sum * re + tl.sum(p, 1)
        e_max = n_e_max

    acc = acc / tl.maximum(e_sum, 1e-20)[:, None]
    tl.store(
        out_ptr + token * stride_ot + heads[:, None] * stride_oh + offs_d[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=mask_h[:, None],
    )


def sparse_int8_mla_decode(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_scale: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_cache: Optional[torch.Tensor] = None,
    extra_scale: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lens: Optional[torch.Tensor] = None,
    block_h: Optional[int] = None,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    if q.shape[-1] != HEAD_DIM:
        raise ValueError(f"expected q head dim {HEAD_DIM}, got {q.shape[-1]}")
    if swa_indices.dim() == 3:
        if swa_indices.shape[1] != 1:
            raise ValueError(f"expected singleton sparse index head dim, got {swa_indices.shape}")
        swa_indices = swa_indices[:, 0]
    if extra_indices is not None and extra_indices.dim() == 3:
        if extra_indices.shape[1] != 1:
            raise ValueError(f"expected singleton sparse index head dim, got {extra_indices.shape}")
        extra_indices = extra_indices[:, 0]

    has_extra = extra_cache is not None and extra_scale is not None and extra_indices is not None and extra_lens is not None
    if scale is None:
        scale = 1.0 / math.sqrt(HEAD_DIM)
    if block_h is None:
        block_h = 16 if q.shape[0] <= 4 else 32
    if block_n is None:
        block_n = 32 if has_extra else 64
    if not has_extra:
        extra_cache = swa_cache
        extra_scale = swa_scale
        extra_indices = swa_indices[:, :1]
        extra_lens = swa_lens

    assert extra_cache is not None
    assert extra_scale is not None
    assert extra_indices is not None
    assert extra_lens is not None

    out = torch.empty_like(q)
    grid = (q.shape[0], triton.cdiv(q.shape[1], block_h))
    _sparse_int8_mla_kernel[grid](
        q,
        swa_cache,
        swa_scale,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_scale,
        extra_indices,
        extra_lens,
        attn_sink if attn_sink is not None else q,
        out,
        q.shape[0],
        q.shape[1],
        swa_indices.shape[-1],
        extra_indices.shape[-1] if has_extra else 0,
        swa_cache.shape[0],
        extra_cache.shape[0],
        swa_cache.shape[1],
        extra_cache.shape[1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        swa_cache.stride(0),
        swa_cache.stride(1),
        swa_cache.stride(2),
        swa_scale.stride(0),
        swa_scale.stride(1),
        swa_indices.stride(0),
        swa_indices.stride(1),
        extra_cache.stride(0),
        extra_cache.stride(1),
        extra_cache.stride(2),
        extra_scale.stride(0),
        extra_scale.stride(1),
        extra_indices.stride(0),
        extra_indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        float(scale) * LOG2E,
        LOG2E,
        HEAD_DIM,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        HAS_EXTRA=has_extra,
        HAS_SINK=attn_sink is not None,
        num_warps=8,
        num_stages=3,
    )
    return out


def sparse_int8_mla_prefill_native_staged(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_scale: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_cache: Optional[torch.Tensor] = None,
    extra_scale: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q.shape[-1] != HEAD_DIM:
        raise ValueError(f"expected q head dim {HEAD_DIM}, got {q.shape[-1]}")
    if scale is None:
        scale = 1.0 / math.sqrt(HEAD_DIM)
    if swa_indices.dim() == 3:
        if swa_indices.shape[1] != 1:
            raise ValueError(f"expected singleton sparse index head dim, got {swa_indices.shape}")
        swa_indices = swa_indices[:, 0]
    if extra_indices is not None and extra_indices.dim() == 3:
        if extra_indices.shape[1] != 1:
            raise ValueError(f"expected singleton sparse index head dim, got {extra_indices.shape}")
        extra_indices = extra_indices[:, 0]
    if not hasattr(torch.ops.flash_mla, "fwd_sparse_int8_prefill_staged_mla"):
        raise NotImplementedError(
            "torch.ops.flash_mla.fwd_sparse_int8_prefill_staged_mla is not built yet"
        )
    return torch.ops.flash_mla.fwd_sparse_int8_prefill_staged_mla(
        q,
        swa_cache,
        swa_scale,
        swa_indices,
        swa_lens,
        float(scale),
        attn_sink,
        extra_cache,
        extra_scale,
        extra_indices,
        extra_lens,
    )
