from typing import Optional, Tuple

import torch

import flash_mla_cuda  # noqa: F401  (loads the .so -> registers torch.ops.flash_mla.*)


def _flatten_sparse_indices(indices: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if indices is None:
        return None
    if indices.dim() == 3 and indices.shape[1] == 1:
        return indices.squeeze(1)
    if indices.dim() != 2:
        raise ValueError(
            "sparse MLA indices must have shape (num_tokens, topk) or "
            "(num_tokens, 1, topk)"
        )
    return indices


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return torch.ops.flash_mla.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    warp_spec: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        warp_spec: bool. Use warp-specialized SM80 kernel (only affects SM80).

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops.flash_mla.fwd_kvcache_mla(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        warp_spec,
    )
    return out, softmax_lse


def flash_mla_with_kvcache_int8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully-fused int8 MLA decode for sm_86 (task #61).

    Q is bf16 and is quantized to int8 rowwise IN-KERNEL. K/V cache is int8
    (rowwise-symmetric, scales supplied per kv token). QK^T runs on the s8 IMMA
    tensor cores (SM80_16x8x16_F32S8S8S32_TN) and dequantizes by q_scale*k_scale;
    V is dequantized int8->bf16 in-kernel and PV runs in bf16. Storing K/V as
    int8 halves their SMEM footprint, which is what lets the kP=2 double-buffer
    fit inside sm_86's ~100KB shared-memory budget.

    Arguments:
        q: (batch, seq_len_q, num_heads_q, head_dim) bfloat16.
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim) int8.
        k_scale: (num_blocks, page_block_size, num_heads_k) float32 -- per-kv-token K scale.
        v_scale: (num_blocks, page_block_size, num_heads_k) float32 -- per-kv-token V scale.
        block_table: (batch, max_num_blocks_per_seq) int32.
        cache_seqlens: (batch,) int32.
        head_dim_v: head dim of V.
        tile_scheduler_metadata, num_splits: from get_mla_metadata.
        softmax_scale: defaults to 1/sqrt(head_dim).
        causal: causal mask.

    Returns:
        out: (batch, seq_len_q, num_heads_q, head_dim_v) bfloat16.
        softmax_lse: (batch, num_heads_q, seq_len_q) float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if not hasattr(torch.ops.flash_mla, "fwd_kvcache_mla_int8"):
        raise NotImplementedError(
            "torch.ops.flash_mla.fwd_kvcache_mla_int8 is not built yet "
            "(fused int8 sm_86 kernel, task #61)."
        )
    out, softmax_lse = torch.ops.flash_mla.fwd_kvcache_mla_int8(
        q,
        k_cache,
        k_scale,
        v_scale,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse


def sparse_mla_decode_fp8(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ampere (sm_86) CUDA sparse-MLA decode (DeepSeek-V4-Flash absorbed form).

    Attends each (decode token, head) over the indexer-selected top-k slots only;
    head_dim is 512, V == K, the paged caches are ``fp8_ds_mla`` (uint8: 448 fp8 NoPE
    with UE8M0 per-64-group scales + 64 bf16 RoPE), dequantized in-kernel.

    Arguments:
        q: (num_decode_tokens, num_heads, 512) bfloat16.
        swa_cache: (num_blocks, block_size, 584) uint8, fp8_ds_mla.
        swa_indices: (num_decode_tokens, swa_topk) int32 -- selected slot ids (-1 = pad).
        swa_lens: (num_decode_tokens,) int32 -- valid count per token.
        scale: softmax scale; defaults to 1/sqrt(512).
        attn_sink: (num_heads,) float32 or None.
        extra_cache/extra_indices/extra_lens: optional second (compressed) cache stream.

    Returns:
        out: (num_decode_tokens, num_heads, 512) bfloat16.
    """
    if scale is None:
        scale = q.shape[-1] ** (-0.5)
    if not hasattr(torch.ops.flash_mla, "fwd_sparse_decode_mla"):
        raise NotImplementedError(
            "torch.ops.flash_mla.fwd_sparse_decode_mla is not built yet "
            "(Ampere sm_86 CUDA sparse-MLA decode kernel)."
        )
    swa_indices = _flatten_sparse_indices(swa_indices)
    extra_indices = _flatten_sparse_indices(extra_indices)
    return torch.ops.flash_mla.fwd_sparse_decode_mla(
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        float(scale),
        attn_sink,
        extra_cache,
        extra_indices,
        extra_lens,
    )


def sparse_mla_prefill(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ampere (sm_86) CUDA sparse-MLA prefill (DeepSeek-V4-Flash absorbed form).

    Sparse-MLA prefill is the same absorbed attention as decode (V == K, head_dim 512,
    attn_sink merged once; causality is already encoded in the per-query selected indices),
    differing only in that ``q`` carries many query tokens.

    On Ampere this runs the native fused tensor-core kernel: the whole fp8_ds_mla cache
    is dequantized once into a dense bf16 buffer, then a gather-bf16 mma.m16n8k16
    attention kernel with a cp.async ring computes the output directly
    (csrc/sparse_mla_prefill_fused_sm80.cu). The legacy staged two-kernel gather
    path and its ``FLASH_MLA_PREFILL_FUSED`` kill-switch were removed: the fused kernel
    is the sole path (unreachable-by-production dead code, ~4x slower at the true 16k
    footprint; see README).

    Arguments mirror :func:`sparse_mla_decode_fp8`, with ``q`` shaped
    ``(num_query_tokens, num_heads, 512)`` and ``swa_indices``/``swa_lens`` carrying the
    per-query-token selection. Returns ``(num_query_tokens, num_heads, 512)`` bfloat16.
    """
    if scale is None:
        scale = q.shape[-1] ** (-0.5)
    if not hasattr(torch.ops.flash_mla, "fwd_sparse_prefill_mla"):
        raise NotImplementedError(
            "torch.ops.flash_mla.fwd_sparse_prefill_mla is not built yet"
        )
    swa_indices = _flatten_sparse_indices(swa_indices)
    extra_indices = _flatten_sparse_indices(extra_indices)
    return torch.ops.flash_mla.fwd_sparse_prefill_mla(
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        float(scale),
        attn_sink,
        extra_cache,
        extra_indices,
        extra_lens,
    )


# Deprecated aliases (pre-harmonization names; the appmana kernel-config
# symbols are now role-named: sparse_mla_prefill / sparse_mla_decode_fp8 /
# sparse_mla_decode_int8). Kept importable for older configs and callers.
flash_sparse_mla_decode = sparse_mla_decode_fp8
flash_sparse_mla_prefill = sparse_mla_prefill
