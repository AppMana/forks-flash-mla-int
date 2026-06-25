# Ampere (sm_86) CUDA sparse-MLA decode kernel — design

**Status:** new kernel project. No public Ampere CUDA sparse-MLA decode exists — DeepSeek's
sparse FlashMLA is SM90 (WGMMA) / SM100 (tcgen05) only; every Ampere fork ports only the
*dense* kernel; cross-arch support is a TileLang/Triton reference (vLLM #30644, #35021). This
is the first Ampere CUDA implementation, written from the algorithm (not a port of the Hopper
WGMMA kernel, whose core instructions have no Ampere equivalent).

## Exact algorithm (matches `nvidia_sm86/triton_kernels._decode_sparse_attention_fp8_kernel`)

DeepSeek-V4-Flash decode is *compressed + sparse* MLA in the **absorbed** form:
- `head_dim = 512` (`DEEPSEEK_V4_MLA_HEAD_DIM`); per decode token, **V = K** (the kv_lora).
- Attend to only the indexer-selected top-k tokens (two index streams: `swa` + `extra`).

Per (decode_token t, head h in [0,H)):
```
scores[h,j] = (q[t,h,:] · K[idx_j,:]) * softmax_scale          # over the selected j
p = softmax(scores + attn_sink[h])                            # online, exp2
out[t,h,:] = Σ_j p[h,j] * K[idx_j,:]                          # V = K, dv = 512
```

### fp8_ds_mla cache byte-layout (the kernel must read this exactly)
Paged `uint8` cache, block = `[block_size × TOKEN_DATA_SIZE data]` then `[block_size × SCALE_DIM scales]`:
- `FP8_DIM = 448` — NoPE dims, **fp8e4m3** (1 byte each).
- `ROPE_DIM = 64` — RoPE dims, **bf16** (2 bytes each) → 128 bytes.
- `TOKEN_DATA_SIZE = 448 + 64*2 = 576` bytes (== `TOKEN_BYTES`).
- `SCALE_DIM = 8` — per-token scale bytes: `SCALE_GROUP = 64` → 448/64 = **7 UE8M0 scales** (+1 pad).
- token i in block b: data at `b*stride + i*576`; scales at `b*stride + block_size*576 + i*8`.
- Dequant NoPE dim d (d<448): `fp8e4m3_to_f32(byte) * exp2(scale[d//64] - 127)`. RoPE: bf16 as-is.
- The reconstructed K row is `[448 dequant-fp8 ; 64 bf16] = 512` bf16.

### Index streams
For each of the up to `swa_topk + extra_topk` selected positions j: an int32 slot index into
the corresponding cache (`-1` / `>= lens` = masked, contributes 0). `swa_lens[t]`, `extra_lens[t]`
give the valid counts. `extra` present only when `compress_ratio>1` (else swa-only).

## Ampere implementation plan (structural template: `flash_fwd_mla_kernel_sm80.h`)

This is a flash-attention decode; reuse the dense kernel's online-softmax + cp.async + kP-smem
patterns. The NEW seam vs the dense kernel is the **K load**: contiguous `block_table` →
**sparse-index gather + in-kernel fp8 dequant**. And `V == K`, `head_dim 512`.

Grid `(num_decode_tokens, cdiv(H, BLOCK_H))`, BLOCK_H heads/CTA (8 like Triton; pad to 16 for MMA).
Per CTA:
1. Load `q[BLOCK_H, 512]` bf16 → smem/regs.
2. `e_max = attn_sink[h]*log2e or -inf`; `e_sum = (sink?1:0)`; `acc[BLOCK_H,512] = 0` (f32).
3. For each tile of `BLOCK_N` selected positions (swa then extra):
   a. Each thread reads its positions' int32 index → slot base; `cp.async` the fp8(448)+bf16(64)
      bytes → smem; dequant (fp8→f32 × UE8M0 scale, rope bf16) → `K[BLOCK_N,512]` bf16 in smem.
   b. `scores[BLOCK_H,BLOCK_N] = q @ Kᵀ * scale_log2`; mask invalid j → -inf.
   c. online softmax (running max/sum, rescale acc) — exp2.
   d. `acc += p @ K`  (V == K).
4. `out[BLOCK_H,512] = acc / max(e_sum, eps)` → bf16.

### Phasing (TDD red→green per phase, validated vs the torch oracle)
- **P0 (now):** torch oracle (reuse `_write_fp8_ds_mla_token`) + red test + C++ API + params + a
  STUB kernel that fails the numeric assertion.
- **P1 correctness:** a straightforward kernel — sparse gather + fp8 dequant + softmax + PV with
  **fp32 FMA** math (no tensor cores). Goal: cos < the oracle bound vs fp32, ANY speed. Proves the
  gather/dequant/softmax dataflow in isolation from MMA layout complexity.
- **P2 tensor cores:** replace QK/PV with `mma.sync m16n8k16` bf16 (pad BLOCK_H 8→16); smem swizzle
  + `cp.async` double-buffer (kP=2) within the 100KB budget; vectorized fp8 dequant.
- **P3 int8 KV (the bf16-Q + int8-KV win):** read int8 NoPE instead of fp8 (per-token scale),
  halving K traffic + smem; bf16 Q, bf16 PV. Needs the int8 cache writer (separate).
- **P4 integrate + bench:** env-gated dispatch in `nvidia_sm86`, bench vs the Triton decode on a 3090.

## API
`csrc/flash_sparse_mla_decode_sm80.{h,cu}`; binding `flash_mla_cuda.fwd_sparse_decode_mla` (stable
ABI, via `run_sparse_mla_decode` in a dispatch .cu so `flash_api.cpp` stays cutlass-free). Python:
`flash_sparse_mla_decode(q, swa_cache, swa_indices, swa_lens, extra_cache, extra_indices,
extra_lens, scale, attn_sink) -> out`.

## Oracle / tests
`tests/test_sparse_mla_decode_sm86.py`: builds a realistic fp8_ds_mla cache via the
`_write_fp8_ds_mla_token` layout, random top-k indices/lens, computes the fp32 reference, and
asserts the CUDA kernel matches (cos < 8e-5 for fp8; looser for int8). A second case cross-checks
against `decode_sparse_attention_triton` (same inputs → same output).
```
