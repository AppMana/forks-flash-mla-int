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

### Phasing + MEASURED RESULTS (A5000 sm_86, T=1 H=64 topk=512, vs live Triton ~217us)
- **P0 done:** torch oracle + red test + C++ API + params (committed).
- **P1 done (correct):** warp-per-head, fp32 FMA. **2917 us.** cos<8e-5. (commit 38f09b4)
- **P2 K-share done (correct, BEST):** gather+dequant K once into smem, reuse across heads.
  **850 us** (3.4x). cos<8e-5. (commit 67c8cd6) — THIS is the shipped kernel.
- **P2 tensor cores — DEAD END for this workload (both tried, both correct, both SLOWER):**
  - naive WMMA 16x16x16: **1463 us** (acc round-trips smem every tile; warp-0-only QK).
  - flash-2 raw `mma.sync.m16n8k16` (register-resident acc, layout-aware per-row rescale,
    cross-warp max+sum reductions): **1345 us.** Correct (manual fragment loads from the
    documented PTX operand layout; the last bug was a per-warp vs cross-warp softmax DENOM).
  - **Why tensor cores lose here:** T=1 decode has M=16 heads/CTA -> only H/16 = 4 CTAs (or 8
    at BLOCK_H=8) for the whole problem; the GPU's 64 SMs starve. M=16 is tiny for MMA (most
    of the 16x8 tile idle). The kernel is parallelism- and memory-bound, not compute-bound,
    so swapping the (already-cheap) dot for MMA doesn't help and adds smem/occupancy overhead.
- **The real lever to beat Triton = SPLIT-KV (flash-decoding), NOT tensor cores:** split the
  topk across many CTAs (e.g. 4-8 splits) -> 4-8x more CTAs -> fills the SMs -> + a combine
  pass (log-sum-exp merge of the per-split partials). Plus vectorized fp8 dequant / cp.async on
  the gather. This is the next major effort (a different parallelization of the SAME correct
  inner loop the K-share kernel already has).
- **P3 int8 KV:** NOT a memory win here — the cache is already fp8 (8-bit). Skip.
- **P4 integrate:** env-gated dispatch in `nvidia_sm86`; the gather-then-dense shim. Deferred
  until the kernel actually beats Triton (split-KV).

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
