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
- **P2 SPLIT-KV (flash-decoding) — THE WIN. 121 us @ topk=512, 136 us @ topk=1024.** BEATS the
  production Triton decode (~217 us) by **1.8x**. cos<8e-5. (commit 81a7b23, shipped.)
  Each (token, head-block, split) CTA processes a slice of the concatenated swa+extra selection
  and writes a PARTIAL (un-normalized acc + running max m + denom l); a combine kernel merges
  the splits per (token, head) via log-sum-exp and applies attn_sink once. `num_splits` auto-
  sized to ~2*SM_count, capped to >= ~64 slots/split. This is the same correct K-share inner
  loop, re-parallelized — filling the SMs at T=1 is what mattered (NOT tensor cores).
- **P3 int8 KV:** NOT a memory win here — the cache is already fp8 (8-bit). Skip.
- **P2 squeeze (post-split-KV tuning) — FINAL 59.8 us @ topk=512, 87.9 us @ topk=1024 (3.6x vs Triton):**
  - vectorized fp8 gather (`cvt_fp8x2` software path on sm_86 + direct bf16 RoPE copy): 121->69 us.
  - `BLOCK_N` 32->16 (16KB K_s -> ~100% occupancy): minor (68 us; helps more at large topk).
  - `BLOCK_H` 8->16 heads/CTA: 68->60 us. KEY: fewer head-blocks => less REDUNDANT fp8 decode
    (each slot was decoded once per head-block; 8 blocks=8x, 4 blocks=4x). fp8 decode is real ALU
    on Ampere (software cvt). BLOCK_H=32 REGRESSED (106 us — 2 blocks + 1024 thr -> 1 CTA/SM, too
    few CTAs). Sweet spot = 16.
  - `num_splits` cap = ~32 slots/split (16 splits @ topk=512, ~1 wave). Finer (16 slots/split)
    REGRESSED (66 us — combine/oaccum global traffic outweighs extra parallelism).
  - Full journey: P1 2917 -> K-share 850 -> split-KV 121 -> vec-gather 69 -> BLOCK_N16 68 ->
    BLOCK_H16 59.8 -> bf16-oaccum 56.5 -> **fused-combine 53.6 us @ topk=512** (78.6 @ topk=1024).
  - **bf16 oaccum (commit ad3c8c1):** un-normalized acc written as bf16 (not fp32) — halves the
    oaccum global write/read traffic. 59.8->56.5 us. cos still < 8e-5 (the partial accs are small,
    bf16 mantissa is enough since the final combine re-normalizes in fp32).
  - **fused combine (commit 4ee3cd2):** instead of a 2nd kernel launch reading every split's
    partial, the split CTAs share an atomic counter per (token, head_block); after `__threadfence()`
    the LAST split CTA to arrive runs the log-sum-exp reduction (`combine_one_head`) in-kernel and
    writes the bf16 output. Eliminates the 2nd launch + the kernel-relaunch oaccum round-trip.
    56.5->53.6 us. 4/4 parity green (atomic + threadfence give the needed cross-CTA ordering:
    every split's oaccum/mlse write is visible to the last CTA before it reduces).
  - **cp.async double-buffered gather (commit 0f841b9):** each tile's raw fp8+rope cache bytes are
    `cp.async`-staged into a 2-deep smem ring while the previous tile computes; the fp8 decode then
    reads smem, off the global-load-latency path. Neutral at topk=512 (49.4 us — only ~2 tiles/split
    at the 32-slot sweet spot, so little to overlap) but **78.5 -> 74.4 us at topk=1024** (~5%, more
    tiles => more overlap). 4/4 parity green. A num_splits sweep with the pipeline confirms 32
    slots/split is still the sweet spot (coarser splits improve under prefetch — 64@1024: 84.5->73.5 —
    but stay slower than 32). One OOB found+fixed via compute-sanitizer: `issue_copies` was passed the
    tile index instead of the buffer parity, indexing `raw_s[2+]` past the 2-deep ring.
  - **num_splits is env-tunable (commit a7bbae8):** `FLASH_MLA_SLOTS_PER_SPLIT` (default 32) lets the
    cluster retune per real serving shape without a rebuild. Sweep on A5000: 16->54.6, 24->59.9,
    **32->49.4**, 48->64.2, 64->80.4, 96->104.1 us @ topk=512. 32 is the floor.
  - Full journey: P1 2917 -> K-share 850 -> split-KV 121 -> vec-gather 69 -> BLOCK_N16 68 ->
    BLOCK_H16 59.8 -> bf16-oaccum 56.5 -> fused-combine 53.6 -> **cp.async 49.4 us @ topk=512 /
    74.4 us @ topk=1024** (vs Triton ~217 -> **4.4x / 2.9x**). Parameter space fully mapped.
  - Remaining ideas (genuine diminishing returns, kernel at its floor for T=1): tensor cores become
    worth it only at batched/MTP decode where M grows (the flash-2 mma kernel is preserved in git
    history for that). Next highest-value step is integration (P4), not more kernel micro-tuning.
- **PREFILL — same kernel, num_splits==1 fast path (commit ccce5f9).** Sparse-MLA prefill is the
  identical absorbed attention as decode (V==K, head_dim 512, attn_sink merged once; causality is
  already encoded in the per-query selected indices — no extra causal mask, confirmed against
  upstream `reference_sparse_mla_prefill`). The only difference is many query tokens (large T). The
  decode kernel already loops over T tokens each with their own selection; at large T `num_splits`
  auto-collapses to 1. Added a `num_splits==1` fast path: each (token, head-block) CTA normalizes +
  applies sink + writes output in-register, skipping the split-KV oaccum/mlse/combine; the binding
  skips those allocations (oaccum alone was T*H*512 bf16 = **134 MB at T=2048**). Correct at every T
  (cos ~3e-6). `flash_sparse_mla_prefill` interface wrapper + 2 prefill parity tests (T=256/1024,
  variable lens). Throughput **~24.6 us/token** at large T (vs 49 us at T=1 where split-KV fills the
  SMs). BLOCK_H=32 tested for prefill (cuts redundant fp8 decode 4x->2x) — **no win** (24.4 us/tok),
  so prefill is NOT decode-ALU-redundancy bound; further speedup needs batched tensor-core QK across
  query tokens (the preserved flash-2 mma path), a separate project.
- **FEATURES validated (6/6 tests):** swa-only, two-stream swa+extra (distinct block sizes),
  attn_sink, MTP/multi-token (s_q=2), variable per-token lens, fp8_ds_mla decode on sm_86, PREFILL
  (large T, num_splits==1 direct-write fast path).
- **P4 integrate:** env-gated dispatch in `nvidia_sm86._forward_decode` (the gather-then-call shim)
  + cluster 3090 validation — the remaining step to land the 3.6x in serving.

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
