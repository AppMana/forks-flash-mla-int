# Ampere (sm_86) CUDA sparse-MLA decode kernel — design

**Status:** production-validated Ampere kernel. No public Ampere CUDA sparse-MLA decode existed when
this implementation was started — DeepSeek's
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
  so prefill is NOT decode-ALU-redundancy / gather-volume bound.
- **PREFILL tensor-core QK+PV (mma_pf) — correct but ~2x SLOWER; default OFF.** Recovered the flash-2
  `mma.sync.m16n8k16` kernel (BLOCK_M=16 heads/CTA, 4 warps, register-resident O accumulator, two-
  stream + sink + variable lens, direct write) and wired it as the num_splits==1 prefill path, gated
  by `FLASH_MLA_PREFILL_MMA` (env, default OFF). Correct (cos 1.78e-6, 6/6 green) but **53 us/tok vs
  FMA 24.6**. WHY it loses, and why this is a CEILING not a tuning gap: prefill is gather-MEMORY-
  LATENCY bound (24.6 us/tok = ~6% of bandwidth — the SMs sit idle waiting on the fp8 cache reads).
  The FMA path hides that latency with cp.async + 16 warps; mma_pf's 64-slot tiles cost 85 KB smem
  -> 1 CTA/SM + only 4 warps, so its gather latency is fully exposed (and 85 KB leaves no room for a
  cp.async double-buffer). Tensor cores accelerate QK/PV *compute*, which is OFF the critical path
  here, so even a perfectly cp.async'd MMA kernel could at best TIE the FMA path (both bounded by the
  same memory latency), never beat it. Same lesson as tensor-core decode: memory-bound workload, MMA
  is the wrong tool. Kept opt-in for a future QK-bound regime (much larger BLOCK_M, compute-heavier).
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

## PREFILL REWRITE (2026-07-09) — fused tensor-core kernel, beats production Triton 1.7-1.9x

The old staged prefill (`flash_sparse_mla_prefill_staged_sm80.cu`, gather-to-dense +
smem attention) LOST to the production vLLM Triton prefill (dequant_gather +
`sparse_attention_triton`) at REAL 16k-slot caches: 13.65 vs 5.74 us/tok at
T=1024/width 512 (the old ~24.6 us/tok "FMA wins / gather-latency-bound at 6% BW"
conclusion was an artifact of the 192-slot L2-resident microbench in
`bench_sparse_mla_sm86_shapes`; at production shapes the tensor-core Triton paths
are 2.4-5x faster than both native paths). Ground-up rewrite in
`csrc/flash_sparse_mla_prefill_fused_sm80.cu`, exported through the
`flash_sparse_mla_prefill` op.

**(2026-07-09, removal)** The staged kernel, its `FLASH_MLA_PREFILL_FUSED=0`
kill-switch, and the `use_staged_prefill` wrapper kwarg were DELETED. The fused
kernel is the sole sparse-MLA prefill path (fp8 and int8). The staged path was
unreachable in production (vLLM never set `use_staged_prefill`) and measured ~4x
slower at the true 16k footprint. The table below is kept as a historical record.

**The two structural insights (both from ncu):**
1. **Per-use fp8 decode is the dominant cost, not the gather or the FLOPs.** At
   prefill every cached slot is selected ~T*topk/context (~32-64)x per chunk; the
   software fp8->bf16 cvt chain on Ampere ran per (token, slot, head-block):
   97M ALU + 52M FMA instructions vs 4M HMMA. Fix: **dequantize the WHOLE cache
   once per call** into a dense bf16 [total_slots, 512] buffer (~124 us for 2x16k
   slots = 3% of the op at T=1024), then gather bf16 rows with pure cp.async —
   zero ALU on the load path. This alone: 8.05 -> 3.84 us/tok.
2. **ldmatrix, not manual fragment packing.** pack_contig/pack2 fragment builds
   (~160 LDS + PRMT per warp-tile; PV's column-wise pack2 worst) saturate the LSU
   pipe (mio_throttle 1.94). ldmatrix x4/x2 + x4.trans (PV's n-major transpose is
   free) cut fragment LDS ~4x: 3.84 -> 3.30 us/tok.

**Kernel shape:** one CTA per (token, 32-head block), 8 warps; BLOCK_N=16 ring of
2 filled by cp.async (16B chunks, 4/thread, slot index resolved via one
L1-broadcast load; invalid rows STS-zeroed); QK = mma.m16n8k16, 2-way k-split so
all 8 warps contribute (partial C exchanged through smem); PV = 2m x 4(d128) with
V==K read from the same ring; online softmax with SEPARATE row-max and row-sum
exchange buffers (sharing one buffer races: the sum write and the other wn-warp's
max read share a barrier interval); attn_sink folded into (m,l) once; direct
write. ~71KB smem, 122 regs, no spills. Grid is (head_blocks, T) so a token's two
head-block CTAs are adjacent in issue order.

**Measured (A5000, T=1024 chunk, 16k-slot caches, H=64, D=512, us/token):**

| backend                          | width 512 | width 1024 |
|----------------------------------|-----------|------------|
| native staged (old default)      | 13.65     | 24.10      |
| native fused FMA (num_splits==1) | 28.70     | 57.54      |
| triton fp8 dequant+attn (PROD)   | 5.74      | 11.54      |
| triton int8 fused IMMA           | 3.74      | 7.30       |
| **native fused (this kernel)**   | **~3.2**  | **~6.0**   |

1.7-1.9x faster than the production Triton fp8 prefill, and ahead of the int8
IMMA Triton path despite gathering 2x the bytes. Parity 24/24 green (adversarial
ragged lens incl. 0, off-tile lens, invalid indices inside lens, T=67, widths
512/1024, swa-only and two-stream) at cos < 8e-5 vs the fp32 oracle.

**Dead ends mapped (measured, one commit each):**
- BLOCK_M=16 + Q-in-register-fragments + 4-way k-split + launch_bounds(256,2):
  occupancy 16.7 -> 33% as designed, but gather DRAM doubled with 4 head-block
  passes (242 -> 520MB reads @ T=256, 70% DRAM util) — 35% SLOWER. Gather
  redundancy, not occupancy, is the binding constraint at 6MB L2.
- dropping 2 barriers/tile (parity-double-buffered softmax exchanges + inline
  resolve): flat at width 512, within noise at width 1024.
- in-CTA fp8 decode from a raw-byte cp.async ring (v1): correct but 2.4x slower
  than the whole-cache dequant split (the ALU redundancy above).

Remaining headroom (documented, not taken): barrier ~2.5 + short-scoreboard ~2.3
stalls/issue at 8 warps/SM. Pairwise named barriers for the k-split/row-max
exchanges and warp-specialized producer/consumer staging are the next levers if
another ~15-20% is ever needed; both add real complexity for gains inside the
GPU0 measurement noise band today.

## DECODE sanity check at true 16k cache footprint (2026-07-09) — NO microbench artifact

The prefill rewrite invalidated an L2-resident microbench, so decode was re-measured
at a REAL 16k-slot cache (ncu + wall clock, T=1, H=64, ctx=16384, GPU0 A5000).

Wall clock (`bench_sparse_mla_16k_matrix --decode-tokens 1`, 100 iters):
  topk=512: 43 us @ ctx 16384 vs 53 us @ ctx 1024 (compact)
  topk=1024: 83 us @ ctx 16384 vs 79 us @ ctx 1024
Identical within noise — the historical tiny-cache numbers (49.4/74.4 us) HOLD.
Why decode is immune to the artifact that broke prefill: at T=1 the working set is
topk x 576B (~0.3-0.6 MB), L2-resident regardless of total cache size; prefill's
working set is the whole cache (~10-34 MB), which is what the tiny bench hid.

ncu (sparse_mla_decode_split_kernel, profiled clocks 1.17 GHz):
  topk=512:  DRAM 3.9%, L2 hit 89.6% (619 KB DRAM reads), L1/TEX 73%, SM 58.6%,
             occupancy 33%; stalls: short_scoreboard 2.41, mio_throttle 1.60;
             pipes: ALU 2.07M + FMA 2.54M + LSU 1.42M, tensor 0.
  topk=1024: DRAM 4.2%, L2 hit 91.8%, L1/TEX 84.9%, SM 67.4%, occupancy 56%;
             stalls: mio_throttle 3.84, short_scoreboard 2.69;
             pipes: ALU 4.14M + FMA 5.08M + LSU 2.84M, tensor 0.

VERDICT: decode is genuinely bound where the decode journey concluded — the
ALU/FMA/LSU issue stream (software fp8 cvt + FMA dots + smem traffic) at small-T
parallelism, not DRAM and not an L2 artifact. The per-head-block cvt redundancy
(4x at BLOCK_H=16) exists but the prefill fix does not transfer: whole-cache
dequant of 16k rows (~26 MB traffic ~= 40-50 us) would exceed the entire 43 us
kernel at T=1. A selected-rows-only bf16 staging + ldmatrix decode is the only
plausible follow-up (mio_throttle 3.84 at topk=1024 says the L1/LSU pipe is the
ceiling there), expected O(20%) at topk=1024 — noted, not taken.

## DECODE REWRITE (2026-07-09) — selection-scratch + heads-as-M tensor cores, 2.1x/2.9x

The split-KV FMA decode above (49.4/74.4 us on the L2-resident microbench cache) was
re-baselined at TRUE footprint (16k-slot caches, 9.6+9.6 MB, `benchmarks/
bench_sparse_mla_decode_16k.py`): **47.2 us @ width 512 / 80.6 us @ width 1024** — no
microbench artifact in the totals, but ncu showed the structure differently than the
"memory-bound" story: DRAM read is only ~512 KB (minimal) while L2 read is ~3.57 MB
(the 4 head-block CTAs re-gather+re-dequant the SAME selection; the redundancy is
absorbed by L2 but its fp8-cvt ALU is not: 2.28M ALU + 2.57M FMA + 1.42M LSU insts,
0 HMMA, stalls short_scoreboard 2.30 / wait 1.62 / mio_throttle 1.52 per issue).

Rewrite (same op, env kill-switches at every layer), per-commit journey at true
footprint (us, width 512 / 1024):

| step | w512 | w1024 | what |
|------|------|-------|------|
| baseline (FMA split-KV)        | 47.2 | 80.6 | `FLASH_MLA_DECODE_FUSED=0` |
| H1 selection-scratch dequant   | 43.3 | 76.4 | dequant topk rows ONCE -> bf16 [T*width,512] scratch (~4.6 us pre-pass); attention reads clean rows, zero cvt ALU (`FLASH_MLA_DECODE_MMA=0` for this FMA variant) |
| H2 heads-as-M mma_dec          | 46.9 | 77.8 | BLOCK_M=32 heads = M, m16n8k16 QK/PV, ldmatrix + cp.async ring (port of the fused prefill CTA shape) + split-KV partials; slower until retuned |
| H3 split retune                | 39.0 | 48.4 | 8 coarse splits (fused last-CTA combine forces coarse) |
| H4 fixed costs                 | 33.0 | 42.3 | q tile cp.async'd into ring[0]'s commit group; uint4 combine; counter memset folded into the dequant pre-pass |
| H5 standalone parallel combine | 25.0 | 30.7 | one warp per head -> 16 fine splits viable; chunk must be a multiple of BLOCK_N=16 (sharp minima at chunk 32/64) |
| H6 warp-wide combine           | **22.5** | **28.3** | mlse one-split-per-lane + shfl broadcast, branchless pipelined uint4 accumulation (~6.0 -> ~2.5 us) |

**2.10x / 2.85x vs the production FMA baseline** (and ~9.5x / 15x vs the live Triton
decode: 214/423 us at this footprint). Kernel split at the end (torch profiler, us):
dequant 4.6/7.0 + mma 12.7/16.1 + combine ~2.5 + launch gaps ~2.5.

Why tensor cores won this time (vs the P2 "dead end"): (1) the 64 heads ARE the M
dimension — BLOCK_M=32 makes full m16 tiles at T=1 (the P2 attempts had M=16 heads/CTA
but kept per-CTA gather+dequant and only head_blocks CTAs, so they starved and paid the
cvt 4x); (2) the H1 scratch gives bf16 operands that `ldmatrix` can fragment directly;
(3) split-KV supplies the CTA count MMA needs (num_splits=1 at T=1 measures 530 us).
After H2-H6 the kernel executes ~100K dynamic instructions (was ~4.9M) and is pure
latency: the remaining floor is the dequant pre-pass, per-tile ring waits/barriers, and
3 launch gaps.

Dead ends / notes:
- calling the fused prefill kernel at T=1 without split-KV: 530 us (SM starvation).
- ragged split chunks (not x16): +20% (whole wasted MMA tiles at every split tail).
- benching in a worktree: the venv's editable `flash_mla` resolves to the MAIN
  checkout — set `PYTHONPATH` to the worktree or script runs import the wrong .so.
- torch caching-allocator pollution (e.g. an fp32 oracle in the same process) slows
  per-op host allocations and inflates wall time ~10 us; the bench empty_cache()s
  after its parity check.
- GPU0 desktop/sibling load adds up to +7 us jitter on GPU1 wall times; use min-of-3
  at >=300 iters.

Env matrix: `FLASH_MLA_DECODE_FUSED=0` -> legacy in-CTA-dequant FMA kernel (bitwise
old path, re-verified 47.0/80.2); `FLASH_MLA_DECODE_MMA=0` -> H1 scratch + FMA
attention; `FLASH_MLA_DECODE_FUSED_COMBINE=1` -> in-kernel last-CTA combine;
`FLASH_MLA_SLOTS_PER_SPLIT` -> split sweeps. Binding change: the fused path allocates
a bf16 [T*width, 512] scratch (`sel_kv_ptr`/`sel_width`) only when num_splits>1, so
large-T prefill fallbacks never allocate it.

## Production split-precision correction and RTX 3090 policy (2026-08-31)

The selection scratch remains bf16, but split attention partials do not. Storing each
split's unnormalized output accumulator in bf16 made the combine result depend on the
number of splits. This was small under ordinary random-error tolerances but became a
deterministic token error in a long-context serving request. The split `oaccum` buffer,
stores, loads, and vectorized combine are now fp32 for both fp8 and int8 decode. The
final output remains bf16.

The precision test was authored against the old implementation first. At the production
long-context shape it compares a one-split result with both coarse and fine split layouts.
The bf16-partial implementation failed both cases (RMSE about 6.1e-4); the fp32-partial
implementation passes both. The complete sparse INT8 decode suite passes 33 tests,
including adversarial cache layout, ragged selections, invalid indices, and split-count
invariance.

Split selection now models the two attention implementations separately:

- MMA: 32 heads per CTA, one resident CTA per SM, target 96 selected rows per split,
  capped by useful occupancy.
- FMA: 16 heads per CTA, up to three queued CTAs per SM.

The 96-row MMA policy was swept on a real 82-SM RTX 3090 at the production packed-INT8
layout. CUDA-event medians for the selected defaults are:

| tokens | selected width | splits | median latency |
| ---: | ---: | ---: | ---: |
| 1 | 640 | 7 | 33.26 us |
| 1 | 1024 | 11 | 34.11 us |
| 1 | 4224 | 41 | 58.13 us |
| 8 | 640 | 5 | 59.16 us |
| 8 | 1024 | 5 | 81.56 us |
| 8 | 4224 | 5 | 263.68 us |

Against the previous eight-oriented policy on the same GPU, the corrected default is
about 33% faster for the long single-token shape and 15% faster for the eight-token
shape. Neither kernel has register spills in the active attention or combine paths.

The end-to-end acceptance test used the OpenAI-compatible chat-completions API with a
512k-token input and a 1k-token verbatim-retrieval output. With decode CUDA graphs
enabled, the fp32-partial native kernel reproduced the needle exactly and sustained
35.36 decoded tokens/s. Running the same binary in forced eager mode roughly halved
decode throughput, so eager diagnostic deployments must not be used as production
performance baselines.
