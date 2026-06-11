# sm_86 launch failure: shared-memory accounting and fix design

Status 2026-06-10. Both SM80 MLA kernels (plain and warp-specialized) compile
for sm_86 but **cannot launch on RTX 3090 / RTX A5000**:
`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` fails with
`invalid argument` (`flash_fwd_mla_kernel_sm80.h:689`) because the requested
dynamic shared memory exceeds the device limit. The kernels have only ever
launched on A100-class parts.

## Accounting (traits `<576, kBlockM=32, kBlockN=32, kNWarps=8, bf16, dV=512>`, kP=2)

| member | bytes |
|---|---|
| `smem_q` (32 x 576 x 2B) | 36,864 |
| `smem_k` (32 x 576 x 2B x kP=2) | 73,728 |
| union alternative `smem_o` (32 x 512 x 4B) | 65,536 |
| union resolves to max(q+k, o) | 110,592 |
| `smem_p` (32 x 32 x 2B) | 2,048 |
| `smem_reduce` (32 x 5 x 4B) | 640 |
| **total** | **113,280** |

Device `sharedMemPerBlockOptin`: A100 = 166,912; **sm_86 = 101,376**.
113,280 > 101,376, hence the launch failure. The warp-specialized variant uses
the same tile sizes and fails identically.

## Constraint map for shrinking below 101,376

- `kBlockN` is pinned: `FLASH_ASSERT(params.page_block_size == kBlockN)`;
  the paged KV cache uses 32-token blocks. Changing kBlockN means changing
  the cache layout in the serving stack.
- `kBlockM=32` is pinned by the MMA tiling (`Tile<16*kNWarpsM=32, ...>`,
  `kNWarpsM=2` hardcoded).
- Reducing `kNWarps` 8 -> 4 would allow `kBlockN=16` tiling-wise, but see the
  page-size pin above.
- The remaining lever is **kP (K pipeline depth) 2 -> 1**: smem_k halves to
  36,864; union resolves to max(73,728, 65,536) = 73,728; total ~76.4 KB,
  comfortably under 101,376.

## kP=1 is not just a constant change

The current loop issues the next block's K load at the **top** of the
iteration into `smem_pipe_write`, then `cp_async_wait<1>` (one group may
remain in flight) before computing on `smem_pipe_read`
(`flash_fwd_mla_kernel_sm80.h:415-435`, same shape in the non-masking loop at
~505-512). With kP=1, read == write == 0, so the top-of-loop issue would
clobber the buffer the current iteration is about to read.

Required restructure for a `kP==1` variant (guard with `if constexpr`):

1. Move the "advance gK" cp.async issue from the top of each loop to after
   the PV GEMM (`gemm_8x(acc_o, ...)`), preceded by `__syncthreads()` so all
   consumer reads of `sK`/`sVt` complete first (WAR hazard).
2. Change `cp_async_wait<1>` to `cp_async_wait<Kernel_traits::kP::value - 1>`
   at both wait sites (lines ~432, ~510).
3. Keep the prologue load + fence unchanged.
4. Dispatch at runtime in `mha_fwd_splitkv_mla<>::run`: query
   `cudaDevAttrMaxSharedMemoryPerBlockOptin`; choose the kP=2 traits when the
   kP=2 SharedStorage fits, else the kP=1 traits. Instantiate both.

Expected cost: single-buffered K removes load/compute overlap, so decode
attention throughput drops relative to a hypothetical double-buffered fit,
but it is the difference between running and not running on the chain's
hardware. A follow-up could restore overlap by splitting K staging to the
existing `kNumInnerStagesK` granularity (192-column slices) instead of whole
blocks.

## Build notes for this machine (CUDA 13)

- `cuda/std/utility` lives under `/usr/local/cuda/include/cccl`; build with
  `CPATH=/usr/local/cuda/include/cccl`.
- Delete any stale repo-root `flash_mla_cuda*.so` before testing; an editable
  install plus a stale root artifact shadows imports and embeds old paths.
- Build: `CPATH=/usr/local/cuda/include/cccl TORCH_CUDA_ARCH_LIST=8.6
  MAX_JOBS=16 uv pip install -e . --no-build-isolation --python <venv>/bin/python`.
- Test (script-style, not pytest): `python tests/test_flash_mla_sm80.py`.

## kP=1 implementation status (2026-06-10, branch appmana/sm86-smem-analysis)

Implemented and committed:

- `kPipe_` trait parameter (1 or 2) with the kP=2 default byte-identical.
- Both pipeline loops restructured under `if constexpr (kP == 1)`: top-of-loop
  prefetch removed, `cp_async_wait<0>` before compute, and the next-K issue
  moved to after the PV GEMM behind a `__syncthreads()` WAR barrier.
- Runtime dispatch in `mha_fwd_splitkv_mla::run` via
  `cudaDevAttrMaxSharedMemoryPerBlockOptin`.

Compile blocker RESOLVED the same day: `tile_to_shape` over a pipe extent
of 1 emits a compound `(_1,_1):(_0,_0)` trailing mode that breaks the
composition's shape_div bookkeeping (verified with standalone cute probes;
the identical rank-2 composition succeeds against the kP=2 layout). Fix:
for kPipe_=1, build the pipe-free `(kBlockN, kHeadDim)` layout first,
compose the V-transpose in rank 2, then append a clean `_1:_0` pipe mode to
both `SmemLayoutK` and `SmemLayoutVtransposed` (constexpr builder functions
in the traits). The kP=1 path is enabled by default
(`#ifndef FLASH_MLA_DISABLE_KP1`).

**Validation 2026-06-10 on RTX A5000 (sm_86)**: `tests/test_flash_mla_sm80.py`
passes all 32 configurations (b=128, s in 4096/8192, h_q in 16/32/64/128,
s_q in 1/2, varlen both; cos_diff < 8e-5 vs the FP32 torch reference on out
and lse), ~39-41 TFLOPS and up to ~171 GB/s with the single-buffered
pipeline. This is the first time the kernel runs on sm_86 at all. Follow-up
perf work: restore load/compute overlap at kNumInnerStagesK (192-column)
granularity within the single-buffer budget.

## Fork integration map (recon 2026-06-11)

The vLLM fork already contains a complete `flashmla_sparse` attention backend
(`vllm/v1/attention/backends/mla/flashmla_sparse.py`) that consumes the V4
`fp8_ds_mla` main-MLA cache (584 B/token: 448 B fp8 NoPE + 128 B bf16 RoPE +
8 B scale) and calls `flash_mla_with_kvcache` (dense/indices path) and
`flash_mla_sparse_fwd` (sparse path) from the flash_mla package. It is gated
at `supports_compute_capability: major in [9, 10]`.

Integration deltas for sm_86, in order:

1. This repo: a sparse forward (`flash_mla_sparse_fwd`) sm_86 variant, and
   `fp8_ds_mla` ingest for the kP=1 kernel (decode fp8 NoPE arithmetically,
   or natively once the INT8 main-KV half of the int8-cache campaign lands;
   RoPE part is already bf16).
2. vLLM fork: extend the capability gate to major 8 behind a flag, plus any
   sm-count/num_sm_parts metadata differences.

The plain BF16 dense kernel validated on the A5000 is the foundation; the
sparse + fp8_ds_mla deltas are the remaining kernel work.
