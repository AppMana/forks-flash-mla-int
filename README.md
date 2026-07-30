# FlashMLA (Ampere)

An Ampere port of [FlashMLA](https://github.com/deepseek-ai/FlashMLA/), whose kernels are
Hopper (WGMMA) and Blackwell (tcgen05) only.

Two kernel families ship here:

- **Sparse MLA** for DeepSeek-V4-Flash in the absorbed form (`head_dim = 512`, `V == K`,
  attention restricted to the indexer-selected top-k slots). Native CUDA tensor-core
  decode and prefill kernels for `sm_80`/`sm_86`, over both `fp8_ds_mla` and `int8_ds_mla`
  paged caches. In every one of them the cache dtype names the *storage*: the math is bf16
  tensor-core throughout — see
  [`int8` names the cache, not the math](#int8-names-the-cache-not-the-math).
- **Dense MLA** — the upstream paged-KV `head_dim 576 / 512` decode kernel, with an
  additional warp-specialized SM80 variant alongside the SM90 originals.

The sparse kernels are the active work. They are written from the algorithm rather than
ported from the Hopper WGMMA kernel, whose core instructions have no Ampere equivalent.

## Requirements

- CUDA 12.3 or newer (12.8 or newer for `sm_100` / `sm_120` codegen)
- PyTorch 2.9 or newer — the binding is torch-stable-ABI, so wheels are `cp39-abi3` and
  work across torch versions
- Compute capability 8.0 or newer. The Ampere kernels use `cp.async`, `ldmatrix` and
  `mma.m16n8k16`, so `sm_80` is a hard floor.

The sparse operators check the device at call time and accept compute capability **8.x
(Ampere) and 12.x (consumer Blackwell)**; the dense operator accepts 8.x and 9.0.

Consumer Blackwell — `sm_120`/`sm_121`, including the GB10 in a DGX Spark — runs the
same kernels: it has all three Ampere instructions, and its
`cudaDevAttrMaxSharedMemoryPerBlockOptin` is 101376 B, identical to `sm_86`, so the
shared-memory budgets are unchanged. This is not merely convenient — it is the only
sparse-MLA path available there, since upstream FlashMLA covers Hopper via WGMMA
(`sm_90`) and datacenter Blackwell via tcgen05 (`sm_100`), and consumer Blackwell
implements neither.

Instruction availability is not numerical parity, so treat a new architecture as
unvalidated until the `tests/` suites — which check every kernel against an fp32 oracle
— pass on that hardware.

## Install

### From the published index (recommended)

CI publishes wheels to a PEP 503 simple index on this repo's `gh-pages` branch, so
pip resolves the right one for your platform — x86_64 or aarch64, `manylinux_2_28`
or jammy — without pinning a URL:

```bash
pip install flash-mla --extra-index-url https://appmana.github.io/forks-flash-mla-int/
```

Every wheel is `cp39-abi3` and torch-stable, so one wheel works across torch >= 2.9
built against CUDA 13.x. The SASS matrix baked in is `FLASH_MLA_CUDA_ARCHS` below.

### From source

```bash
pip install .
```

### Building for a specific architecture

`FLASH_MLA_CUDA_ARCHS` is a comma-separated list of `code=sm_XX` names and selects the SASS
emitted for the Ampere family (`csrc/*_sm80.cu`). It defaults to `80`.

```bash
# native build for RTX A5000 / RTX 3090
FLASH_MLA_CUDA_ARCHS=86 pip install .

# fat binary
FLASH_MLA_CUDA_ARCHS=80,86,89,90,100,120 python setup.py bdist_wheel
```

The Hopper sources (`csrc/*_sm90.cu`) are always compiled for `sm_90a`, and the
architecture-agnostic translation units are compiled for `sm_90a` plus every requested
architecture.

Other build-time switches:

| Variable | Default | Effect |
| --- | --- | --- |
| `FLASH_MLA_CUDA_ARCHS` | `80` | Target architectures for the Ampere kernels |
| `FLASH_MLA_DISABLE_FP16` | `FALSE` | Drop the fp16 SM90 kernel |
| `FLASH_MLA_DISABLE_STABLE_ABI` | `FALSE` | Build a version-specific pybind extension instead of `cp39-abi3` |
| `FLASH_MLA_DEBUG` | `FALSE` | `-O0 -g -G -lineinfo` |
| `FLASH_MLA_LOCAL_VERSION` | `FALSE` | Append `+<git sha>` to the wheel version |
| `NVCC_THREADS` | `32` | `nvcc --threads` |

A CMake path (`CMakeLists.txt`, `make`) also exists for compile-database and
standalone-library builds; there the target architecture is `CMAKE_CUDA_ARCHITECTURES`,
which defaults to `80`.

## Python API

```python
from flash_mla import (
    get_mla_metadata, flash_mla_with_kvcache,          # dense paged MLA
    sparse_mla_decode_fp8, sparse_mla_prefill,         # sparse MLA, fp8_ds_mla cache
    sparse_mla_decode_int8, sparse_mla_prefill_int8,   # sparse MLA, int8_ds_mla cache
    quantize_int8_ds_mla_rows,                         # int8_ds_mla row quantizer
    sparse_mla_decode_int8_triton,                     # Triton reference decode
)
```

The four sparse entry points share a shape contract. `q` is `(num_tokens, num_heads, 512)`
bfloat16 and the output has the same shape; `swa_indices` is `(num_tokens, topk)` int32
slot ids (`-1` and out-of-range entries are masked) and `swa_lens` is `(num_tokens,)`
int32. An optional second index stream (`extra_cache` / `extra_indices` / `extra_lens`,
plus `extra_scale` on the int8 entry points) covers the compressed cache, and `attn_sink`
is an optional `(num_heads,)` float32 tensor merged once into the softmax denominator.
Causality is already encoded in the per-query selection, so decode and prefill differ only
in how many query tokens `q` carries.

```python
out = sparse_mla_decode_fp8(q, swa_cache, swa_indices, swa_lens,
                            scale=None, attn_sink=None,
                            extra_cache=None, extra_indices=None, extra_lens=None)

out = sparse_mla_prefill_int8(q, swa_cache, swa_scale, swa_indices, swa_lens,
                              scale=None, attn_sink=None,
                              extra_cache=None, extra_scale=None,
                              extra_indices=None, extra_lens=None)
```

Dense MLA is unchanged from upstream apart from the `warp_spec` flag:

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)
o, lse = flash_mla_with_kvcache(
    q, kvcache, block_table, cache_seqlens, dv,
    tile_scheduler_metadata, num_splits, causal=True,
    warp_spec=True,   # SM80 warp-specialized kernel; ignored on SM90
)
```

### Registered operators

`import flash_mla_cuda` registers `torch.ops.flash_mla.*` (`csrc/flash_api.cpp`):

| Operator | Python wrapper |
| --- | --- |
| `get_mla_metadata` | `flash_mla.get_mla_metadata` |
| `fwd_kvcache_mla` | `flash_mla.flash_mla_with_kvcache` |
| `fwd_sparse_decode_mla` | `flash_mla.sparse_mla_decode_fp8` |
| `fwd_sparse_prefill_mla` | `flash_mla.sparse_mla_prefill` |
| `fwd_sparse_int8_decode_mla` | `flash_mla.sparse_mla_decode_int8` |
| `fwd_sparse_int8_prefill_mla` | `flash_mla.sparse_mla_prefill_int8` |
| `debug_imma_m16n8k32_s8s8` | — (test-only IMMA fragment probe) |

`flash_mla.flash_mla_with_kvcache_int8` is declared in the Python API, but its operator
(`fwd_kvcache_mla_int8`, a fully-fused int8 *dense* decode) is not built; calling it raises
`NotImplementedError`. The design is in `docs/int8_fused_sm86_design.md`.

## Cache layouts

**`fp8_ds_mla`** — paged `uint8`. A block is `[block_size x 576 data][block_size x 8
scales]`. Per token: 448 fp8e4m3 NoPE bytes, then 64 bf16 RoPE values (128 bytes). The
scales are 7 UE8M0 exponents, one per 64 NoPE dims, plus one pad byte. The reconstructed
row is `[448 dequantized ; 64 bf16] = 512` bf16 values.

**`int8_ds_mla`** — 512 signed int8 payload bytes per token plus one fp32 rowwise scale;
the row is `int8 * scale`. Cache and scales are passed as separate tensors (`[nb, bs, 512]`
int8 and `[nb, bs]` float32) and are addressed exclusively through their runtime strides,
so a strided view over an interleaved 528-byte token stride works unchanged. The prefill
entry point additionally requires contiguous rows and 16-byte-aligned block and token
strides, because it gathers with 16-byte `cp.async` chunks. `quantize_int8_ds_mla_rows`
produces this format from bf16 rows.

### `int8` names the cache, not the math

This is the point most often misread, so it is worth stating outright: in
`sparse_mla_decode_int8` and `sparse_mla_prefill_int8`, `int8` describes the **KV cache
format**. The attention arithmetic is **bf16 tensor-core** in both, exactly as in the fp8
entry points.

Every sparse kernel here issues `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for
QK and PV. `q` is bfloat16 on all four entry points — the binding rejects anything else —
and the int8 cache rows are gathered raw and converted to bf16 (`int8 * scale`, one
convert-and-multiply per element) on their way into the mma ring. **No integer MMA is
involved.** The only `s32.s8.s8` instruction in this repository is
`debug_imma_m16n8k32_s8s8`, a test-only fragment probe that no production path calls.

This is by design, not a missing optimization. What the int8 cache buys is **memory**:
528 bytes per token against `fp8_ds_mla`'s 584, halved random-gather traffic on the
prefill path, and — because the int8 prefill dequantizes in-kernel rather than through a
whole-cache pre-pass — no allocation sized by the KV pool at all. What it does not buy is
integer tensor-core throughput, and it is not intended to: the accuracy analysis in
`docs/int8_fused_sm86_design.md` is what justifies storing the cache this way, and the
bf16 math is what keeps the result element-for-element identical to the fp8 pre-pass path.

Integer MMA does appear in the DeepSeek-V4 stack, but in a different kernel and a
different repository: the **lightning indexer** logits, where both the query and the K
cache are int8 (vLLM's `indexer_query_int8` and `indexer_cache_int8` roles). That is a
separate code path from sparse MLA and is not part of this fork.

## Kernel reference

### Sparse MLA decode — `csrc/flash_sparse_mla_decode_sm80.cu`

Host entry `run_sparse_mla_decode(Sparse_mla_decode_params&, cudaStream_t)`. The grid is
split-KV (flash-decoding): each `(token, head block, split)` CTA folds a contiguous slice
of the concatenated `swa + extra` selection into an online softmax and writes an
un-normalized partial plus `{m, l}`, which a combine step merges by log-sum-exp.

| Symbol | Role | Selected when |
| --- | --- | --- |
| `sparse_mla_selection_dequant_kernel` | Pre-pass. Dequantizes the *selected* rows once into a dense bf16 scratch `[T x sel_width, 512]` and zeroes the combine counters. Branches on `params.int8_cache`: `int8 * fp32 scale`, or fp8e4m3 + UE8M0 + bf16 RoPE. This is the only place int8 rows are read on the decode path. | Whenever the scratch is allocated; unconditionally for `int8_ds_mla` |
| `mma_dec::sparse_mla_decode_mma_kernel<bool kFusedCombine>` | Default attention kernel. Heads supply the M dimension (`BLOCK_M = 32`, `BLOCK_N = 16`, 8 warps); QK and PV run on `mma.sync.m16n8k16.f32.bf16.bf16.f32` with `ldmatrix` fragment loads over a 2-deep `cp.async` ring fed from the bf16 scratch. `kFusedCombine=true` has the last split-CTA per head block combine in-kernel. | Default (`FLASH_MLA_DECODE_MMA` not `0`) |
| `mma_dec::sparse_mla_combine_kernel` | Standalone parallel combine: one warp per `(token, head)`, one split per lane, branchless `uint4` accumulation. | Default when `num_splits > 1` |
| `sparse_mla_decode_fused_split_kernel` | FMA attention over the same bf16 selection scratch (16 warps, `BLOCK_H = 16`). A/B reference for the mma kernel. | `FLASH_MLA_DECODE_MMA=0` |
| `sparse_mla_decode_split_kernel` | No scratch: gathers and fp8-dequantizes inside each attention CTA. Reads `fp8_ds_mla` bytes directly and cannot accept an int8 cache. | `FLASH_MLA_DECODE_FUSED=0`, fp8 only |
| `mma_pf::sparse_mla_prefill_mma_kernel` | Batched tensor-core variant for `num_splits == 1`. Off by default: its 64-slot tiles cost about 85 KB of shared memory, so one CTA and four warps per SM, which exposes the gather latency this shape is bound by. | `FLASH_MLA_PREFILL_MMA=1`, fp8 only |

`sparse_mla_decode_fused_enabled()` and `sparse_mla_decode_mma_enabled()` expose the two
switches to the binding, which uses them to size the scratch and pick the split policy.

### Sparse MLA prefill — `csrc/flash_sparse_mla_prefill_fused_sm80.cu`

Host entry `run_sparse_mla_prefill(Sparse_mla_prefill_params&, cudaStream_t)`.

| Symbol | Role | Selected when |
| --- | --- | --- |
| `sparse_prefill_fused_mma_kernel<bool INT8_GATHER>` | The attention kernel: one CTA per `(token, 32-head block)`, 8 warps, `BLOCK_M = 32` / `BLOCK_N = 16`, `mma.sync.m16n8k16` for QK (2-way k-split, so every warp contributes) and PV, `ldmatrix` fragment loads, 2-deep `cp.async` ring, online softmax per head row, direct output write. | Always |
| — `INT8_GATHER = false` | Consumes bf16 rows produced by the whole-cache pre-pass. | `int8_cache == false` |
| — `INT8_GATHER = true` | Gathers raw int8 rows into a separate int8 staging ring (`SmemInt8::k8_s`, half the random-gather bytes of the bf16 ring) and dequantizes `int8 * scale` into the mma ring between the `cp.async` wait and the QK mma. No pre-pass, and nothing sized by the KV pool is allocated. Element-for-element identical to the pre-pass. | `int8_cache == true` |
| `dequant_cache_kernel` | fp8-only pre-pass: dequantizes the *whole* paged cache once into a dense bf16 `[total_slots, 512]` buffer. | `int8_cache == false` |

`static_assert(sizeof(SmemInt8) <= 100 * 1024)` keeps the int8 variant inside the `sm_86`
opt-in shared-memory budget of 101,376 bytes.

The two stages dequantize at different granularities on purpose. Within a prefill chunk
every cached row is reused tens of times, so decoding it once amortizes to noise — but the
`fp8_ds_mla` pre-pass buffer is 1 KiB per pool slot, which does not scale to large pools.
The int8 path therefore dequantizes in-kernel instead: one convert-and-multiply per
element, next to nothing beside the tile's 32x16x512 of QK/PV work, and no pool-sized
allocation. At `T = 1` decode there is no reuse to amortize at all, so only the selected
rows are dequantized. Running the prefill kernel at `T = 1` costs 530 us of SM starvation;
the two are not interchangeable.

### Dense MLA

| Symbol | File |
| --- | --- |
| `mha_fwd_splitkv_mla<T, Headdim, is_sm90>` | `csrc/flash_fwd_mla_kernel_sm80.h`, `csrc/flash_fwd_mla_kernel_sm90.h` |
| `mha_fwd_splitkv_mla_ws<T, Headdim>` | `csrc/flash_fwd_mla_kernel_sm80_ws.h` |
| `run_mha_fwd_splitkv_mla` | `csrc/flash_api_dispatch.cu` (cutlass-free dispatch seam) |
| `get_mla_metadata_func` | `csrc/flash_fwd_mla_metadata.cu` |
| `run_debug_imma_m16n8k32_s8s8` | `csrc/debug_imma_sm80.cu` |

The warp-specialized SM80 kernel (`warp_spec=True`) splits 8 warps into 4 consumer
(QK^T + softmax) and 4 producer (global load) warps, with all 8 cooperating on PV. It uses
double-buffered K/V loads and raw PTX barriers for SM80 compatibility. Both dense SM80
kernels are tiled for A100-class shared memory; the accounting is in
`docs/sm86_smem_analysis.md`.

### Runtime switches

| Variable | Default | Effect |
| --- | --- | --- |
| `FLASH_MLA_DECODE_MMA` | on | `0` selects `sparse_mla_decode_fused_split_kernel` (FMA) instead of the mma kernel |
| `FLASH_MLA_DECODE_FUSED` | on | `0` selects `sparse_mla_decode_split_kernel` (fp8 caches only) |
| `FLASH_MLA_DECODE_FUSED_COMBINE` | off | `1` combines in the last split-CTA instead of launching `sparse_mla_combine_kernel` |
| `FLASH_MLA_PREFILL_MMA` | off | `1` routes `num_splits == 1` fp8 decode calls to `mma_pf::sparse_mla_prefill_mma_kernel` |
| `FLASH_MLA_SLOTS_PER_SPLIT` | auto | Overrides the split-KV chunk size |

## Downstream serving evidence

The AppMana vLLM fork selects the INT8-cache operators by their importable
symbols:

```text
decode=flash_mla.sparse_mla_decode_int8
prefill=flash_mla.sparse_mla_prefill_int8
cache_type=int8_ds_mla
```

That exact kernel selection produced correct DeepSeek-V4 chat answers on two
DGX Sparks at TP=2 and completed an 8,004-token prompt / 1,128-token output
needle run at 5.23 s TTFT, approximately 1,532 input tok/s, and 38.5 decode
tok/s. The 1,000-token needle passed with a 0.94 word-level match ratio.

This is integration evidence, not a release benchmark for this repository:
the server ran from a native host environment whose exact installed
FlashMLA/vLLM revisions were not frozen, and one Triton JIT miss occurred
during inference. It also says nothing about the separate
SparkInfer/NVFP4 lane, which is currently incorrect. The evidence boundaries,
kernel matrix, public comparison numbers, and reproduction checklist are in
the
[Hilton performance reference](https://github.com/hannesholste/dragonintel/blob/main/docs/dsv4-spark-performance-references.md).

## Benchmarks

RTX A5000 (`sm_86`), 16k-slot caches, 64 heads, `T = 1` decode and a 1024-token prefill
chunk, CUDA-event timing, min-of-3 at 300 or more iterations. The Triton baseline is the
reference sparse-MLA Triton implementation for `sm_86`.

| Case | this fork | Triton baseline |
| --- | ---: | ---: |
| Decode top-k 512 | **22.5 us** | 213.7 us |
| Decode top-k 1024 | **28.2 us** | 409.2 us |
| Prefill width 512 (us/token) | **3.0-3.3** | 5.8 |
| Prefill width 1024 (us/token) | **5.5** | 11.4 |

int8 prefill at a 2.3M-slot pool, `T = 1024`, top-k 2048: 12.8 ms, with a 64 MiB peak
allocation for the op.

Every harness also checks parity against an fp32 oracle, so a number is never reported for
a kernel that is producing the wrong answer.

```bash
# true-footprint sparse decode, fp8 cache (self-contained)
python benchmarks/bench_sparse_mla_decode_16k.py --widths 512 1024 --iters 100

# int8 decode: native CUDA vs Triton vs fp8, with cos_diff vs fp32 (self-contained)
python benchmarks/bench_int8_sparse_mla.py --tokens 1 --topk 128 512

# decode and prefill shape sweep; adds Triton columns when vLLM is importable
python benchmarks/bench_sparse_mla_sm86_shapes.py \
  --warmup 10 --iters 30 \
  --decode-batches 1 2 4 6 --prefill-tokens 256 512 1024 2048

# full fp8/int8 x native/Triton matrix at a fixed context (requires vLLM)
python benchmarks/bench_sparse_mla_16k_matrix.py --context 16384 --only prefill-int8-native

# dense MLA, cooperative vs warp-specialized SM80
python tests/bench_compare_sm80.py
```

## Tests

```bash
python -m pytest tests/
```

- `test_sparse_mla_decode_sm86.py`, `test_sparse_mla_prefill_adversarial.py` — sparse fp8
  decode and prefill against an fp32 oracle, including adversarial index patterns.
- `test_sparse_mla_decode_int8_adversarial.py`,
  `test_sparse_mla_prefill_int8_adversarial.py` — the same for `int8_ds_mla`, exercising
  both the interleaved 528-byte strided views and the separate-tensor layout.
- `test_sparse_mla_prefill_int8_memory.py` — asserts the int8 prefill op's peak allocation
  is bounded by the selection, not by the KV pool.
- `test_int8_sparse_mla.py` — Triton int8 decode reference and `quantize_int8_ds_mla_rows`.
- `test_imma_sm80_fragment.py` — `debug_imma_m16n8k32_s8s8` against a torch int32 matmul.
- `test_flash_mla_sm80.py`, `test_flash_mla_sm90.py`, `test_flash_mla_dsv4_sm80.py` —
  dense paged MLA.

## Documentation

- `docs/sparse_mla_decode_sm86_design.md` — sparse decode algorithm and cache byte layout
- `docs/int8_fused_sm86_design.md` — int8 accuracy analysis and the fused int8 design
- `docs/sm86_smem_analysis.md` — shared-memory accounting for the dense SM80 kernels

## Acknowledgement

FlashMLA is inspired by the [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/)
and [cutlass](https://github.com/nvidia/cutlass) projects.

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernel},
      author={Jiashi Li},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
