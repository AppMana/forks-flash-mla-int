# FlashMLA

***Adapted from：*** https://github.com/deepseek-ai/FlashMLA/

FlashMLA was initially developed based on Hopper(can refer to:https://github.com/deepseek-ai/FlashMLA/), and I adapted it to Ampere GPUs. Due to the different architectures, the performance of Ampere is currently poor due to register overflow. Welcome to add good optimization ideas.

Currently released:
- BF16
- Paged kvcache with block size of 32
- Warp-specialized SM80 kernel (`warp_spec=True`): splits 8 warps into 4 consumer (QK^T GEMM + softmax) and 4 producer (global memory loads), with all 8 warps cooperating on PV GEMM. Uses double-buffered K/V loads and raw PTX barriers for SM80 compatibility.
- sm86 DeepSeek-V4 sparse MLA decode over fp8_ds_mla paged KV cache.
- sm86 sparse MLA prefill entry point. This is a correctness-tested alias for
  the same direct fp8_ds_mla paged-cache kernel used by decode; the AppMana vLLM
  serving fork currently uses Triton for sm86 prefill because the measured
  gathered-bf16 Triton attention body is faster on RTX A5000.

## AppMana DeepSeek-V4 sm86 serving status

Both sparse-MLA stages now run this repository's fused tensor-core kernels
(tag `dsv4-sm86-kernels-2026-07-09`). The legacy **decode** kernel remains callable
via `FLASH_MLA_DECODE_FUSED=0` as an A/B reference (it is not broken). The legacy
**prefill** staged kernel was **removed** (`csrc/flash_sparse_mla_prefill_staged_sm80.cu`,
the `FLASH_MLA_PREFILL_FUSED=0` kill-switch, and the `use_staged_prefill` /
`staged_chunk_tokens` wrapper kwargs are all gone).

**Why the staged prefill was removed (do not resurrect it).** It was reported to
abort at runtime on torch 2.11 via a stable-ABI `torch_call_dispatcher "aten::new_empty"`
failure, and the fused kernel had already replaced it as the default and is
substantially faster (1.8-2.1x in the shipped README table; ~4x measured head-to-head
at the true 16k footprint: fused 5.3 ms vs staged 22.2 ms at T=1024/width 1024,
9.9 ms vs 44.8 ms at width 2048). No production caller reached it (the vLLM fork
calls `flash_sparse_mla_prefill` without `use_staged_prefill`, so it always got fused),
so it was dead code. The fused kernel handles BOTH `fp8_ds_mla` and `int8_ds_mla`
caches — int8 does not depend on any staged fallback. (Note: the exact torch-2.11
abort could not be reproduced on 2.11.0+cu130 in this checkout; the staged path ran
but ~4x slower. The removal stands on the dead-code + performance grounds regardless.)

| Stage | Kernel path |
| --- | --- |
| Decode sparse MLA | `flash_mla.flash_sparse_mla_decode` (fused, heads-as-M mma) |
| Prefill sparse MLA | `flash_mla.flash_sparse_mla_prefill` (fused; fp8: whole-cache dequant pre-pass, int8: in-kernel dequant) |

Both take the KV row stride as a runtime argument and accept `fp8_ds_mla` and
`int8_ds_mla` caches (the int8 layout is 512 int8 payload + fp32 scale at byte
offset 512 + 12B pad = a 528-byte 16-byte-aligned token stride; see
`vllm/models/deepseek_v4/common/ops/cache_utils.py` in the vLLM fork).

RTX A5000 (sm_86), **true 16k-slot caches**, T=1 decode / T=1024 prefill chunk,
64 heads, CUDA events, min-of-3 at >=300 iters:

| Case | fused | prior native | production Triton |
| --- | ---: | ---: | ---: |
| Decode top-k 512 | **22.5 us** | 47.2 us (2.10x) | 213.7 us |
| Decode top-k 1024 | **28.2 us** | 80.6 us (2.85x) | 409.2 us |
| Prefill width 512 (us/token) | **3.0-3.3** | 13.7 | 5.8 |
| Prefill width 1024 (us/token) | **5.5** | 24.1 | 11.4 |

Why the old "tensor cores are a dead end for low-M" conclusion was wrong: at
T=1 the *heads* supply the M dimension, not the tokens. ncu on the old decode
kernel showed 2.28M ALU + 2.57M FMA + 1.42M LSU and **zero HMMA** — it was
issue-bound on software fp8->bf16 conversion. Dequantizing the selected rows
once into a bf16 scratch and running `mma.m16n8k16` with heads as M cuts
dynamic instructions ~50x. Prefill applies the same idea at a different
granularity: it dequantizes the *whole cache* once (3% of the op) because every
row is reused ~32-64x within a chunk, whereas decode dequantizes only the
selected rows because T=1 has no reuse to amortize. Running the prefill kernel
at T=1 costs 530 us (SM starvation) — the two are not interchangeable.

The whole-cache pre-pass is now **fp8-only**. Its bf16 buffer is 2 KiB per pool
slot — 2.25 GiB per call at the production 2.3M-slot pool — which OOM'd 24GB
ranks. The int8 prefill instead gathers raw int8 rows (half the random-gather
bytes) and dequantizes int8\*scale in smem between the cp.async wait and the QK
mma; unlike the fp8 cvt chain this is one cvt+mul per element. Bit-exact vs the
retired pre-pass path. At the production geometry (2.3M-slot pool, T=1024,
top-k 2048) this is 23% faster (16.7 -> 12.8 ms) with a 64 MiB peak op
footprint (was 2.31 GiB); at a 16k-slot pool with T=256/width 1024 it is 20%
slower (1.36 -> 1.64 ms) because the tiny pre-pass amortized perfectly there —
accepted, long-context prefill is the binding regime.

### Benchmarking caveats (read before trusting a number)

- **Cache footprint decides the answer.** The pre-2026-07 conclusion that
  Triton beat native prefill 3.25x was an artifact of a 192-slot microbench
  whose cache was L2-resident. Always bench prefill against a >=16k-slot cache
  (`benchmarks/bench_sparse_mla_decode_16k.py` and the 16k matrix bench do
  this). Decode is immune — its T=1 working set is `topk x 576B`, L2-resident
  regardless of total cache size — but bench it at 16k anyway so the numbers
  compose.
- **A worktree can silently bench the wrong build.** The venv's editable
  `flash_mla` resolves to the main checkout; `PYTHONPATH` must point at the
  worktree, or a script-style run imports the other build. This invalidated one
  measurement before it was caught.
- **In-process fp32 oracles pollute the allocator** and inflate wall time by
  ~10 us; call `torch.cuda.empty_cache()` after parity checks, before timing.
- **A busy sibling GPU adds up to +7 us jitter** even on a different device.
  Use min-of-3 at >=300 iters, and pin `CUDA_VISIBLE_DEVICES`.
- The decode call is 3 kernels (selection-dequant, mma attention, warp-wide
  combine); ~2.5 us of the 22.5 us is launch gap that CUDA-graph capture in
  serving recovers.

Rejected experiments (measured, not assumed): whole-cache dequant at decode
(would exceed the entire kernel budget at T=1); occupancy-first prefill restructure
(BM=16 + 4-way k-split reached 33% occupancy but doubled gather DRAM and ran 35%
slower — gather redundancy, not occupancy, is the binding constraint at 6MB L2);
ragged non-x16 split chunks (+20%); finer-than-16 decode splits (combine/oaccum
overhead dominates).

Benchmark command:

```bash
source /home/administrator/Documents/forks-vllm-ampere/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_sparse_mla_sm86_shapes.py \
  --device 0 --warmup 10 --iters 30 --torch-iters 1 \
  --decode-batches 1 2 4 6 --prefill-tokens 256 512 1024 2048 \
  --max-torch-tokens 0 --max-torch-topk 0
```

## Quick start

### Install

```bash
python setup.py install
```

### Benchmark

```bash
# amphere gpus
python tests/test_flash_mla_sm80.py

# hopper gpus
python tests/test_flash_mla_sm90.py
```

It is able up to 464 GB/s in memory-bound configuration and 59 TFLOPS in computation-bound configuration on A100 SXM, using CUDA 12.8.
For [reference](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), the peak bandwidth and fp16 FLOPS of A100 SXM are 2039 GB/s and 312 TFLOPS respectively. More efforts are needed to optimize the performance.

### Usage

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    # Cooperative kernel (default)
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    # Warp-specialized kernel (SM80 only)
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True, warp_spec=True,
    )
    ...
```

## Requirements

- Ampere GPUs
- CUDA 12.3 and above
- PyTorch 2.0 and above

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

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
