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

The serving path in `AppMana/forks-vllm-ampere` is hybrid:

| Stage | Kernel path |
| --- | --- |
| Decode sparse MLA | `flash_mla.flash_sparse_mla_decode` from this repository |
| Prefill sparse MLA | vLLM gathered bf16 `sparse_attention_triton` |

The direct FlashMLA prefill integration exposed a real tensor-shape bug: vLLM's
SWA metadata is shaped `[T, 1, window]`, while this extension's C++ binding reads
the sparse-index width from `size(1)`. The Python wrapper now normalizes
`[T, 1, W]` sparse-index tensors to `[T, W]` before calling the op. A RED test
using real CUDA fp8_ds_mla tensors reproduced the mismatch at `cos_diff=1.92e-03`
before the wrapper fix.

Local RTX A5000 timings with fp8_ds_mla cache tensors, 64 heads, random
selected slots, no LMCache reuse:

| Case | FlashMLA | Triton | Result |
| --- | ---: | ---: | --- |
| Decode C4A B=1 top-k 512 | 0.074 ms | 0.255 ms | FlashMLA 3.44x faster |
| Decode C4A B=6 top-k 512 | 0.226 ms | 0.258 ms | FlashMLA 1.14x faster |
| Decode C128A 16k B=6 top-k 128 | 0.098 ms | 0.108 ms | FlashMLA 1.10x faster |
| Decode C128A 200k B=6 top-k 1664 | 0.601 ms | 0.732 ms | FlashMLA 1.22x faster |
| Prefill C4A T=256 top-k 512 | 8.170 ms | 2.506 ms | Triton 3.26x faster |
| Prefill C4A T=2048 top-k 512 | 65.028 ms | 19.996 ms | Triton 3.25x faster |
| Prefill C128A T=2048 top-k 512 | 65.260 ms | 20.106 ms | Triton 3.25x faster |

Benchmark command:

```bash
source /home/administrator/Documents/forks-vllm-ampere/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_sparse_mla_sm86_shapes.py \
  --device 0 --warmup 10 --iters 30 --torch-iters 1 \
  --decode-batches 1 2 4 6 --prefill-tokens 256 512 1024 2048 \
  --max-torch-tokens 0 --max-torch-topk 0
```

Small reference run including torch oracle:

```bash
TRITON_CACHE_DIR=/tmp/triton-sm86-sparse-cache CUDA_VISIBLE_DEVICES=1 \
  python benchmarks/bench_sparse_mla_sm86_shapes.py \
  --device 0 --warmup 3 --iters 10 --torch-iters 1 \
  --decode-batches 1 --prefill-tokens 64 \
  --max-torch-tokens 64 --max-torch-topk 512
```

That run measured C4A prefill T=64 at 2.212 ms for the direct FlashMLA
fp8_ds_mla path, 0.577 ms for Triton over already-gathered bf16 KV, and
13.272 ms for the torch reference.

Future sm86 prefill work should specialize the kernel for the `T >= 64` regime
and compare against the gathered-bf16 Triton path before changing the serving
default.

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
