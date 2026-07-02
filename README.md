# FlashMLA

***Adapted from：*** https://github.com/deepseek-ai/FlashMLA/

FlashMLA was initially developed based on Hopper(can refer to:https://github.com/deepseek-ai/FlashMLA/), and I adapted it to Ampere GPUs. Due to the different architectures, the performance of Ampere is currently poor due to register overflow. Welcome to add good optimization ideas.

Currently released:
- BF16
- Paged kvcache with block size of 32
- Warp-specialized SM80 kernel (`warp_spec=True`): splits 8 warps into 4 consumer (QK^T GEMM + softmax) and 4 producer (global memory loads), with all 8 warps cooperating on PV GEMM. Uses double-buffered K/V loads and raw PTX barriers for SM80 compatibility.
- sm86 DeepSeek-V4 sparse MLA decode over fp8_ds_mla paged KV cache.
- sm86 sparse MLA prefill entry point. This is tested for correctness, but the
  AppMana vLLM serving fork currently uses Triton for sm86 prefill because the
  measured gathered-bf16 Triton path is faster on RTX A5000.

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

Local RTX A5000 timings with fp8_ds_mla cache tensors, 64 heads, top-k 512:

| Case | FlashMLA | Triton | Result |
| --- | ---: | ---: | --- |
| Decode T=1 | 0.045 ms | 0.233 ms | FlashMLA 5.23x faster |
| Decode T=4 | 0.142 ms | 0.225 ms | FlashMLA 1.59x faster |
| Prefill T=64 | 1.949 ms | 0.492 ms | Triton 3.96x faster |
| Prefill T=256 | 7.722 ms | 2.113 ms | Triton 3.65x faster |
| Prefill T=1024 | 30.388 ms | 8.285 ms | Triton 3.67x faster |

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
