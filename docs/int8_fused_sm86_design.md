# Fused int8 FlashMLA decode for sm_86 (task #61)

## Goal
A fully-fused int8 decode kernel: int8 QK^T on the s8 IMMA tensor cores, int8 K/V
cache (which halves their SMEM footprint), PV kept in bf16. No separate quant/dequant
passes — Q is quantized in-kernel, K is consumed as int8 directly, V is dequantized
in-kernel just before the bf16 PV MMA.

## Why it pays
- **Accuracy (validated, `tests/int8_oracle_accuracy.py`):** rowwise-symmetric int8 Q/K +
  int8-KV with bf16 PV gives cos_diff ~7e-5 vs fp32 at DSV4 decode dims (h=64, d=576,
  dv=512, sk=512) — under the FlashMLA suite's own `cos_diff < 8e-5` accept bound.
- **SMEM (the sm_86 lever):** bf16 K(72KB)+V(64KB) at kP=2 overflows the 101,376-byte
  sm_86 opt-in budget, forcing the kP=1 single-buffer. int8 K(36KB)+V(32KB) frees ~68KB,
  so the kP=2 double-buffer fits and the K-prefetch overlaps compute again.
- **Baseline to beat (`tests/bench_decode_vs_triton.py`):** the live triton sparse decode
  is 217µs/389µs/1550µs at sk=512/1024/4096; the bf16 FlashMLA CUDA kernel is
  16.8/19.2/31.7µs (12.9x/20.2x/48.9x). int8 widens the memory-bound tail further.

## Kernel structure (DRY: one body, `if constexpr` seams — bf16 path byte-identical)
The existing sm80 kernel (`flash_fwd_mla_kernel_sm80.h`) is homogeneous bf16: one
`TiledMma` (`SM80_16x8x16_F32BF16BF16F32_TN`) drives both GEMMs. Crucially P already
round-trips through smem (`sP`, the "route P through smem" block), so gemm0's output and
gemm1's input are already decoupled — that is the bridge between a different-typed int8
QK MMA and the bf16 PV MMA.

Traits gain `kIsInt8QK` (bool) + an int8 QK MMA atom. Seams, all `if constexpr`:

1. **Traits:** when `kIsInt8QK`, `Element` (smem K/V dtype) = `int8_t`; `MMA_Atom_QK` =
   `SM80_16x8x32_S32S8S8S32_TN` (k=32, s32 accum). PV MMA stays bf16. SmemLayoutK/V become
   int8 → smem halves → kP=2 traits selected by the existing smem-fits dispatch.
2. **acc_s:** partitioned from the int8 QK tiled-mma → `int32`. After gemm0, convert to
   float and multiply by `q_scale[row] * k_scale[col]` (outer product) before softmax.
   bf16 path keeps the f32 acc_s unchanged.
3. **Q quant (in-kernel prologue):** after bf16 Q is in smem, compute per-row (per
   query-head) amax over kHeadDim, `q_scale = amax/127`, write int8 Q to the (now int8) sQ
   smem buffer. q_scale[kBlockM] kept in a small smem/register array, applied at step 2.
4. **K load:** identical cp.async path, `Element=int8_t` → loads int8 cache directly. New
   params `k_scale_ptr` (per kv-token) read alongside.
5. **V dequant:** V aliases the int8 K smem; before the bf16 PV MMA, dequant int8→bf16
   using `v_scale[col]`. Implemented in the V smem→fragment copy (scale-on-load), so PV
   runs in bf16 unchanged. P (softmax probs) stays bf16 — never quantized.
6. **softmax / PV:** unchanged (f32 accum, bf16 P, f32 acc_o).

## Scale factoring (why rowwise works with IMMA)
`score[h,q,k] = (Σ_d q_i8[q,h,d]·k_i8[k,d]) · q_scale[q,h] · k_scale[k] · softmax_scale`.
Per-row Q scale and per-kv-token K scale both factor out of the d-contraction, so the s32
IMMA result is dequantized by a single outer-product multiply. Full-576 rowwise int8
(NoPE+RoPE together) is accurate enough — no separate bf16 RoPE sub-GEMM needed.

## Params / binding / dispatch
- `Flash_fwd_mla_params`: add `k_scale_ptr`, `v_scale_ptr` (+ row/batch strides). Q scale
  is computed in-kernel (no param).
- New `flash_fwd_mla_int8_sm80.cu` instantiates `mha_fwd_splitkv_mla` with the int8-QK
  traits for headdim 576/512. New binding `flash_mla_cuda.fwd_kvcache_mla_int8`.
- Python: `flash_mla_with_kvcache_int8` (already in `flash_mla_interface.py`).

## Test / bench gates
- RED→GREEN: `tests/test_flash_mla_int8_sm86.py` (parity vs the exact int8 oracle, <8e-5).
- Bench: extend `bench_decode_vs_triton.py` with the int8 column once green.

## Cache-format note (deployment, not kernel)
The live cache is fp8_ds_mla (uint8: fp8 NoPE + bf16 RoPE + scales). int8-KV needs an int8
cache writer in the indexer/cache_utils path. For kernel bring-up + microbench we feed int8
KV directly (synthetic); the cache-writer integration is a separate downstream step.
