// Ampere (sm_86) CUDA sparse-MLA decode for DeepSeek-V4-Flash (absorbed form).
//
// Phase 1: a correct, tensor-core-free implementation. Each warp owns one head;
// it streams the indexer-selected top-k slots (swa then extra), gathers + dequantizes
// the fp8_ds_mla K row in-kernel, does the q.K dot via a warp reduction, runs the
// online (exp2) softmax with attn_sink, and accumulates V==K. Validated against the
// fp32 torch oracle in tests/test_sparse_mla_decode_sm86.py. P2 swaps QK/PV onto
// mma.sync tensor cores + cp.async double-buffer + K-sharing across heads.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math.h>

#include "flash_mla.h"

namespace {

constexpr int FP8_DIM = 448;          // NoPE dims, fp8e4m3
constexpr int ROPE_DIM = 64;          // RoPE dims, bf16
constexpr int SCALE_DIM = 8;          // per-token scale bytes (7 UE8M0 + 1 pad)
constexpr int TOKEN_DATA_SIZE = FP8_DIM + ROPE_DIM * 2;  // 576
constexpr int SCALE_GROUP = 64;
constexpr int HEAD_DIM = 512;         // 448 NoPE + 64 RoPE
constexpr int VEC = HEAD_DIM / 32;    // 16 dims/lane
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BLOCK_H = 8;            // heads per CTA (one warp each)
constexpr int NTHREADS = BLOCK_H * 32;

__device__ __forceinline__ float decode_k_dim(const uint8_t *data, const uint8_t *scale, int d) {
    if (d < FP8_DIM) {
        __half h = __half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)data[d], __NV_E4M3));
        float s = exp2f((float)scale[d / SCALE_GROUP] - 127.0f);
        return __half2float(h) * s;
    }
    const __nv_bfloat16 *rope = reinterpret_cast<const __nv_bfloat16 *>(data + FP8_DIM);
    return __bfloat162float(rope[d - FP8_DIM]);
}

__device__ __forceinline__ void slot_ptrs(const uint8_t *cache, int64_t block_stride, int block_size,
                                          int slot, const uint8_t *&data, const uint8_t *&scale) {
    int b = slot / block_size, i = slot - b * block_size;
    const uint8_t *blk = cache + (int64_t)b * block_stride;
    data = blk + (int64_t)i * TOKEN_DATA_SIZE;
    scale = blk + (int64_t)block_size * TOKEN_DATA_SIZE + (int64_t)i * SCALE_DIM;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One stream (swa or extra): fold all valid selected slots into the online softmax state.
__device__ __forceinline__ void run_stream(const Sparse_mla_decode_params &p, int t, int lane,
                                           const float *q_reg, const uint8_t *cache,
                                           int64_t block_stride, int block_size, int num_blocks,
                                           const int *indices, const int *lens, int topk,
                                           float &m, float &l, float *acc) {
    if (cache == nullptr) return;
    int len = lens[t];
    const int *idx_row = indices + (int64_t)t * topk;
    for (int j = 0; j < len; ++j) {
        int slot = idx_row[j];
        if (slot < 0 || slot >= num_blocks * block_size) continue;
        const uint8_t *data, *scale;
        slot_ptrs(cache, block_stride, block_size, slot, data, scale);
        float k_reg[VEC];
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            k_reg[i] = decode_k_dim(data, scale, lane + i * 32);
            dot += q_reg[i] * k_reg[i];
        }
        float s = warp_sum(dot) * p.scale_log2;
        float m_new = fmaxf(m, s);
        float resc = exp2f(m - m_new);
        float pj = exp2f(s - m_new);
        l = l * resc + pj;
#pragma unroll
        for (int i = 0; i < VEC; ++i) acc[i] = acc[i] * resc + pj * k_reg[i];
        m = m_new;
    }
}

__global__ void __launch_bounds__(NTHREADS)
sparse_mla_decode_kernel(__grid_constant__ const Sparse_mla_decode_params p) {
    const int t = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = blockIdx.y * BLOCK_H + warp;
    if (h >= p.num_heads) return;

    const __nv_bfloat16 *q_ptr = reinterpret_cast<const __nv_bfloat16 *>(p.q_ptr)
        + (int64_t)t * p.q_token_stride + (int64_t)h * p.q_head_stride;
    float q_reg[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) q_reg[i] = __bfloat162float(q_ptr[lane + i * 32]);

    float m = (p.attn_sink_ptr != nullptr) ? p.attn_sink_ptr[h] * LOG2E : -INFINITY;
    float l = (p.attn_sink_ptr != nullptr) ? 1.f : 0.f;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.f;

    run_stream(p, t, lane, q_reg, reinterpret_cast<const uint8_t *>(p.swa_cache_ptr),
               p.swa_block_stride, p.block_size, p.swa_num_blocks,
               p.swa_indices_ptr, p.swa_lens_ptr, p.swa_topk, m, l, acc);
    run_stream(p, t, lane, q_reg, reinterpret_cast<const uint8_t *>(p.extra_cache_ptr),
               p.extra_block_stride, p.extra_block_size, p.extra_num_blocks,
               p.extra_indices_ptr, p.extra_lens_ptr, p.extra_topk, m, l, acc);

    float inv = 1.f / fmaxf(l, 1e-20f);
    __nv_bfloat16 *out_ptr = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr)
        + (int64_t)t * p.out_token_stride + (int64_t)h * p.out_head_stride;
#pragma unroll
    for (int i = 0; i < VEC; ++i) out_ptr[lane + i * 32] = __float2bfloat16(acc[i] * inv);
}

}  // namespace

void run_sparse_mla_decode(Sparse_mla_decode_params &params, cudaStream_t stream) {
    dim3 grid(params.num_tokens, (params.num_heads + BLOCK_H - 1) / BLOCK_H);
    sparse_mla_decode_kernel<<<grid, NTHREADS, 0, stream>>>(params);
}
