// Ampere (sm_86) CUDA sparse-MLA decode for DeepSeek-V4-Flash (absorbed form).
//
// P2 step 1 (K-sharing): each tile of BLOCK_N selected slots is gathered + fp8-dequantized
// ONCE into shared memory by the whole CTA, then reused by all BLOCK_H heads (P1 re-did the
// gather per head -> 8x redundant). Online (exp2) softmax with attn_sink, V==K, head_dim 512.
// Validated against the fp32 torch oracle (tests/test_sparse_mla_decode_sm86.py).
// Still fp32-FMA math; P2 step 2 puts QK/PV on mma.sync tensor cores.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math.h>

#include "flash_mla.h"

namespace {

constexpr int FP8_DIM = 448;
constexpr int ROPE_DIM = 64;
constexpr int SCALE_DIM = 8;
constexpr int TOKEN_DATA_SIZE = FP8_DIM + ROPE_DIM * 2;  // 576
constexpr int SCALE_GROUP = 64;
constexpr int HEAD_DIM = 512;
constexpr int VEC = HEAD_DIM / 32;        // 16 dims/lane
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BLOCK_H = 8;                // heads/CTA (one warp each)
constexpr int BLOCK_N = 32;               // selected slots per tile
constexpr int NTHREADS = BLOCK_H * 32;    // 256

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

struct StreamArgs {
    const uint8_t *cache;
    int64_t block_stride;
    int block_size, num_blocks, topk;
    const int *indices, *lens;
};

// K-shared tiled processing of one stream. All NTHREADS gather K into K_s; the h<H warps then
// fold each tile into their per-head online-softmax state.
__device__ __forceinline__ void process_stream(const StreamArgs &st, int t, int tid, int warp,
                                               int lane, int h, bool active, float scale_log2,
                                               const float *q_reg, __nv_bfloat16 (*K_s)[HEAD_DIM],
                                               int *slot_s, float &m, float &l, float *acc) {
    if (st.cache == nullptr) return;
    int len = st.lens[t];
    const int *idx_row = st.indices + (int64_t)t * st.topk;
    int num_slots = st.num_blocks * st.block_size;

    for (int base = 0; base < len; base += BLOCK_N) {
        int n_valid = min(BLOCK_N, len - base);
        if (tid < BLOCK_N) slot_s[tid] = (tid < n_valid) ? idx_row[base + tid] : -1;
        __syncthreads();
        // cooperative gather + dequant: thread handles (n,d) pairs, contiguous d => coalesced.
        for (int v = tid; v < BLOCK_N * HEAD_DIM; v += NTHREADS) {
            int n = v / HEAD_DIM, d = v - n * HEAD_DIM;
            float val = 0.f;
            if (n < n_valid) {
                int slot = slot_s[n];
                if (slot >= 0 && slot < num_slots) {
                    const uint8_t *data, *scale;
                    slot_ptrs(st.cache, st.block_stride, st.block_size, slot, data, scale);
                    val = decode_k_dim(data, scale, d);
                }
            }
            K_s[n][d] = __float2bfloat16(val);
        }
        __syncthreads();
        if (active) {
            for (int n = 0; n < n_valid; ++n) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < VEC; ++i) dot += q_reg[i] * __bfloat162float(K_s[n][lane + i * 32]);
                float s = warp_sum(dot) * scale_log2;
                float m_new = fmaxf(m, s);
                float resc = exp2f(m - m_new);
                float pj = exp2f(s - m_new);
                l = l * resc + pj;
#pragma unroll
                for (int i = 0; i < VEC; ++i)
                    acc[i] = acc[i] * resc + pj * __bfloat162float(K_s[n][lane + i * 32]);
                m = m_new;
            }
        }
        __syncthreads();  // K_s reused next tile
    }
}

__global__ void __launch_bounds__(NTHREADS)
sparse_mla_decode_kernel(__grid_constant__ const Sparse_mla_decode_params p) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int h = blockIdx.y * BLOCK_H + warp;
    const bool active = h < p.num_heads;

    __shared__ __nv_bfloat16 K_s[BLOCK_N][HEAD_DIM];   // 32*512*2 = 32 KB
    __shared__ int slot_s[BLOCK_N];

    float q_reg[VEC];
    if (active) {
        const __nv_bfloat16 *q_ptr = reinterpret_cast<const __nv_bfloat16 *>(p.q_ptr)
            + (int64_t)t * p.q_token_stride + (int64_t)h * p.q_head_stride;
#pragma unroll
        for (int i = 0; i < VEC; ++i) q_reg[i] = __bfloat162float(q_ptr[lane + i * 32]);
    } else {
#pragma unroll
        for (int i = 0; i < VEC; ++i) q_reg[i] = 0.f;
    }

    float m = (active && p.attn_sink_ptr != nullptr) ? p.attn_sink_ptr[h] * LOG2E : -INFINITY;
    float l = (active && p.attn_sink_ptr != nullptr) ? 1.f : 0.f;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.f;

    StreamArgs swa{reinterpret_cast<const uint8_t *>(p.swa_cache_ptr), p.swa_block_stride,
                   p.block_size, p.swa_num_blocks, p.swa_topk, p.swa_indices_ptr, p.swa_lens_ptr};
    process_stream(swa, t, tid, warp, lane, h, active, p.scale_log2, q_reg, K_s, slot_s, m, l, acc);
    StreamArgs ex{reinterpret_cast<const uint8_t *>(p.extra_cache_ptr), p.extra_block_stride,
                  p.extra_block_size, p.extra_num_blocks, p.extra_topk, p.extra_indices_ptr, p.extra_lens_ptr};
    process_stream(ex, t, tid, warp, lane, h, active, p.scale_log2, q_reg, K_s, slot_s, m, l, acc);

    if (active) {
        float inv = 1.f / fmaxf(l, 1e-20f);
        __nv_bfloat16 *out_ptr = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr)
            + (int64_t)t * p.out_token_stride + (int64_t)h * p.out_head_stride;
#pragma unroll
        for (int i = 0; i < VEC; ++i) out_ptr[lane + i * 32] = __float2bfloat16(acc[i] * inv);
    }
}

}  // namespace

void run_sparse_mla_decode(Sparse_mla_decode_params &params, cudaStream_t stream) {
    dim3 grid(params.num_tokens, (params.num_heads + BLOCK_H - 1) / BLOCK_H);
    sparse_mla_decode_kernel<<<grid, NTHREADS, 0, stream>>>(params);
}
