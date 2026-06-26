// Ampere (sm_86) CUDA sparse-MLA decode for DeepSeek-V4-Flash (absorbed form).
//
// P2 step 4 (SPLIT-KV / flash-decoding): the lever to fill the SMs at T=1 decode. Each
// (token, head-block, split) CTA processes a contiguous slice of the concatenated swa+extra
// selection, gathering + fp8-dequantizing K once into smem (shared across heads) and folding
// it into a per-row online softmax. Each CTA writes a PARTIAL (un-normalized acc + running
// max m + denom l). A combine kernel then merges the splits per (token, head) via log-sum-exp
// and applies attn_sink. fp32 math; tensor cores don't help this low-M memory/parallelism-
// bound decode (see docs/sparse_mla_decode_sm86_design.md). Validated vs the fp32 oracle.

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
constexpr int BLOCK_H = 16;               // heads/CTA (one warp each) — fewer head-blocks => less redundant fp8 decode
constexpr int BLOCK_N = 16;               // selected slots per tile
constexpr int NTHREADS = BLOCK_H * 32;    // 512

__device__ __forceinline__ float decode_k_dim(const uint8_t *data, const uint8_t *scale, int d) {
    if (d < FP8_DIM) {
        __half h = __half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)data[d], __NV_E4M3));
        return __half2float(h) * ldexpf(1.0f, (int)scale[d / SCALE_GROUP] - 127);
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

// merge all splits for one (token, head) via log-sum-exp + attn_sink, write final output.
// (run by the last split-CTA to finish for this (t, head_block) — fused, no 2nd launch.)
__device__ __forceinline__ void combine_one_head(const Sparse_mla_decode_params &p, int t, int h, int lane) {
    const int64_t pm0 = ((int64_t)t * p.num_heads + h) * p.num_splits;
    float gm = (p.attn_sink_ptr != nullptr) ? p.attn_sink_ptr[h] * LOG2E : -INFINITY;
    for (int sp = 0; sp < p.num_splits; ++sp) gm = fmaxf(gm, p.mlse_ptr[(pm0 + sp) * 2]);
    float gl = (p.attn_sink_ptr != nullptr) ? exp2f(p.attn_sink_ptr[h] * LOG2E - gm) : 0.f;
    float cacc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) cacc[i] = 0.f;
    for (int sp = 0; sp < p.num_splits; ++sp) {
        float scale = exp2f(p.mlse_ptr[(pm0 + sp) * 2] - gm);
        if (scale == 0.f) continue;
        gl += p.mlse_ptr[(pm0 + sp) * 2 + 1] * scale;
        const __nv_bfloat16 *oacc = reinterpret_cast<const __nv_bfloat16 *>(p.oaccum_ptr) + (pm0 + sp) * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < VEC; ++i) cacc[i] += __bfloat162float(oacc[lane + i * 32]) * scale;
    }
    float inv = 1.f / fmaxf(gl, 1e-20f);
    __nv_bfloat16 *out_ptr = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr)
        + (int64_t)t * p.out_token_stride + (int64_t)h * p.out_head_stride;
#pragma unroll
    for (int i = 0; i < VEC; ++i) out_ptr[lane + i * 32] = __float2bfloat16(cacc[i] * inv);
}

// ---- main kernel: grid (T, head_blocks, num_splits) -> partial (acc, m, l) per split ----
__global__ void __launch_bounds__(NTHREADS)
sparse_mla_decode_split_kernel(__grid_constant__ const Sparse_mla_decode_params p) {
    const int t = blockIdx.x;
    const int sp = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int h = blockIdx.y * BLOCK_H + warp;
    const bool active = h < p.num_heads;

    __shared__ __nv_bfloat16 K_s[BLOCK_N][HEAD_DIM];
    __shared__ int slot_s[BLOCK_N];
    __shared__ uint8_t stream_s[BLOCK_N];

    const int swa_len = p.swa_lens_ptr[t];
    const int extra_len = (p.extra_cache_ptr != nullptr) ? p.extra_lens_ptr[t] : 0;
    const int total = swa_len + extra_len;
    const int chunk = (total + p.num_splits - 1) / p.num_splits;
    const int g_start = sp * chunk;
    const int g_end = min(g_start + chunk, total);

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

    float m = -INFINITY, l = 0.f;     // NO sink here; the combine adds it once
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.f;

    const uint8_t *swa_cache = reinterpret_cast<const uint8_t *>(p.swa_cache_ptr);
    const uint8_t *extra_cache = reinterpret_cast<const uint8_t *>(p.extra_cache_ptr);
    const int swa_slots = p.swa_num_blocks * p.block_size;
    const int extra_slots = p.extra_num_blocks * p.extra_block_size;

    for (int base = g_start; base < g_end; base += BLOCK_N) {
        int n_valid = min(BLOCK_N, g_end - base);
        if (tid < BLOCK_N) {
            if (tid < n_valid) {
                int g = base + tid;
                if (g < swa_len) { slot_s[tid] = p.swa_indices_ptr[(int64_t)t * p.swa_topk + g]; stream_s[tid] = 0; }
                else { slot_s[tid] = p.extra_indices_ptr[(int64_t)t * p.extra_topk + (g - swa_len)]; stream_s[tid] = 1; }
            } else { slot_s[tid] = -1; stream_s[tid] = 0; }
        }
        __syncthreads();
        // vectorized gather: 2 dims/thread. fp8 decoded 2-at-a-time (cvt_fp8x2, one scale/pair,
        // d even => same 64-group); RoPE is already bf16 -> direct copy (no convert).
        for (int vp = tid; vp < BLOCK_N * (HEAD_DIM / 2); vp += NTHREADS) {
            int n = vp / (HEAD_DIM / 2), d = (vp - n * (HEAD_DIM / 2)) * 2;
            __nv_bfloat16 v0 = __float2bfloat16(0.f), v1 = v0;
            if (n < n_valid) {
                int slot = slot_s[n];
                int is_ex = stream_s[n];
                const uint8_t *cache = is_ex ? extra_cache : swa_cache;
                int64_t bstride = is_ex ? p.extra_block_stride : p.swa_block_stride;
                int bsize = is_ex ? p.extra_block_size : p.block_size;
                int nslots = is_ex ? extra_slots : swa_slots;
                if (slot >= 0 && slot < nslots) {
                    const uint8_t *data, *scale;
                    slot_ptrs(cache, bstride, bsize, slot, data, scale);
                    if (d < FP8_DIM) {
                        unsigned short raw2 = *reinterpret_cast<const unsigned short *>(data + d);
                        __half2 h2 = __half2(__nv_cvt_fp8x2_to_halfraw2(raw2, __NV_E4M3));
                        float sc = ldexpf(1.f, (int)scale[d / SCALE_GROUP] - 127);
                        v0 = __float2bfloat16(__low2float(h2) * sc);
                        v1 = __float2bfloat16(__high2float(h2) * sc);
                    } else {
                        const __nv_bfloat16 *rope = reinterpret_cast<const __nv_bfloat16 *>(data + FP8_DIM);
                        v0 = rope[d - FP8_DIM];
                        v1 = rope[d + 1 - FP8_DIM];
                    }
                }
            }
            K_s[n][d] = v0;
            K_s[n][d + 1] = v1;
        }
        __syncthreads();
        if (active) {
            for (int n = 0; n < n_valid; ++n) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < VEC; ++i) dot += q_reg[i] * __bfloat162float(K_s[n][lane + i * 32]);
                float s = warp_sum(dot) * p.scale_log2;
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
        __syncthreads();
    }

    if (active) {
        // partial out: oaccum[t][h][sp][:] = un-normalized acc ; mlse[t][h][sp] = {m, l}
        int64_t po = (((int64_t)t * p.num_heads + h) * p.num_splits + sp) * HEAD_DIM;
        __nv_bfloat16 *oacc = reinterpret_cast<__nv_bfloat16 *>(p.oaccum_ptr) + po;
#pragma unroll
        for (int i = 0; i < VEC; ++i) oacc[lane + i * 32] = __float2bfloat16(acc[i]);
        if (lane == 0) {
            int64_t pm = ((int64_t)t * p.num_heads + h) * p.num_splits + sp;
            p.mlse_ptr[pm * 2] = m;
            p.mlse_ptr[pm * 2 + 1] = l;
        }
    }

    // fused combine: the LAST split-CTA to finish for this (t, head_block) merges all splits
    // (split-K reduction pattern) -> no separate combine launch + no host round-trip.
    __threadfence();        // make this CTA's partials globally visible
    __syncthreads();
    __shared__ bool s_last;
    if (tid == 0) {
        int cidx = t * (int)gridDim.y + (int)blockIdx.y;
        s_last = (atomicAdd(&p.combine_counter_ptr[cidx], 1) == p.num_splits - 1);
    }
    __syncthreads();
    if (s_last) {
        __threadfence();    // acquire every CTA's partials
        if (active) combine_one_head(p, t, h, lane);
    }
}

}  // namespace

void run_sparse_mla_decode(Sparse_mla_decode_params &params, cudaStream_t stream) {
    int head_blocks = (params.num_heads + BLOCK_H - 1) / BLOCK_H;
    cudaMemsetAsync(params.combine_counter_ptr, 0,
                    (size_t)params.num_tokens * head_blocks * sizeof(int), stream);
    dim3 grid(params.num_tokens, head_blocks, params.num_splits);
    sparse_mla_decode_split_kernel<<<grid, NTHREADS, 0, stream>>>(params);
}
