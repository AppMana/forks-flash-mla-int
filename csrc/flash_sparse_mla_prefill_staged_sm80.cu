#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "flash_mla.h"

namespace {

constexpr int FP8_DIM = 448;
constexpr int ROPE_DIM = 64;
constexpr int SCALE_DIM = 8;
constexpr int TOKEN_DATA_SIZE = FP8_DIM + ROPE_DIM * 2;
constexpr int SCALE_GROUP = 64;
constexpr int HEAD_DIM = 512;
constexpr int VEC = HEAD_DIM / 32;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float fp8_decode_dim(const uint8_t *data,
                                                 const uint8_t *scale,
                                                 int d) {
    if (d < FP8_DIM) {
        __half h = __half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)data[d],
                                                  __NV_E4M3));
        return __half2float(h) * ldexpf(1.0f, (int)scale[d / SCALE_GROUP] - 127);
    }
    const __nv_bfloat16 *rope =
        reinterpret_cast<const __nv_bfloat16 *>(data + FP8_DIM);
    return __bfloat162float(rope[d - FP8_DIM]);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__global__ void gather_fp8_prefill_kernel(Sparse_mla_prefill_staged_params p) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.num_tokens * p.width * HEAD_DIM;
    const uint8_t *swa_cache = reinterpret_cast<const uint8_t *>(p.swa_cache_ptr);
    const uint8_t *extra_cache = reinterpret_cast<const uint8_t *>(p.extra_cache_ptr);
    __nv_bfloat16 *kv = reinterpret_cast<__nv_bfloat16 *>(p.kv_ptr);

    for (int idx = linear; idx < total; idx += blockDim.x * gridDim.x) {
        int d = idx % HEAD_DIM;
        int n = (idx / HEAD_DIM) % p.width;
        int t = idx / (HEAD_DIM * p.width);
        int swa_len = p.swa_lens_ptr[t];
        int extra_len = extra_cache != nullptr ? p.extra_lens_ptr[t] : 0;
        float val = 0.f;
        if (n < swa_len) {
            int slot = p.swa_indices_ptr[(int64_t)t * p.swa_topk + n];
            int num_slots = p.swa_num_blocks * p.swa_block_size;
            if (slot >= 0 && slot < num_slots) {
                int b = slot / p.swa_block_size;
                int pos = slot - b * p.swa_block_size;
                const uint8_t *blk = swa_cache + (int64_t)b * p.swa_block_stride;
                const uint8_t *data = blk + (int64_t)pos * TOKEN_DATA_SIZE;
                const uint8_t *scale =
                    blk + (int64_t)p.swa_block_size * TOKEN_DATA_SIZE +
                    (int64_t)pos * SCALE_DIM;
                val = fp8_decode_dim(data, scale, d);
            }
        } else if (n < swa_len + extra_len) {
            int en = n - swa_len;
            int slot = p.extra_indices_ptr[(int64_t)t * p.extra_topk + en];
            int num_slots = p.extra_num_blocks * p.extra_block_size;
            if (slot >= 0 && slot < num_slots) {
                int b = slot / p.extra_block_size;
                int pos = slot - b * p.extra_block_size;
                const uint8_t *blk = extra_cache + (int64_t)b * p.extra_block_stride;
                const uint8_t *data = blk + (int64_t)pos * TOKEN_DATA_SIZE;
                const uint8_t *scale =
                    blk + (int64_t)p.extra_block_size * TOKEN_DATA_SIZE +
                    (int64_t)pos * SCALE_DIM;
                val = fp8_decode_dim(data, scale, d);
            }
        }
        kv[((int64_t)t * p.width + n) * HEAD_DIM + d] = __float2bfloat16(val);
    }
}

__global__ void gather_int8_prefill_kernel(Sparse_mla_prefill_staged_params p) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.num_tokens * p.width * HEAD_DIM;
    const int8_t *swa_cache = reinterpret_cast<const int8_t *>(p.swa_cache_ptr);
    const int8_t *extra_cache = reinterpret_cast<const int8_t *>(p.extra_cache_ptr);
    __nv_bfloat16 *kv = reinterpret_cast<__nv_bfloat16 *>(p.kv_ptr);

    for (int idx = linear; idx < total; idx += blockDim.x * gridDim.x) {
        int d = idx % HEAD_DIM;
        int n = (idx / HEAD_DIM) % p.width;
        int t = idx / (HEAD_DIM * p.width);
        int swa_len = p.swa_lens_ptr[t];
        int extra_len = extra_cache != nullptr ? p.extra_lens_ptr[t] : 0;
        float val = 0.f;
        if (n < swa_len) {
            int slot = p.swa_indices_ptr[(int64_t)t * p.swa_topk + n];
            int num_slots = p.swa_num_blocks * p.swa_block_size;
            if (slot >= 0 && slot < num_slots) {
                int b = slot / p.swa_block_size;
                int pos = slot - b * p.swa_block_size;
                const int8_t *row =
                    swa_cache + (int64_t)b * p.swa_block_stride +
                    (int64_t)pos * p.swa_pos_stride;
                const float scale =
                    p.swa_scale_ptr[(int64_t)b * p.swa_scale_block_stride +
                                    (int64_t)pos * p.swa_scale_pos_stride];
                val = (float)row[d] * scale;
            }
        } else if (n < swa_len + extra_len) {
            int en = n - swa_len;
            int slot = p.extra_indices_ptr[(int64_t)t * p.extra_topk + en];
            int num_slots = p.extra_num_blocks * p.extra_block_size;
            if (slot >= 0 && slot < num_slots) {
                int b = slot / p.extra_block_size;
                int pos = slot - b * p.extra_block_size;
                const int8_t *row =
                    extra_cache + (int64_t)b * p.extra_block_stride +
                    (int64_t)pos * p.extra_pos_stride;
                const float scale =
                    p.extra_scale_ptr[(int64_t)b * p.extra_scale_block_stride +
                                      (int64_t)pos * p.extra_scale_pos_stride];
                val = (float)row[d] * scale;
            }
        }
        kv[((int64_t)t * p.width + n) * HEAD_DIM + d] = __float2bfloat16(val);
    }
}

template <int BLOCK_H, int BLOCK_N>
__global__ void sparse_prefill_gathered_kernel(Sparse_mla_prefill_staged_params p) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * BLOCK_H + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const bool active = h < p.num_heads;
    __shared__ __nv_bfloat16 k_s[BLOCK_N][HEAD_DIM];

    float q[VEC];
    if (active) {
        const __nv_bfloat16 *q_ptr = reinterpret_cast<const __nv_bfloat16 *>(p.q_ptr) +
            (int64_t)t * p.q_token_stride + (int64_t)h * p.q_head_stride;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            q[i] = __bfloat162float(q_ptr[lane + i * 32]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            q[i] = 0.f;
        }
    }

    float m = active && p.attn_sink_ptr != nullptr
        ? p.attn_sink_ptr[h] * LOG2E : -INFINITY;
    float l = active && p.attn_sink_ptr != nullptr ? 1.f : 0.f;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        acc[i] = 0.f;
    }

    int len = p.swa_lens_ptr[t] + (p.extra_cache_ptr != nullptr ? p.extra_lens_ptr[t] : 0);
    const __nv_bfloat16 *kv = reinterpret_cast<const __nv_bfloat16 *>(p.kv_ptr) +
        (int64_t)t * p.width * HEAD_DIM;
    for (int base = 0; base < len; base += BLOCK_N) {
        int n_valid = min(BLOCK_N, len - base);
        for (int e = threadIdx.x; e < BLOCK_N * HEAD_DIM; e += BLOCK_H * 32) {
            int n = e / HEAD_DIM;
            int d = e - n * HEAD_DIM;
            k_s[n][d] = (n < n_valid) ? kv[(base + n) * HEAD_DIM + d]
                                      : __float2bfloat16(0.f);
        }
        __syncthreads();
        if (active) {
            for (int n = 0; n < n_valid; ++n) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    dot += q[i] * __bfloat162float(k_s[n][lane + i * 32]);
                }
                float s = warp_sum(dot) * p.scale_log2;
                float nm = fmaxf(m, s);
                float resc = exp2f(m - nm);
                float pj = exp2f(s - nm);
                l = l * resc + pj;
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    acc[i] = acc[i] * resc + pj * __bfloat162float(k_s[n][lane + i * 32]);
                }
                m = nm;
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv = 1.f / fmaxf(l, 1e-20f);
        __nv_bfloat16 *out = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr) +
            (int64_t)t * p.out_token_stride + (int64_t)h * p.out_head_stride;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            out[lane + i * 32] = __float2bfloat16(acc[i] * inv);
        }
    }
}

namespace mma_gathered {

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 64;
constexpr int NWARPS = 4;
constexpr int NTHREADS = NWARPS * 32;
constexpr int D_PER_WARP = HEAD_DIM / NWARPS;
constexpr int KS_LD = HEAD_DIM + 8;
constexpr int KT = HEAD_DIM / 16;
constexpr int NT = BLOCK_N / 16;

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 a, __nv_bfloat16 b) {
    return (uint32_t)(*reinterpret_cast<uint16_t *>(&a)) |
           ((uint32_t)(*reinterpret_cast<uint16_t *>(&b)) << 16);
}

__device__ __forceinline__ uint32_t pack_contig(const __nv_bfloat16 *p) {
    return *reinterpret_cast<const uint32_t *>(p);
}

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Smem {
    __nv_bfloat16 q_s[BLOCK_M][HEAD_DIM];
    __nv_bfloat16 k_s[BLOCK_N][KS_LD];
    float row_scratch[NWARPS][BLOCK_M];
    __nv_bfloat16 p_s[BLOCK_M][BLOCK_N];
};

__global__ void __launch_bounds__(NTHREADS)
sparse_prefill_gathered_mma_kernel(Sparse_mla_prefill_staged_params p) {
    extern __shared__ char smem_raw[];
    Smem &s = *reinterpret_cast<Smem *>(smem_raw);
    const int t = blockIdx.x;
    const int hb = blockIdx.y * BLOCK_M;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int qr0 = lane >> 2;
    const int qc = (lane & 3) * 2;

    for (int e = tid; e < BLOCK_M * HEAD_DIM; e += NTHREADS) {
        int r = e / HEAD_DIM;
        int c = e - r * HEAD_DIM;
        int h = hb + r;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (h < p.num_heads) {
            const __nv_bfloat16 *qp = reinterpret_cast<const __nv_bfloat16 *>(p.q_ptr) +
                (int64_t)t * p.q_token_stride + (int64_t)h * p.q_head_stride;
            v = qp[c];
        }
        s.q_s[r][c] = v;
    }

    float acc[D_PER_WARP / 8][4];
#pragma unroll
    for (int d = 0; d < D_PER_WARP / 8; ++d) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[d][j] = 0.f;
        }
    }

    int h0 = hb + qr0;
    int h1 = hb + qr0 + 8;
    bool sink = p.attn_sink_ptr != nullptr;
    float m0 = (sink && h0 < p.num_heads) ? p.attn_sink_ptr[h0] * LOG2E : -INFINITY;
    float m1 = (sink && h1 < p.num_heads) ? p.attn_sink_ptr[h1] * LOG2E : -INFINITY;
    float l0 = (sink && h0 < p.num_heads) ? 1.f : 0.f;
    float l1 = (sink && h1 < p.num_heads) ? 1.f : 0.f;
    int len = p.swa_lens_ptr[t] + (p.extra_cache_ptr != nullptr ? p.extra_lens_ptr[t] : 0);
    const __nv_bfloat16 *kv = reinterpret_cast<const __nv_bfloat16 *>(p.kv_ptr) +
        (int64_t)t * p.width * HEAD_DIM;
    __syncthreads();

    for (int base = 0; base < len; base += BLOCK_N) {
        int n_valid = min(BLOCK_N, len - base);
        for (int v = tid; v < BLOCK_N * HEAD_DIM; v += NTHREADS) {
            int n = v / HEAD_DIM;
            int d = v - n * HEAD_DIM;
            __nv_bfloat16 val = __float2bfloat16(0.f);
            if (n < n_valid) {
                val = kv[(base + n) * HEAD_DIM + d];
            }
            s.k_s[n][d] = val;
        }
        __syncthreads();

        float sc[2][4];
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc[nn][j] = 0.f;
            }
        }
        const int ncol0 = warp * 16;
        const int group = lane >> 2;
        const int tig = lane & 3;
        for (int k = 0; k < KT; ++k) {
            int k0 = k * 16;
            uint32_t a[4];
            a[0] = pack_contig(&s.q_s[group][k0 + tig * 2]);
            a[1] = pack_contig(&s.q_s[group + 8][k0 + tig * 2]);
            a[2] = pack_contig(&s.q_s[group][k0 + tig * 2 + 8]);
            a[3] = pack_contig(&s.q_s[group + 8][k0 + tig * 2 + 8]);
            uint32_t b0[2], b1[2];
            b0[0] = pack_contig(&s.k_s[ncol0 + group][k0 + tig * 2]);
            b0[1] = pack_contig(&s.k_s[ncol0 + group][k0 + tig * 2 + 8]);
            b1[0] = pack_contig(&s.k_s[ncol0 + 8 + group][k0 + tig * 2]);
            b1[1] = pack_contig(&s.k_s[ncol0 + 8 + group][k0 + tig * 2 + 8]);
            mma16816(sc[0], a, b0);
            mma16816(sc[1], a, b1);
        }
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc[nn][j] *= p.scale_log2;
            }
        }

        float tmax0 = fmaxf(fmaxf(sc[0][0], sc[0][1]), fmaxf(sc[1][0], sc[1][1]));
        float tmax1 = fmaxf(fmaxf(sc[0][2], sc[0][3]), fmaxf(sc[1][2], sc[1][3]));
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, o));
            tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, o));
        }
        if ((lane & 3) == 0) {
            s.row_scratch[warp][qr0] = tmax0;
            s.row_scratch[warp][qr0 + 8] = tmax1;
        }
        __syncthreads();
        float tile_m0 = -INFINITY, tile_m1 = -INFINITY;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            tile_m0 = fmaxf(tile_m0, s.row_scratch[w][qr0]);
            tile_m1 = fmaxf(tile_m1, s.row_scratch[w][qr0 + 8]);
        }
        float nm0 = fmaxf(m0, tile_m0);
        float nm1 = fmaxf(m1, tile_m1);
        float resc0 = exp2f(m0 - nm0);
        float resc1 = exp2f(m1 - nm1);
        m0 = nm0;
        m1 = nm1;

        float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            float p00 = exp2f(sc[nn][0] - m0);
            float p01 = exp2f(sc[nn][1] - m0);
            float p10 = exp2f(sc[nn][2] - m1);
            float p11 = exp2f(sc[nn][3] - m1);
            int col = ncol0 + nn * 8 + qc;
            bool v0 = col < n_valid;
            bool v1 = (col + 1) < n_valid;
            p00 = v0 ? p00 : 0.f;
            p01 = v1 ? p01 : 0.f;
            p10 = v0 ? p10 : 0.f;
            p11 = v1 ? p11 : 0.f;
            ps0 += p00 + p01;
            ps1 += p10 + p11;
            s.p_s[qr0][col] = __float2bfloat16(p00);
            s.p_s[qr0][col + 1] = __float2bfloat16(p01);
            s.p_s[qr0 + 8][col] = __float2bfloat16(p10);
            s.p_s[qr0 + 8][col + 1] = __float2bfloat16(p11);
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, o);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, o);
        }
        __syncthreads();
        if ((lane & 3) == 0) {
            s.row_scratch[warp][qr0] = ps0;
            s.row_scratch[warp][qr0 + 8] = ps1;
        }
        __syncthreads();
        float fps0 = 0.f, fps1 = 0.f;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            fps0 += s.row_scratch[w][qr0];
            fps1 += s.row_scratch[w][qr0 + 8];
        }
        l0 = l0 * resc0 + fps0;
        l1 = l1 * resc1 + fps1;

#pragma unroll
        for (int d = 0; d < D_PER_WARP / 8; ++d) {
            acc[d][0] *= resc0;
            acc[d][1] *= resc0;
            acc[d][2] *= resc1;
            acc[d][3] *= resc1;
        }
        __syncthreads();

        const int dcol0 = warp * D_PER_WARP;
        for (int nt = 0; nt < NT; ++nt) {
            int sl = nt * 16 + tig * 2;
            uint32_t pa[4];
            pa[0] = pack_contig(&s.p_s[group][nt * 16 + tig * 2]);
            pa[1] = pack_contig(&s.p_s[group + 8][nt * 16 + tig * 2]);
            pa[2] = pack_contig(&s.p_s[group][nt * 16 + tig * 2 + 8]);
            pa[3] = pack_contig(&s.p_s[group + 8][nt * 16 + tig * 2 + 8]);
#pragma unroll
            for (int d8 = 0; d8 < D_PER_WARP / 8; ++d8) {
                int dcol = dcol0 + d8 * 8 + group;
                uint32_t kb[2];
                kb[0] = pack2(s.k_s[sl][dcol], s.k_s[sl + 1][dcol]);
                kb[1] = pack2(s.k_s[sl + 8][dcol], s.k_s[sl + 9][dcol]);
                mma16816(acc[d8], pa, kb);
            }
        }
        __syncthreads();
    }

    float inv0 = 1.f / fmaxf(l0, 1e-20f);
    float inv1 = 1.f / fmaxf(l1, 1e-20f);
    const int dcol0 = warp * D_PER_WARP;
    __nv_bfloat16 *o0 = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr) +
        (int64_t)t * p.out_token_stride + (int64_t)h0 * p.out_head_stride;
    __nv_bfloat16 *o1 = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr) +
        (int64_t)t * p.out_token_stride + (int64_t)h1 * p.out_head_stride;
#pragma unroll
    for (int d8 = 0; d8 < D_PER_WARP / 8; ++d8) {
        int col = dcol0 + d8 * 8 + qc;
        if (h0 < p.num_heads) {
            o0[col] = __float2bfloat16(acc[d8][0] * inv0);
            o0[col + 1] = __float2bfloat16(acc[d8][1] * inv0);
        }
        if (h1 < p.num_heads) {
            o1[col] = __float2bfloat16(acc[d8][2] * inv1);
            o1[col + 1] = __float2bfloat16(acc[d8][3] * inv1);
        }
    }
}

}  // namespace mma_gathered

namespace mma_stream16 {

constexpr int BLOCK_M = 16;
constexpr int HEAD_STEP = 16;
constexpr int BLOCK_N = 16;
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;
constexpr int D_PER_WARP = HEAD_DIM / NWARPS;
constexpr int KS_LD = HEAD_DIM + 8;
constexpr int KT = HEAD_DIM / 16;

using mma_gathered::mma16816;
using mma_gathered::pack2;
using mma_gathered::pack_contig;

struct Smem {
    __nv_bfloat16 q_s[BLOCK_M][HEAD_DIM];
    __nv_bfloat16 k_s[BLOCK_N][KS_LD];
    __nv_bfloat16 p_s[BLOCK_M][BLOCK_N];
    float resc_s[BLOCK_M];
    float inv_s[BLOCK_M];
};

__global__ void __launch_bounds__(NTHREADS)
sparse_prefill_stream16_mma_kernel(Sparse_mla_prefill_staged_params p) {
    extern __shared__ char smem_raw[];
    Smem &s = *reinterpret_cast<Smem *>(smem_raw);
    const int t = blockIdx.x;
    const int hb = blockIdx.y * HEAD_STEP;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int qr0 = group;
    const int qr1 = group + 8;
    const int qc = tig * 2;

    for (int e = tid; e < BLOCK_M * HEAD_DIM; e += NTHREADS) {
        int r = e / HEAD_DIM;
        int c = e - r * HEAD_DIM;
        int h = hb + r;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (h < p.num_heads) {
            const __nv_bfloat16 *qp =
                reinterpret_cast<const __nv_bfloat16 *>(p.q_ptr) +
                (int64_t)t * p.q_token_stride + (int64_t)h * p.q_head_stride;
            v = qp[c];
        }
        s.q_s[r][c] = v;
    }

    float acc[D_PER_WARP / 8][4];
#pragma unroll
    for (int d = 0; d < D_PER_WARP / 8; ++d) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[d][j] = 0.f;
        }
    }

    const int h0 = hb + qr0;
    const int h1 = hb + qr1;
    const bool sink = p.attn_sink_ptr != nullptr;
    float m0 = (sink && h0 < p.num_heads) ? p.attn_sink_ptr[h0] * LOG2E : -INFINITY;
    float m1 = (sink && h1 < p.num_heads) ? p.attn_sink_ptr[h1] * LOG2E : -INFINITY;
    float l0 = (sink && h0 < p.num_heads) ? 1.f : 0.f;
    float l1 = (sink && h1 < p.num_heads) ? 1.f : 0.f;
    const int len =
        p.swa_lens_ptr[t] + (p.extra_cache_ptr != nullptr ? p.extra_lens_ptr[t] : 0);
    const __nv_bfloat16 *kv =
        reinterpret_cast<const __nv_bfloat16 *>(p.kv_ptr) +
        (int64_t)t * p.width * HEAD_DIM;
    __syncthreads();

    for (int base = 0; base < len; base += BLOCK_N) {
        const int n_valid = min(BLOCK_N, len - base);
        for (int v = tid; v < BLOCK_N * HEAD_DIM; v += NTHREADS) {
            int n = v / HEAD_DIM;
            int d = v - n * HEAD_DIM;
            __nv_bfloat16 val = __float2bfloat16(0.f);
            if (n < n_valid) {
                val = kv[(base + n) * HEAD_DIM + d];
            }
            s.k_s[n][d] = val;
        }
        __syncthreads();

        float resc0 = 1.f;
        float resc1 = 1.f;
        if (warp == 0) {
            float sc0[4] = {0.f, 0.f, 0.f, 0.f};
            float sc1[4] = {0.f, 0.f, 0.f, 0.f};
            uint32_t b0[2], b1[2];
#pragma unroll
            for (int k = 0; k < KT; ++k) {
                int k0 = k * 16;
                uint32_t a[4];
                a[0] = pack_contig(&s.q_s[group][k0 + tig * 2]);
                a[1] = pack_contig(&s.q_s[group + 8][k0 + tig * 2]);
                a[2] = pack_contig(&s.q_s[group][k0 + tig * 2 + 8]);
                a[3] = pack_contig(&s.q_s[group + 8][k0 + tig * 2 + 8]);
                b0[0] = pack_contig(&s.k_s[group][k0 + tig * 2]);
                b0[1] = pack_contig(&s.k_s[group][k0 + tig * 2 + 8]);
                b1[0] = pack_contig(&s.k_s[group + 8][k0 + tig * 2]);
                b1[1] = pack_contig(&s.k_s[group + 8][k0 + tig * 2 + 8]);
                mma16816(sc0, a, b0);
                mma16816(sc1, a, b1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc0[j] *= p.scale_log2;
                sc1[j] *= p.scale_log2;
            }

            float tmax0 = fmaxf(sc0[0], sc0[1]);
            float tmax1 = fmaxf(sc0[2], sc0[3]);
            tmax0 = fmaxf(tmax0, fmaxf(sc1[0], sc1[1]));
            tmax1 = fmaxf(tmax1, fmaxf(sc1[2], sc1[3]));
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, o));
                tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, o));
            }
            if (h0 >= p.num_heads) tmax0 = -INFINITY;
            if (h1 >= p.num_heads) tmax1 = -INFINITY;
            float nm0 = fmaxf(m0, tmax0);
            float nm1 = fmaxf(m1, tmax1);
            resc0 = exp2f(m0 - nm0);
            resc1 = exp2f(m1 - nm1);
            m0 = nm0;
            m1 = nm1;

            float ps0 = 0.f;
            float ps1 = 0.f;
            const bool row0 = h0 < p.num_heads;
            const bool row1 = h1 < p.num_heads;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                int col = nn * 8 + qc;
                bool v0 = row0 && col < n_valid;
                bool v1 = row0 && (col + 1) < n_valid;
                bool v2 = row1 && col < n_valid;
                bool v3 = row1 && (col + 1) < n_valid;
                float p00 = v0 ? exp2f((nn == 0 ? sc0[0] : sc1[0]) - m0) : 0.f;
                float p01 = v1 ? exp2f((nn == 0 ? sc0[1] : sc1[1]) - m0) : 0.f;
                float p10 = v2 ? exp2f((nn == 0 ? sc0[2] : sc1[2]) - m1) : 0.f;
                float p11 = v3 ? exp2f((nn == 0 ? sc0[3] : sc1[3]) - m1) : 0.f;
                ps0 += p00 + p01;
                ps1 += p10 + p11;
                s.p_s[qr0][col] = __float2bfloat16(p00);
                s.p_s[qr0][col + 1] = __float2bfloat16(p01);
                s.p_s[qr1][col] = __float2bfloat16(p10);
                s.p_s[qr1][col + 1] = __float2bfloat16(p11);
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                ps0 += __shfl_xor_sync(0xffffffffu, ps0, o);
                ps1 += __shfl_xor_sync(0xffffffffu, ps1, o);
            }
            l0 = l0 * resc0 + ps0;
            l1 = l1 * resc1 + ps1;
            if (tig == 0) {
                s.resc_s[qr0] = resc0;
                s.resc_s[qr1] = resc1;
            }
        }
        __syncthreads();

        resc0 = s.resc_s[qr0];
        resc1 = s.resc_s[qr1];
#pragma unroll
        for (int d = 0; d < D_PER_WARP / 8; ++d) {
            acc[d][0] *= resc0;
            acc[d][1] *= resc0;
            acc[d][2] *= resc1;
            acc[d][3] *= resc1;
        }

        const int dcol0 = warp * D_PER_WARP;
        int sl = tig * 2;
        uint32_t pa[4];
        pa[0] = pack_contig(&s.p_s[group][tig * 2]);
        pa[1] = pack_contig(&s.p_s[group + 8][tig * 2]);
        pa[2] = pack_contig(&s.p_s[group][tig * 2 + 8]);
        pa[3] = pack_contig(&s.p_s[group + 8][tig * 2 + 8]);
#pragma unroll
        for (int d8 = 0; d8 < D_PER_WARP / 8; ++d8) {
            int dcol = dcol0 + d8 * 8 + group;
            uint32_t kb[2];
            kb[0] = pack2(s.k_s[sl][dcol], s.k_s[sl + 1][dcol]);
            kb[1] = pack2(s.k_s[sl + 8][dcol], s.k_s[sl + 9][dcol]);
            mma16816(acc[d8], pa, kb);
        }
        __syncthreads();

#if 0
        if (warp < 2) {
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
            const int nbase = warp * 8;
#pragma unroll
            for (int k = 0; k < KT; ++k) {
                int k0 = k * 16;
                uint32_t a[4];
                a[0] = pack_contig(&s.q_s[group][k0 + tig * 2]);
                a[1] = pack_contig(&s.q_s[group + 8][k0 + tig * 2]);
                a[2] = pack_contig(&s.q_s[group][k0 + tig * 2 + 8]);
                a[3] = pack_contig(&s.q_s[group + 8][k0 + tig * 2 + 8]);
                uint32_t b[2];
                b[0] = pack_contig(&s.k_s[nbase + group][k0 + tig * 2]);
                b[1] = pack_contig(&s.k_s[nbase + group][k0 + tig * 2 + 8]);
                mma16816(sc, a, b);
            }
            s.qk_s[warp][qr0][qc] = sc[0];
            s.qk_s[warp][qr0][qc + 1] = sc[1];
            s.qk_s[warp][qr1][qc] = sc[2];
            s.qk_s[warp][qr1][qc + 1] = sc[3];
        }
        __syncthreads();

        float resc0 = 1.f;
        float resc1 = 1.f;
        if (warp == 0) {
            float sc0[4] = {0.f, 0.f, 0.f, 0.f};
            float sc1[4] = {0.f, 0.f, 0.f, 0.f};
            sc0[0] = s.qk_s[0][qr0][qc];
            sc0[1] = s.qk_s[0][qr0][qc + 1];
            sc0[2] = s.qk_s[0][qr1][qc];
            sc0[3] = s.qk_s[0][qr1][qc + 1];
            sc1[0] = s.qk_s[1][qr0][qc];
            sc1[1] = s.qk_s[1][qr0][qc + 1];
            sc1[2] = s.qk_s[1][qr1][qc];
            sc1[3] = s.qk_s[1][qr1][qc + 1];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc0[j] *= p.scale_log2;
                sc1[j] *= p.scale_log2;
            }

            float tmax0 = fmaxf(sc0[0], sc0[1]);
            float tmax1 = fmaxf(sc0[2], sc0[3]);
            tmax0 = fmaxf(tmax0, fmaxf(sc1[0], sc1[1]));
            tmax1 = fmaxf(tmax1, fmaxf(sc1[2], sc1[3]));
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, o));
                tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, o));
            }
            if (h0 >= p.num_heads) tmax0 = -INFINITY;
            if (h1 >= p.num_heads) tmax1 = -INFINITY;
            float nm0 = fmaxf(m0, tmax0);
            float nm1 = fmaxf(m1, tmax1);
            resc0 = exp2f(m0 - nm0);
            resc1 = exp2f(m1 - nm1);
            m0 = nm0;
            m1 = nm1;

            float ps0 = 0.f;
            float ps1 = 0.f;
            const bool row0 = h0 < p.num_heads;
            const bool row1 = h1 < p.num_heads;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                int col = nn * 8 + qc;
                bool v0 = row0 && col < n_valid;
                bool v1 = row0 && (col + 1) < n_valid;
                bool v2 = row1 && col < n_valid;
                bool v3 = row1 && (col + 1) < n_valid;
                float p00 = v0 ? exp2f((nn == 0 ? sc0[0] : sc1[0]) - m0) : 0.f;
                float p01 = v1 ? exp2f((nn == 0 ? sc0[1] : sc1[1]) - m0) : 0.f;
                float p10 = v2 ? exp2f((nn == 0 ? sc0[2] : sc1[2]) - m1) : 0.f;
                float p11 = v3 ? exp2f((nn == 0 ? sc0[3] : sc1[3]) - m1) : 0.f;
                ps0 += p00 + p01;
                ps1 += p10 + p11;
                s.p_s[qr0][col] = __float2bfloat16(p00);
                s.p_s[qr0][col + 1] = __float2bfloat16(p01);
                s.p_s[qr1][col] = __float2bfloat16(p10);
                s.p_s[qr1][col + 1] = __float2bfloat16(p11);
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                ps0 += __shfl_xor_sync(0xffffffffu, ps0, o);
                ps1 += __shfl_xor_sync(0xffffffffu, ps1, o);
            }
            l0 = l0 * resc0 + ps0;
            l1 = l1 * resc1 + ps1;
            if (tig == 0) {
                s.resc_s[qr0] = resc0;
                s.resc_s[qr1] = resc1;
            }
        }
        __syncthreads();

        resc0 = s.resc_s[qr0];
        resc1 = s.resc_s[qr1];
#pragma unroll
        for (int d = 0; d < D_PER_WARP / 8; ++d) {
            acc[d][0] *= resc0;
            acc[d][1] *= resc0;
            acc[d][2] *= resc1;
            acc[d][3] *= resc1;
        }

        const int dcol0 = warp * D_PER_WARP;
        int sl = tig * 2;
        uint32_t pa[4];
        pa[0] = pack_contig(&s.p_s[group][tig * 2]);
        pa[1] = pack_contig(&s.p_s[group + 8][tig * 2]);
        pa[2] = pack_contig(&s.p_s[group][tig * 2 + 8]);
        pa[3] = pack_contig(&s.p_s[group + 8][tig * 2 + 8]);
#pragma unroll
        for (int d8 = 0; d8 < D_PER_WARP / 8; ++d8) {
            int dcol = dcol0 + d8 * 8 + group;
            uint32_t kb[2];
            kb[0] = pack2(s.k_s[sl][dcol], s.k_s[sl + 1][dcol]);
            kb[1] = pack2(s.k_s[sl + 8][dcol], s.k_s[sl + 9][dcol]);
            mma16816(acc[d8], pa, kb);
        }
        __syncthreads();
#endif
    }

    if (warp == 0 && tig == 0) {
        s.inv_s[qr0] = 1.f / fmaxf(l0, 1e-20f);
        s.inv_s[qr1] = 1.f / fmaxf(l1, 1e-20f);
    }
    __syncthreads();

    const float inv0 = s.inv_s[qr0];
    const float inv1 = s.inv_s[qr1];
    const int dcol0 = warp * D_PER_WARP;
    __nv_bfloat16 *o0 =
        reinterpret_cast<__nv_bfloat16 *>(p.o_ptr) +
        (int64_t)t * p.out_token_stride + (int64_t)h0 * p.out_head_stride;
    __nv_bfloat16 *o1 =
        reinterpret_cast<__nv_bfloat16 *>(p.o_ptr) +
        (int64_t)t * p.out_token_stride + (int64_t)h1 * p.out_head_stride;
#pragma unroll
    for (int d8 = 0; d8 < D_PER_WARP / 8; ++d8) {
        int col = dcol0 + d8 * 8 + qc;
        if (h0 < p.num_heads) {
            o0[col] = __float2bfloat16(acc[d8][0] * inv0);
            o0[col + 1] = __float2bfloat16(acc[d8][1] * inv0);
        }
        if (h1 < p.num_heads) {
            o1[col] = __float2bfloat16(acc[d8][2] * inv1);
            o1[col + 1] = __float2bfloat16(acc[d8][3] * inv1);
        }
    }
}

}  // namespace mma_stream16

}  // namespace

void run_sparse_mla_prefill_staged(Sparse_mla_prefill_staged_params &params,
                                   cudaStream_t stream) {
    int total = params.num_tokens * params.width * HEAD_DIM;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    if (params.int8_cache) {
        gather_int8_prefill_kernel<<<blocks, threads, 0, stream>>>(params);
    } else {
        gather_fp8_prefill_kernel<<<blocks, threads, 0, stream>>>(params);
    }

    static const bool use_mma = [] {
        const char *e = getenv("FLASH_MLA_PREFILL_MMA");
        return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
    }();
    if (use_mma) {
        static const bool use_stream16 = [] {
            const char *e = getenv("FLASH_MLA_PREFILL_MMA_STREAM16");
            return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
        }();
        if (use_stream16) {
            dim3 grid(params.num_tokens,
                      (params.num_heads + mma_stream16::HEAD_STEP - 1) /
                          mma_stream16::HEAD_STEP);
            int smem = (int)sizeof(mma_stream16::Smem);
            static bool attr_set = false;
            if (!attr_set) {
                cudaFuncSetAttribute(mma_stream16::sparse_prefill_stream16_mma_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                attr_set = true;
            }
            mma_stream16::sparse_prefill_stream16_mma_kernel<<<
                grid, mma_stream16::NTHREADS, smem, stream>>>(params);
            return;
        }
        dim3 grid(params.num_tokens,
                  (params.num_heads + mma_gathered::BLOCK_M - 1) / mma_gathered::BLOCK_M);
        int smem = (int)sizeof(mma_gathered::Smem);
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(mma_gathered::sparse_prefill_gathered_mma_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            attr_set = true;
        }
        mma_gathered::sparse_prefill_gathered_mma_kernel<<<
            grid, mma_gathered::NTHREADS, smem, stream>>>(params);
        return;
    }

    const char *bh_env = getenv("FLASH_MLA_PREFILL_BLOCK_H");
    const char *bn_env = getenv("FLASH_MLA_PREFILL_BLOCK_N");
    int bh = bh_env ? atoi(bh_env) : 32;
    int bn = bn_env ? atoi(bn_env) : 32;
    if (bh == 8) {
        dim3 grid(params.num_tokens, (params.num_heads + 7) / 8);
        sparse_prefill_gathered_kernel<8, 16><<<grid, 8 * 32, 0, stream>>>(params);
    } else if (bh == 16 && bn == 16) {
        dim3 grid(params.num_tokens, (params.num_heads + 15) / 16);
        sparse_prefill_gathered_kernel<16, 16><<<grid, 16 * 32, 0, stream>>>(params);
    } else if (bh == 16) {
        dim3 grid(params.num_tokens, (params.num_heads + 15) / 16);
        sparse_prefill_gathered_kernel<16, 32><<<grid, 16 * 32, 0, stream>>>(params);
    } else if (bn == 16) {
        dim3 grid(params.num_tokens, (params.num_heads + 31) / 32);
        sparse_prefill_gathered_kernel<32, 16><<<grid, 32 * 32, 0, stream>>>(params);
    } else {
        dim3 grid(params.num_tokens, (params.num_heads + 31) / 32);
        sparse_prefill_gathered_kernel<32, 32><<<grid, 32 * 32, 0, stream>>>(params);
    }
}
