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
#include <cuda_pipeline.h>
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

// ---- fused-decode pre-pass: dequantize the SELECTED rows once into the dense bf16
// selection scratch [T * sel_width, 512]. One CTA per (selected position g, token t);
// 256 threads, one bf16 pair each. Invalid slot ids inside lens become zero rows
// (matching the oracle: score contribution q.0 = 0, still inside the softmax).
// This removes the per-head-block-CTA gather+fp8-decode redundancy (the attention grid
// re-reads the same selection once per head block) and ALL conversion ALU from the
// attention kernel's critical path.
__global__ void sparse_mla_selection_dequant_kernel(__grid_constant__ const Sparse_mla_decode_params p) {
    const int g = blockIdx.x;
    const int t = blockIdx.y;
    const int swa_len = p.swa_lens_ptr[t];
    const int extra_len = (p.extra_cache_ptr != nullptr) ? p.extra_lens_ptr[t] : 0;
    if (g >= swa_len + extra_len) return;

    const bool is_ex = g >= swa_len;
    const int s = is_ex ? p.extra_indices_ptr[(int64_t)t * p.extra_topk + (g - swa_len)]
                        : p.swa_indices_ptr[(int64_t)t * p.swa_topk + g];
    const uint8_t *cache = reinterpret_cast<const uint8_t *>(is_ex ? p.extra_cache_ptr : p.swa_cache_ptr);
    const int64_t bstride = is_ex ? p.extra_block_stride : p.swa_block_stride;
    const int bsize = is_ex ? p.extra_block_size : p.block_size;
    const int nslots = (is_ex ? p.extra_num_blocks : p.swa_num_blocks) * bsize;

    __nv_bfloat162 *dst = reinterpret_cast<__nv_bfloat162 *>(
        reinterpret_cast<__nv_bfloat16 *>(p.sel_kv_ptr) + ((int64_t)t * p.sel_width + g) * HEAD_DIM);
    const int d = threadIdx.x * 2;

    if (s < 0 || s >= nslots) {
        dst[threadIdx.x] = __nv_bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
        return;
    }
    const uint8_t *data, *scale;
    slot_ptrs(cache, bstride, bsize, s, data, scale);
    __nv_bfloat16 v0, v1;
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
    dst[threadIdx.x] = __nv_bfloat162(v0, v1);
}

// ---- fused-decode attention: identical split-KV FMA attention, but K rows are clean
// bf16 from the selection scratch (cp.async double-buffered ring; zero conversion ALU,
// no per-tile resolve barrier). grid (T, head_blocks, num_splits).
__global__ void __launch_bounds__(NTHREADS)
sparse_mla_decode_fused_split_kernel(__grid_constant__ const Sparse_mla_decode_params p) {
    const int t = blockIdx.x;
    const int sp = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int h = blockIdx.y * BLOCK_H + warp;
    const bool active = h < p.num_heads;

    __shared__ __nv_bfloat16 K_s[2][BLOCK_N][HEAD_DIM];

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

    const __nv_bfloat16 *sel = reinterpret_cast<const __nv_bfloat16 *>(p.sel_kv_ptr)
        + (int64_t)t * p.sel_width * HEAD_DIM;
    const int n_tiles = (g_end > g_start) ? (g_end - g_start + BLOCK_N - 1) / BLOCK_N : 0;
    constexpr int ROW_CHUNKS = HEAD_DIM * 2 / 16;  // 64 x 16B cp.async chunks per row

    auto issue_copies = [&](int tile) {
        const int buf = tile & 1;
        const int base = g_start + tile * BLOCK_N;
#pragma unroll 2
        for (int vp = tid; vp < BLOCK_N * ROW_CHUNKS; vp += NTHREADS) {
            int n = vp >> 6, c = (vp & 63) * 8;
            int g = base + n;
            if (g < g_end)
                __pipeline_memcpy_async(&K_s[buf][n][c], sel + (int64_t)g * HEAD_DIM + c, 16);
        }
        __pipeline_commit();
    };

    if (n_tiles > 0) issue_copies(0);
    for (int i = 0; i < n_tiles; ++i) {
        const int buf = i & 1;
        if (i + 1 < n_tiles) {
            issue_copies(i + 1);      // ring[buf^1]: last read in i-1, fenced by its trailing barrier
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();              // all threads' cp.async for tile i complete + visible
        const int n_valid = min(BLOCK_N, g_end - (g_start + i * BLOCK_N));
        if (active) {
            for (int n = 0; n < n_valid; ++n) {
                float dot = 0.f;
#pragma unroll
                for (int i2 = 0; i2 < VEC; ++i2) dot += q_reg[i2] * __bfloat162float(K_s[buf][n][lane + i2 * 32]);
                float s = warp_sum(dot) * p.scale_log2;
                float m_new = fmaxf(m, s);
                float resc = exp2f(m - m_new);
                float pj = exp2f(s - m_new);
                l = l * resc + pj;
#pragma unroll
                for (int i2 = 0; i2 < VEC; ++i2)
                    acc[i2] = acc[i2] * resc + pj * __bfloat162float(K_s[buf][n][lane + i2 * 32]);
                m = m_new;
            }
        }
        __syncthreads();              // everyone done reading ring[buf] before it is refilled
    }

    if (p.num_splits == 1) {
        if (active) {
            float gm = (p.attn_sink_ptr != nullptr) ? p.attn_sink_ptr[h] * LOG2E : -INFINITY;
            gm = fmaxf(gm, m);
            float gl = l * exp2f(m - gm)
                     + ((p.attn_sink_ptr != nullptr) ? exp2f(p.attn_sink_ptr[h] * LOG2E - gm) : 0.f);
            float sc = exp2f(m - gm) / fmaxf(gl, 1e-20f);
            __nv_bfloat16 *out_ptr = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr)
                + (int64_t)t * p.out_token_stride + (int64_t)h * p.out_head_stride;
#pragma unroll
            for (int i = 0; i < VEC; ++i) out_ptr[lane + i * 32] = __float2bfloat16(acc[i] * sc);
        }
        return;
    }

    if (active) {
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

    __threadfence();
    __syncthreads();
    __shared__ bool s_last;
    if (tid == 0) {
        int cidx = t * (int)gridDim.y + (int)blockIdx.y;
        s_last = (atomicAdd(&p.combine_counter_ptr[cidx], 1) == p.num_splits - 1);
    }
    __syncthreads();
    if (s_last) {
        __threadfence();
        if (active) combine_one_head(p, t, h, lane);
    }
}

// ===================================================================================
// H2: heads-as-M tensor-core decode (mma_dec). The 64 q heads ARE the M dimension
// (BLOCK_M=32 heads/CTA, 2 head-block CTAs), so m16n8k16 HMMA tiles are full even at
// T=1. Split-KV re-parallelizes across the selection exactly like the FMA kernel;
// K rows are clean bf16 from the H1 selection scratch (cp.async ring, ldmatrix
// fragment loads, V==K read from the same ring). Structure ported from the fused
// prefill kernel (csrc/flash_sparse_mla_prefill_fused_sm80.cu), whose per-CTA shape
// at T=1 is exactly this decode shape; new here: split-KV partials + fused combine.
namespace mma_dec {

constexpr int BLOCK_M = 32;   // heads per CTA (2 CTAs cover H=64)
constexpr int BLOCK_N = 16;   // selected slots per ring slot
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;
constexpr int LD = HEAD_DIM + 8;         // bf16 pad: 1040B row stride dodges bank conflicts
constexpr int KT_HALF = 16;              // QK k-tiles per k-split half (256 dims)
constexpr int D_PER_WARP = HEAD_DIM / 4; // PV: 4 d-slices of 128
constexpr int ROW_CHUNKS = HEAD_DIM * 2 / 16;  // 64 x 16B cp.async chunks per row

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t r[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Smem {
    __nv_bfloat16 q_s[BLOCK_M][LD];          // 33.3 KB
    __nv_bfloat16 k_s[2][BLOCK_N][LD];       // 33.3 KB gather ring (cp.async lands here)
    __nv_bfloat16 p_s[BLOCK_M][BLOCK_N + 8]; // softmax probs (padded rows)
    float qk_red[2][2][32][4];               // k-split partial C exchange [wm][wn][lane][4]
    float row_max_s[2][BLOCK_M];             // [wn][row] (separate from sums: see prefill)
    float row_sum_s[2][BLOCK_M];
    float resc_s[BLOCK_M];
    float inv_s[BLOCK_M];
};

// grid (T, head_blocks=2, num_splits); 8 warps.
__global__ void __launch_bounds__(NTHREADS)
sparse_mla_decode_mma_kernel(__grid_constant__ const Sparse_mla_decode_params p) {
    extern __shared__ char smem_raw[];
    Smem &s = *reinterpret_cast<Smem *>(smem_raw);
    const int t = blockIdx.x;
    const int hb = blockIdx.y * BLOCK_M;
    const int sp = blockIdx.z;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp & 1;         // m16 tile
    const int wn = (warp >> 1) & 1;  // QK n8 slice
    const int wk = warp >> 2;        // QK k-split half
    const int wd = warp >> 1;        // PV d128 slice (0..3)
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int qc = tig * 2;

    // ---- load q tile ----
    for (int e = tid; e < BLOCK_M * HEAD_DIM; e += NTHREADS) {
        int r = e >> 9, c = e & 511;
        int h = hb + r;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (h < p.num_heads) {
            const __nv_bfloat16 *qp = reinterpret_cast<const __nv_bfloat16 *>(p.q_ptr) +
                (int64_t)t * p.q_token_stride + (int64_t)h * p.q_head_stride;
            v = qp[c];
        }
        s.q_s[r][c] = v;
    }

    const int r0 = wm * 16 + group;
    const int r1 = r0 + 8;
    const int h0 = hb + r0;
    const int h1 = hb + r1;
    // attn_sink is folded here ONLY on the single-split direct path; the split path
    // adds it once in the combine.
    const bool sink = (p.num_splits == 1) && (p.attn_sink_ptr != nullptr);
    float m0 = (sink && h0 < p.num_heads) ? p.attn_sink_ptr[h0] * LOG2E : -INFINITY;
    float m1 = (sink && h1 < p.num_heads) ? p.attn_sink_ptr[h1] * LOG2E : -INFINITY;
    float l0 = (sink && h0 < p.num_heads) ? 1.f : 0.f;
    float l1 = (sink && h1 < p.num_heads) ? 1.f : 0.f;

    float acc[D_PER_WARP / 8][4];
#pragma unroll
    for (int d = 0; d < D_PER_WARP / 8; ++d)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[d][j] = 0.f;

    const int swa_len = p.swa_lens_ptr[t];
    const int extra_len = (p.extra_cache_ptr != nullptr) ? p.extra_lens_ptr[t] : 0;
    const int total = swa_len + extra_len;
    const int chunk = (total + p.num_splits - 1) / p.num_splits;
    const int g_start = sp * chunk;
    const int g_end = min(g_start + chunk, total);
    const int n_tiles = (g_end > g_start) ? (g_end - g_start + BLOCK_N - 1) / BLOCK_N : 0;
    const __nv_bfloat16 *sel = reinterpret_cast<const __nv_bfloat16 *>(p.sel_kv_ptr)
        + (int64_t)t * p.sel_width * HEAD_DIM;

    // cp.async the tile's bf16 scratch rows into the ring (zero-fill pad rows: they
    // enter the tile row-max as score 0, which only tightens the online rescale --
    // their probs are masked to 0 before p_s/PV, same convention as the prefill kernel)
    auto issue_copies = [&](int tile) {
        const int buf = tile & 1;
        const int base = g_start + tile * BLOCK_N;
#pragma unroll 4
        for (int vp = tid; vp < BLOCK_N * ROW_CHUNKS; vp += NTHREADS) {
            int n = vp >> 6, c = (vp & 63) * 8;
            int g = base + n;
            if (g < g_end) {
                __pipeline_memcpy_async(&s.k_s[buf][n][c], sel + (int64_t)g * HEAD_DIM + c, 16);
            } else {
                *reinterpret_cast<uint4 *>(&s.k_s[buf][n][c]) = uint4{0, 0, 0, 0};
            }
        }
        __pipeline_commit();
    };

    if (n_tiles > 0) issue_copies(0);
    __syncthreads();  // q_s ready (and ring[0] issue ordered before its wait below)

    for (int i = 0; i < n_tiles; ++i) {
        const int buf = i & 1;
        const int n_valid = min(BLOCK_N, g_end - (g_start + i * BLOCK_N));
        __pipeline_wait_prior(0);
        __syncthreads();  // ring[buf] ready for all; p_s/qk_red from prev tile consumed
        if (i + 1 < n_tiles) issue_copies(i + 1);  // overlaps all math below

        // ---- QK: warp (wm, wn, wk) computes partial m16 x n8 over k half wk ----
        float sc[4];
        {
            float scc[2][4];
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int j = 0; j < 4; ++j) scc[c][j] = 0.f;
            const uint32_t a_addr0 =
                smem_addr(&s.q_s[wm * 16 + (lane & 15)][wk * 256 + ((lane >> 4) << 3)]);
            const uint32_t b_addr0 = smem_addr(
                &s.k_s[buf][wn * 8 + (lane & 7)][wk * 256 + (((lane >> 3) & 1) << 3)]);
#pragma unroll
            for (int k = 0; k < KT_HALF; ++k) {
                uint32_t a[4], b[2];
                ldsm_x4(a, a_addr0 + k * 32);
                ldsm_x2(b, b_addr0 + k * 32);
                mma16816(scc[k & 1], a, b);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) sc[j] = scc[0][j] + scc[1][j];
        }
        if (wk == 1) {
#pragma unroll
            for (int j = 0; j < 4; ++j) s.qk_red[wm][wn][lane][j] = sc[j];
        }
        __syncthreads();

        if (wk == 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                sc[j] = (sc[j] + s.qk_red[wm][wn][lane][j]) * p.scale_log2;

            float tmax0 = fmaxf(sc[0], sc[1]);
            float tmax1 = fmaxf(sc[2], sc[3]);
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, o));
                tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, o));
            }
            if (tig == 0) {
                s.row_max_s[wn][r0] = tmax0;
                s.row_max_s[wn][r1] = tmax1;
            }
        }
        __syncthreads();

        if (wk == 0) {
            const float tile_m0 = fmaxf(s.row_max_s[0][r0], s.row_max_s[1][r0]);
            const float tile_m1 = fmaxf(s.row_max_s[0][r1], s.row_max_s[1][r1]);
            const float nm0 = fmaxf(m0, tile_m0);
            const float nm1 = fmaxf(m1, tile_m1);
            const float resc0 = exp2f(m0 - nm0);
            const float resc1 = exp2f(m1 - nm1);
            m0 = nm0;
            m1 = nm1;
            if (wn == 0 && tig == 0) {
                s.resc_s[r0] = resc0;
                s.resc_s[r1] = resc1;
            }
            int col = wn * 8 + qc;
            bool v0 = col < n_valid, v1 = (col + 1) < n_valid;
            float p00 = v0 ? exp2f(sc[0] - m0) : 0.f;
            float p01 = v1 ? exp2f(sc[1] - m0) : 0.f;
            float p10 = v0 ? exp2f(sc[2] - m1) : 0.f;
            float p11 = v1 ? exp2f(sc[3] - m1) : 0.f;
            float ps0 = p00 + p01;
            float ps1 = p10 + p11;
            s.p_s[r0][col] = __float2bfloat16(p00);
            s.p_s[r0][col + 1] = __float2bfloat16(p01);
            s.p_s[r1][col] = __float2bfloat16(p10);
            s.p_s[r1][col + 1] = __float2bfloat16(p11);
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                ps0 += __shfl_xor_sync(0xffffffffu, ps0, o);
                ps1 += __shfl_xor_sync(0xffffffffu, ps1, o);
            }
            if (tig == 0) {
                s.row_sum_s[wn][r0] = ps0;
                s.row_sum_s[wn][r1] = ps1;
            }
            l0 = l0 * resc0;
            l1 = l1 * resc1;
        }
        __syncthreads();

        if (wk == 0) {
            l0 += s.row_sum_s[0][r0] + s.row_sum_s[1][r0];
            l1 += s.row_sum_s[0][r1] + s.row_sum_s[1][r1];
        }

        // ---- PV: warp (wm, wd) accumulates m16 x d128 (V == K, contraction n16) ----
        const float resc0 = s.resc_s[r0];
        const float resc1 = s.resc_s[r1];
#pragma unroll
        for (int d = 0; d < D_PER_WARP / 8; ++d) {
            acc[d][0] *= resc0;
            acc[d][1] *= resc0;
            acc[d][2] *= resc1;
            acc[d][3] *= resc1;
        }
        {
            uint32_t pa[4];
            ldsm_x4(pa, smem_addr(&s.p_s[wm * 16 + (lane & 15)][(lane >> 4) << 3]));
            const int dcol0 = wd * D_PER_WARP;
            const uint32_t kb_addr0 = smem_addr(
                &s.k_s[buf][lane & 15][dcol0 + ((lane >> 4) << 3)]);
#pragma unroll
            for (int d8 = 0; d8 < D_PER_WARP / 8; d8 += 2) {
                uint32_t kb[4];
                ldsm_x4_trans(kb, kb_addr0 + d8 * 16);
                mma16816(acc[d8], pa, kb);
                mma16816(acc[d8 + 1], pa, kb + 2);
            }
        }
        __syncthreads();  // ring[buf]/p_s/qk_red consumed; next iteration may overwrite
    }

    const int dcol0 = wd * D_PER_WARP;
    if (p.num_splits == 1) {
        // direct write (sink already folded into m/l)
        if (wk == 0 && wn == 0 && tig == 0) {
            s.inv_s[r0] = 1.f / fmaxf(l0, 1e-20f);
            s.inv_s[r1] = 1.f / fmaxf(l1, 1e-20f);
        }
        __syncthreads();
        const float inv0 = s.inv_s[r0];
        const float inv1 = s.inv_s[r1];
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
        return;
    }

    // ---- split partials: un-normalized acc (bf16) + {m, l} per (t, h, sp) ----
    if (h0 < p.num_heads) {
        __nv_bfloat16 *oacc = reinterpret_cast<__nv_bfloat16 *>(p.oaccum_ptr)
            + (((int64_t)t * p.num_heads + h0) * p.num_splits + sp) * HEAD_DIM;
#pragma unroll
        for (int d8 = 0; d8 < D_PER_WARP / 8; ++d8) {
            int col = dcol0 + d8 * 8 + qc;
            oacc[col] = __float2bfloat16(acc[d8][0]);
            oacc[col + 1] = __float2bfloat16(acc[d8][1]);
        }
    }
    if (h1 < p.num_heads) {
        __nv_bfloat16 *oacc = reinterpret_cast<__nv_bfloat16 *>(p.oaccum_ptr)
            + (((int64_t)t * p.num_heads + h1) * p.num_splits + sp) * HEAD_DIM;
#pragma unroll
        for (int d8 = 0; d8 < D_PER_WARP / 8; ++d8) {
            int col = dcol0 + d8 * 8 + qc;
            oacc[col] = __float2bfloat16(acc[d8][2]);
            oacc[col + 1] = __float2bfloat16(acc[d8][3]);
        }
    }
    if (wk == 0 && wn == 0 && tig == 0) {
        if (h0 < p.num_heads) {
            int64_t pm = ((int64_t)t * p.num_heads + h0) * p.num_splits + sp;
            p.mlse_ptr[pm * 2] = m0;
            p.mlse_ptr[pm * 2 + 1] = l0;
        }
        if (h1 < p.num_heads) {
            int64_t pm = ((int64_t)t * p.num_heads + h1) * p.num_splits + sp;
            p.mlse_ptr[pm * 2] = m1;
            p.mlse_ptr[pm * 2 + 1] = l1;
        }
    }

    // ---- fused combine: last split-CTA for this (t, head_block) merges all splits ----
    __threadfence();
    __syncthreads();
    __shared__ bool s_last;
    if (tid == 0) {
        int cidx = t * (int)gridDim.y + (int)blockIdx.y;
        s_last = (atomicAdd(&p.combine_counter_ptr[cidx], 1) == p.num_splits - 1);
    }
    __syncthreads();
    if (s_last) {
        __threadfence();
        for (int r = warp; r < BLOCK_M; r += NWARPS) {
            int h = hb + r;
            if (h < p.num_heads) combine_one_head(p, t, h, lane);
        }
    }
}

}  // namespace mma_dec

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
    // cp.async double-buffer: raw token bytes (data + scale) staged from global while the
    // previous tile computes; decode (fp8->bf16) reads from smem, off the global-latency path.
    __shared__ __align__(16) uint8_t raw_s[2][BLOCK_N][TOKEN_DATA_SIZE];
    __shared__ uint8_t rawsc_s[2][BLOCK_N][SCALE_DIM];
    __shared__ const uint8_t *src_data_s[2][BLOCK_N];
    __shared__ const uint8_t *src_scale_s[2][BLOCK_N];
    __shared__ int rslot_s[2][BLOCK_N];

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

    const int n_tiles = (g_end > g_start) ? (g_end - g_start + BLOCK_N - 1) / BLOCK_N : 0;

    // resolve tile's slot indices + global source pointers into smem[buf] (8 threads idle ok).
    auto resolve = [&](int tile, int buf) {
        if (tid < BLOCK_N) {
            int g = g_start + tile * BLOCK_N + tid;
            int slot = -1; const uint8_t *dptr = nullptr, *sptr = nullptr;
            if (g < g_end) {
                int is_ex = (g >= swa_len);
                int s = is_ex ? p.extra_indices_ptr[(int64_t)t * p.extra_topk + (g - swa_len)]
                              : p.swa_indices_ptr[(int64_t)t * p.swa_topk + g];
                const uint8_t *cache = is_ex ? extra_cache : swa_cache;
                int64_t bstride = is_ex ? p.extra_block_stride : p.swa_block_stride;
                int bsize = is_ex ? p.extra_block_size : p.block_size;
                int nslots = is_ex ? extra_slots : swa_slots;
                if (s >= 0 && s < nslots) { slot = s; slot_ptrs(cache, bstride, bsize, s, dptr, sptr); }
            }
            rslot_s[buf][tid] = slot;
            src_data_s[buf][tid] = dptr;
            src_scale_s[buf][tid] = sptr;
        }
    };

    // cp.async the raw bytes for buf (8-byte chunks: safe for any block_size alignment).
    auto issue_copies = [&](int buf) {
        for (int vp = tid; vp < BLOCK_N * (TOKEN_DATA_SIZE / 8); vp += NTHREADS) {
            int n = vp / (TOKEN_DATA_SIZE / 8), c = (vp - n * (TOKEN_DATA_SIZE / 8)) * 8;
            const uint8_t *src = src_data_s[buf][n];
            if (src != nullptr) __pipeline_memcpy_async(&raw_s[buf][n][c], src + c, 8);
        }
        if (tid < BLOCK_N) {
            const uint8_t *src = src_scale_s[buf][tid];
            if (src != nullptr) __pipeline_memcpy_async(&rawsc_s[buf][tid][0], src, SCALE_DIM);
        }
        __pipeline_commit();
    };

    // decode staged raw bytes (smem) -> K_s. 2 dims/thread, cvt_fp8x2 per pair; RoPE direct bf16.
    auto decode_tile = [&](int buf, int n_valid) {
        for (int vp = tid; vp < BLOCK_N * (HEAD_DIM / 2); vp += NTHREADS) {
            int n = vp / (HEAD_DIM / 2), d = (vp - n * (HEAD_DIM / 2)) * 2;
            __nv_bfloat16 v0 = __float2bfloat16(0.f), v1 = v0;
            if (n < n_valid && rslot_s[buf][n] >= 0) {
                const uint8_t *data = raw_s[buf][n];
                if (d < FP8_DIM) {
                    unsigned short raw2 = *reinterpret_cast<const unsigned short *>(data + d);
                    __half2 h2 = __half2(__nv_cvt_fp8x2_to_halfraw2(raw2, __NV_E4M3));
                    float sc = ldexpf(1.f, (int)rawsc_s[buf][n][d / SCALE_GROUP] - 127);
                    v0 = __float2bfloat16(__low2float(h2) * sc);
                    v1 = __float2bfloat16(__high2float(h2) * sc);
                } else {
                    const __nv_bfloat16 *rope = reinterpret_cast<const __nv_bfloat16 *>(data + FP8_DIM);
                    v0 = rope[d - FP8_DIM];
                    v1 = rope[d + 1 - FP8_DIM];
                }
            }
            K_s[n][d] = v0;
            K_s[n][d + 1] = v1;
        }
    };

    if (n_tiles > 0) {
        resolve(0, 0);
        __syncthreads();
        issue_copies(0);
        for (int i = 0; i < n_tiles; ++i) {
            int buf = i & 1;
            if (i + 1 < n_tiles) {
                resolve(i + 1, (i + 1) & 1);
                __syncthreads();           // src ptrs for i+1 visible before issuing its copies
                issue_copies((i + 1) & 1);
                __pipeline_wait_prior(1);  // keep i+1 in flight, wait tile i
            } else {
                __pipeline_wait_prior(0);
            }
            __syncthreads();               // all threads' cp.async for tile i complete + visible
            int n_valid = min(BLOCK_N, g_end - (g_start + i * BLOCK_N));
            decode_tile(buf, n_valid);
            __syncthreads();
            if (active) {
                for (int n = 0; n < n_valid; ++n) {
                    float dot = 0.f;
#pragma unroll
                    for (int i2 = 0; i2 < VEC; ++i2) dot += q_reg[i2] * __bfloat162float(K_s[n][lane + i2 * 32]);
                    float s = warp_sum(dot) * p.scale_log2;
                    float m_new = fmaxf(m, s);
                    float resc = exp2f(m - m_new);
                    float pj = exp2f(s - m_new);
                    l = l * resc + pj;
#pragma unroll
                    for (int i2 = 0; i2 < VEC; ++i2)
                        acc[i2] = acc[i2] * resc + pj * __bfloat162float(K_s[n][lane + i2 * 32]);
                    m = m_new;
                }
            }
            __syncthreads();
        }
    }

    // num_splits == 1 (prefill / large-T regime): this CTA holds the whole selection for
    // (t, h) -> normalize + apply attn_sink + write the output directly, in-register. No
    // oaccum global round-trip, no atomic, no combine (and the binding skips those allocs).
    if (p.num_splits == 1) {
        if (active) {
            float gm = (p.attn_sink_ptr != nullptr) ? p.attn_sink_ptr[h] * LOG2E : -INFINITY;
            gm = fmaxf(gm, m);
            float gl = l * exp2f(m - gm)
                     + ((p.attn_sink_ptr != nullptr) ? exp2f(p.attn_sink_ptr[h] * LOG2E - gm) : 0.f);
            float sc = exp2f(m - gm) / fmaxf(gl, 1e-20f);
            __nv_bfloat16 *out_ptr = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr)
                + (int64_t)t * p.out_token_stride + (int64_t)h * p.out_head_stride;
#pragma unroll
            for (int i = 0; i < VEC; ++i) out_ptr[lane + i * 32] = __float2bfloat16(acc[i] * sc);
        }
        return;
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

// ===================================================================================
// PREFILL tensor-core path (num_splits==1). Batched mma.sync.m16n8k16 QK+PV over BLOCK_M=16
// heads/CTA; register-resident O accumulator, two-stream, attn_sink + variable lens, direct
// output write. At T=1 decode this starves the SMs (only head_blocks CTAs); at large-T prefill
// the T*head_blocks CTAs fill the SMs, so the parallel MMA beats the FMA path's serial per-slot
// warp-reductions. Recovered from the flash-2 mma decode experiment (correct fragment layouts).
namespace mma_pf {

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 64;
constexpr int NWARPS = 4;
constexpr int NTHREADS = NWARPS * 32;            // 128
constexpr int D_PER_WARP = HEAD_DIM / NWARPS;    // 128 -> 16 d8 tiles (PV acc)
constexpr int KS_LD = HEAD_DIM + 8;              // pad to dodge bank conflicts
constexpr int KT = HEAD_DIM / 16;                // 32 k-tiles (QK contraction)
constexpr int NT = BLOCK_N / 16;                 // 4 n16-tiles (PV contraction)

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 a, __nv_bfloat16 b) {
    return (uint32_t)(*reinterpret_cast<uint16_t *>(&a)) |
           ((uint32_t)(*reinterpret_cast<uint16_t *>(&b)) << 16);
}
__device__ __forceinline__ uint32_t pack_contig(const __nv_bfloat16 *p) {
    return *reinterpret_cast<const uint32_t *>(p);
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct StreamArgs {
    const uint8_t *cache;
    int64_t block_stride;
    int block_size, num_blocks, topk;
    const int *indices, *lens;
};
struct Smem {
    __nv_bfloat16 q_s[BLOCK_M][HEAD_DIM];   // 16 KB
    __nv_bfloat16 k_s[BLOCK_N][KS_LD];      // 64*520*2 = 65 KB
    float row_max[BLOCK_M];
    float row_scratch[NWARPS][BLOCK_M];
    __nv_bfloat16 p_s[BLOCK_M][BLOCK_N];    // softmax probs for PV
};

// m16n8 C-fragment: thread holds rows {lane/4, lane/4+8}, cols (lane%4)*2 + {0,1}.
__global__ void __launch_bounds__(NTHREADS)
sparse_mla_prefill_mma_kernel(__grid_constant__ const Sparse_mla_decode_params p) {
    extern __shared__ char smem_raw[];
    Smem &s = *reinterpret_cast<Smem *>(smem_raw);
    const int t = blockIdx.x;
    const int hb = blockIdx.y * BLOCK_M;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int qr0 = lane >> 2;          // C-fragment row group (0..7)
    const int qc = (lane & 3) * 2;      // C-fragment col base (0,2,4,6)

    for (int e = tid; e < BLOCK_M * HEAD_DIM; e += NTHREADS) {
        int r = e / HEAD_DIM, c = e - r * HEAD_DIM, h = hb + r;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (h < p.num_heads) {
            const __nv_bfloat16 *qp = reinterpret_cast<const __nv_bfloat16 *>(p.q_ptr)
                + (int64_t)t * p.q_token_stride + (int64_t)h * p.q_head_stride;
            v = qp[c];
        }
        s.q_s[r][c] = v;
    }

    float acc[D_PER_WARP / 8][4];
#pragma unroll
    for (int d = 0; d < D_PER_WARP / 8; ++d)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[d][j] = 0.f;

    float m0, m1, l0, l1;
    {
        int h0 = hb + qr0, h1 = hb + qr0 + 8;
        bool sink = p.attn_sink_ptr != nullptr;
        m0 = (sink && h0 < p.num_heads) ? p.attn_sink_ptr[h0] * LOG2E : -INFINITY;
        m1 = (sink && h1 < p.num_heads) ? p.attn_sink_ptr[h1] * LOG2E : -INFINITY;
        l0 = (sink && h0 < p.num_heads) ? 1.f : 0.f;
        l1 = (sink && h1 < p.num_heads) ? 1.f : 0.f;
    }
    __syncthreads();

    const StreamArgs streams[2] = {
        {reinterpret_cast<const uint8_t *>(p.swa_cache_ptr), p.swa_block_stride, p.block_size,
         p.swa_num_blocks, p.swa_topk, p.swa_indices_ptr, p.swa_lens_ptr},
        {reinterpret_cast<const uint8_t *>(p.extra_cache_ptr), p.extra_block_stride, p.extra_block_size,
         p.extra_num_blocks, p.extra_topk, p.extra_indices_ptr, p.extra_lens_ptr},
    };

    for (int sidx = 0; sidx < 2; ++sidx) {
        const StreamArgs st = streams[sidx];
        if (st.cache == nullptr) continue;
        int len = st.lens[t];
        const int *idx_row = st.indices + (int64_t)t * st.topk;
        int num_slots = st.num_blocks * st.block_size;

        for (int base = 0; base < len; base += BLOCK_N) {
            int n_valid = min(BLOCK_N, len - base);
            for (int v = tid; v < BLOCK_N * HEAD_DIM; v += NTHREADS) {
                int n = v / HEAD_DIM, d = v - n * HEAD_DIM;
                float val = 0.f;
                if (n < n_valid) {
                    int slot = idx_row[base + n];
                    if (slot >= 0 && slot < num_slots) {
                        const uint8_t *data, *scale;
                        slot_ptrs(st.cache, st.block_stride, st.block_size, slot, data, scale);
                        val = decode_k_dim(data, scale, d);
                    }
                }
                s.k_s[n][d] = __float2bfloat16(val);
            }
            __syncthreads();

            float sc[2][4];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int j = 0; j < 4; ++j) sc[nn][j] = 0.f;
            const int ncol0 = warp * 16;
            const int group = lane >> 2, tig = lane & 3;
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
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int j = 0; j < 4; ++j) sc[nn][j] *= p.scale_log2;

            float tmax0 = fmaxf(fmaxf(sc[0][0], sc[0][1]), fmaxf(sc[1][0], sc[1][1]));
            float tmax1 = fmaxf(fmaxf(sc[0][2], sc[0][3]), fmaxf(sc[1][2], sc[1][3]));
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, o));
                tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, o));
            }
            if ((lane & 3) == 0) { s.row_scratch[warp][qr0] = tmax0; s.row_scratch[warp][qr0 + 8] = tmax1; }
            __syncthreads();
            float tile_m0 = -INFINITY, tile_m1 = -INFINITY;
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                tile_m0 = fmaxf(tile_m0, s.row_scratch[w][qr0]);
                tile_m1 = fmaxf(tile_m1, s.row_scratch[w][qr0 + 8]);
            }
            float nm0 = fmaxf(m0, tile_m0), nm1 = fmaxf(m1, tile_m1);
            float resc0 = exp2f(m0 - nm0), resc1 = exp2f(m1 - nm1);
            m0 = nm0; m1 = nm1;

            float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                float p00 = exp2f(sc[nn][0] - m0), p01 = exp2f(sc[nn][1] - m0);
                float p10 = exp2f(sc[nn][2] - m1), p11 = exp2f(sc[nn][3] - m1);
                int col = ncol0 + nn * 8 + qc;
                bool v0 = (col) < n_valid, v1 = (col + 1) < n_valid;
                p00 = v0 ? p00 : 0.f; p01 = v1 ? p01 : 0.f;
                p10 = v0 ? p10 : 0.f; p11 = v1 ? p11 : 0.f;
                ps0 += p00 + p01; ps1 += p10 + p11;
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
            if ((lane & 3) == 0) { s.row_scratch[warp][qr0] = ps0; s.row_scratch[warp][qr0 + 8] = ps1; }
            __syncthreads();
            float fps0 = 0.f, fps1 = 0.f;
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) { fps0 += s.row_scratch[w][qr0]; fps1 += s.row_scratch[w][qr0 + 8]; }
            l0 = l0 * resc0 + fps0;
            l1 = l1 * resc1 + fps1;

#pragma unroll
            for (int d = 0; d < D_PER_WARP / 8; ++d) {
                acc[d][0] *= resc0; acc[d][1] *= resc0;
                acc[d][2] *= resc1; acc[d][3] *= resc1;
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
    }

    int h0 = hb + qr0, h1 = hb + qr0 + 8;
    float inv0 = 1.f / fmaxf(l0, 1e-20f), inv1 = 1.f / fmaxf(l1, 1e-20f);
    const int dcol0 = warp * D_PER_WARP;
    __nv_bfloat16 *o0 = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr)
        + (int64_t)t * p.out_token_stride + (int64_t)h0 * p.out_head_stride;
    __nv_bfloat16 *o1 = reinterpret_cast<__nv_bfloat16 *>(p.o_ptr)
        + (int64_t)t * p.out_token_stride + (int64_t)h1 * p.out_head_stride;
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

}  // namespace mma_pf

}  // namespace

bool sparse_mla_decode_fused_enabled() {
    static const bool on = [] {
        const char *e = getenv("FLASH_MLA_DECODE_FUSED");
        return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
    }();
    return on;
}

bool sparse_mla_decode_mma_enabled() {
    static const bool on = [] {
        const char *e = getenv("FLASH_MLA_DECODE_MMA");
        return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
    }();
    return on;
}

void run_sparse_mla_decode(Sparse_mla_decode_params &params, cudaStream_t stream) {
    // Prefill (num_splits==1): a batched tensor-core QK+PV kernel (mma_pf) is available but is
    // ~2x SLOWER than the FMA path here -- this workload is gather-MEMORY-LATENCY bound, not
    // QK-compute bound (proven: halving gather volume via BLOCK_H=32 gave 0 gain). The FMA path
    // hides that latency with cp.async + 16 warps; mma_pf's 64-slot tiles cost 85KB smem -> 1
    // CTA/SM + only 4 warps, so its gather latency is exposed. Same lesson as tensor-core decode.
    // Default OFF; opt in with FLASH_MLA_PREFILL_MMA=1 for the batched/larger-BLOCK_M future.
    static const bool prefill_mma = [] {
        const char *e = getenv("FLASH_MLA_PREFILL_MMA");
        return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
    }();
    if (params.num_splits == 1 && prefill_mma) {
        dim3 grid(params.num_tokens, (params.num_heads + mma_pf::BLOCK_M - 1) / mma_pf::BLOCK_M);
        int smem = (int)sizeof(mma_pf::Smem);
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(mma_pf::sparse_mla_prefill_mma_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            attr_set = true;
        }
        mma_pf::sparse_mla_prefill_mma_kernel<<<grid, mma_pf::NTHREADS, smem, stream>>>(params);
        return;
    }

    int head_blocks = (params.num_heads + BLOCK_H - 1) / BLOCK_H;
    if (params.num_splits > 1)  // num_splits==1 fast path writes output directly, no counter/combine
        cudaMemsetAsync(params.combine_counter_ptr, 0,
                        (size_t)params.num_tokens * head_blocks * sizeof(int), stream);
    dim3 grid(params.num_tokens, head_blocks, params.num_splits);
    if (params.sel_kv_ptr != nullptr && params.sel_width > 0) {
        // fused decode: selection-scratch dequant pre-pass, then bf16 attention
        dim3 dq_grid(params.sel_width, params.num_tokens);
        sparse_mla_selection_dequant_kernel<<<dq_grid, HEAD_DIM / 2, 0, stream>>>(params);
        // heads-as-M tensor-core attention (H2); FLASH_MLA_DECODE_MMA=0 falls back to
        // the FMA fused kernel (H1) for A/B comparison.
        if (sparse_mla_decode_mma_enabled()) {
            dim3 mgrid(params.num_tokens,
                       (params.num_heads + mma_dec::BLOCK_M - 1) / mma_dec::BLOCK_M,
                       params.num_splits);
            int smem = (int)sizeof(mma_dec::Smem);
            static bool attr_set = false;
            if (!attr_set) {
                cudaFuncSetAttribute(mma_dec::sparse_mla_decode_mma_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                attr_set = true;
            }
            mma_dec::sparse_mla_decode_mma_kernel<<<mgrid, mma_dec::NTHREADS, smem, stream>>>(params);
            return;
        }
        sparse_mla_decode_fused_split_kernel<<<grid, NTHREADS, 0, stream>>>(params);
        return;
    }
    sparse_mla_decode_split_kernel<<<grid, NTHREADS, 0, stream>>>(params);
}
