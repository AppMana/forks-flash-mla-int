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
    sparse_mla_decode_split_kernel<<<grid, NTHREADS, 0, stream>>>(params);
}
