// Ampere (sm_86) fused sparse-MLA prefill (DeepSeek-V4-Flash absorbed form).
//
// fp8_ds_mla cache — two phases, one op:
//   1. dequant pass: the WHOLE paged fp8 cache is dequantized once into a dense bf16
//      row buffer [total_slots, 512]. At prefill every cached slot is selected ~T*topk/
//      context times per chunk, so per-(token,slot) in-kernel fp8 decode is massively
//      redundant ALU (profiled: 97M ALU-pipe + 52M FMA-pipe instructions vs 4M HMMA —
//      the software fp8->bf16 cvt chain swamped the SM). Decoding each row exactly once
//      costs ~total_slots*1KB of streaming traffic, amortized to noise across the chunk.
//   2. attention: one CTA per (token, 32-head block). Selected bf16 rows are gathered
//      straight into a 2-deep smem ring with cp.async (zero ALU on the load path),
//      QK/PV run on mma.sync.m16n8k16 across all 8 warps (QK is 2-way k-split so every
//      warp contributes), online softmax per head row, attn_sink merged once, direct
//      output write.
//
// int8_ds_mla cache — single phase, in-kernel dequant (INT8_GATHER=true):
//   the whole-cache pre-pass is GONE (its [total_slots, 512] bf16 buffer was 2 KiB per
//   pool slot — 2.23 GiB at the production 2.3M-slot pool — allocated per call and it
//   OOM'd 24GB ranks). int8 rows are gathered raw with cp.async into an int8 smem
//   staging ring (HALF the random-gather bytes of the bf16 ring), then dequantized
//   int8*scale -> bf16 into the mma ring in smem. Unlike the fp8 cvt chain that
//   motivated the pre-pass, int8 dequant is one cvt+mul per element and the convert
//   sits between the cp.async wait and the QK mma of a 16-row tile — noise next to
//   the tile's 32x16x512 QK/PV work. Numerics are IDENTICAL to the pre-pass
//   (same (float)v * scale -> bf16 rounding per element).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include <type_traits>

#include "flash_mla.h"

namespace {

constexpr int FP8_DIM = 448;
constexpr int ROPE_DIM = 64;
constexpr int SCALE_DIM = 8;
constexpr int TOKEN_DATA_SIZE = FP8_DIM + ROPE_DIM * 2;  // 576 bytes
constexpr int SCALE_GROUP = 64;
constexpr int HEAD_DIM = 512;
constexpr float LOG2E = 1.4426950408889634f;

// ---------------- phase 1: whole-cache fp8 -> bf16 dequant ----------------

__global__ void dequant_cache_kernel(Sparse_mla_prefill_params p, int swa_slots,
                                     int total_slots) {
    const uint8_t *swa_cache = reinterpret_cast<const uint8_t *>(p.swa_cache_ptr);
    const uint8_t *extra_cache = reinterpret_cast<const uint8_t *>(p.extra_cache_ptr);
    __nv_bfloat16 *kv = reinterpret_cast<__nv_bfloat16 *>(p.kv_ptr);
    int64_t total = (int64_t)total_slots * (HEAD_DIM / 2);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)blockDim.x * gridDim.x) {
        int s = (int)(i >> 8);
        int d = ((int)i & 255) * 2;
        bool is_ex = s >= swa_slots;
        int local = is_ex ? s - swa_slots : s;
        const uint8_t *cache = is_ex ? extra_cache : swa_cache;
        int64_t bstride = is_ex ? p.extra_block_stride : p.swa_block_stride;
        int bsize = is_ex ? p.extra_block_size : p.swa_block_size;
        int b = local / bsize, pos = local - b * bsize;
        const uint8_t *blk = cache + (int64_t)b * bstride;
        const uint8_t *data = blk + (int64_t)pos * TOKEN_DATA_SIZE;
        __nv_bfloat16 v0, v1;
        if (d < FP8_DIM) {
            const uint8_t *scale =
                blk + (int64_t)bsize * TOKEN_DATA_SIZE + (int64_t)pos * SCALE_DIM;
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
        __nv_bfloat162 *dst =
            reinterpret_cast<__nv_bfloat162 *>(kv + (int64_t)s * HEAD_DIM + d);
        *dst = __nv_bfloat162(v0, v1);
    }
}

// ---------------- phase 2: gather tensor-core attention ----------------

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
// x4: four 8x8 b16 matrices; lanes 0-7/8-15/16-23/24-31 supply the row addresses.
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
    __nv_bfloat16 k_s[2][BLOCK_N][LD];       // 33.3 KB gather ring (mma reads here)
    const void *src_row_s[2][BLOCK_N];       // bf16 row (fp8 path) or raw int8 row
    __nv_bfloat16 p_s[BLOCK_M][BLOCK_N + 8]; // softmax probs (padded rows)
    float qk_red[2][2][32][4];               // k-split partial C exchange [wm][wn][lane][4]
    // max and sum exchanges are SEPARATE buffers: the sum write happens in the same
    // barrier interval as the other wn-warp's max read (same slots would race)
    float row_max_s[2][BLOCK_M];             // [wn][row]
    float row_sum_s[2][BLOCK_M];             // [wn][row]
    float resc_s[BLOCK_M];
    float inv_s[BLOCK_M];
};

// int8 gather mode: raw int8 rows land here via cp.async, then get dequantized
// into k_s. scale8_s holds the fp32 rowwise scale (0 for invalid/pad rows, which
// makes the dequant of never-written staging bytes an exact zero row).
struct SmemInt8 : Smem {
    int8_t k8_s[2][BLOCK_N][HEAD_DIM];       // 16 KB int8 staging ring
    float scale8_s[2][BLOCK_N];
};
static_assert(sizeof(Smem) % 16 == 0, "k8_s must stay 16B-aligned for cp.async");
// sm_86 and sm_121 both report cudaDevAttrMaxSharedMemoryPerBlockOptin =
// 101376 B (measured on an RTX A5000 and on a GB10), so this bound covers
// consumer Blackwell unchanged — the int8 prefill path needed no resizing to
// run on a DGX Spark.
static_assert(sizeof(SmemInt8) <= 100 * 1024, "sm_86 / sm_121 dynamic smem budget");

template <bool INT8_GATHER>
__global__ void __launch_bounds__(NTHREADS)
sparse_prefill_fused_mma_kernel(__grid_constant__ const Sparse_mla_prefill_params p,
                                const int swa_slots, const int total_slots) {
    extern __shared__ char smem_raw[];
    using SmemT = std::conditional_t<INT8_GATHER, SmemInt8, Smem>;
    SmemT &s = *reinterpret_cast<SmemT *>(smem_raw);
    // grid is (head_blocks, T): blockIdx.x varies fastest, so the two head-block CTAs
    // of one token are ADJACENT in issue order and the second hits L2 for the gather
    const int t = blockIdx.y;
    const int hb = blockIdx.x * BLOCK_M;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp & 1;         // m16 tile
    const int wn = (warp >> 1) & 1;  // QK n8 slice (0..1)
    const int wk = warp >> 2;        // QK k-split half (0..1)
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
    const bool sink = p.attn_sink_ptr != nullptr;
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
    const int len = swa_len + extra_len;
    const int n_tiles = (len + BLOCK_N - 1) / BLOCK_N;
    const __nv_bfloat16 *kv = reinterpret_cast<const __nv_bfloat16 *>(p.kv_ptr);

    // map concatenated selection g -> source row pointer (nullptr = zero row):
    // dense bf16 row (fp8 path, whole-cache dequant buffer) or raw int8 cache row
    // plus its fp32 rowwise scale (int8 path).
    auto resolve = [&](int tile) {
        if (tid < BLOCK_N) {
            int g = tile * BLOCK_N + tid;
            const void *src = nullptr;
            float sc = 0.f;
            if (g < len) {
                bool is_ex = g >= swa_len;
                int sidx = is_ex ? p.extra_indices_ptr[(int64_t)t * p.extra_topk + (g - swa_len)]
                                 : p.swa_indices_ptr[(int64_t)t * p.swa_topk + g];
                int nslots = is_ex ? total_slots - swa_slots : swa_slots;
                if (sidx >= 0 && sidx < nslots) {
                    if constexpr (INT8_GATHER) {
                        const int8_t *cache = reinterpret_cast<const int8_t *>(
                            is_ex ? p.extra_cache_ptr : p.swa_cache_ptr);
                        int64_t bstride = is_ex ? p.extra_block_stride : p.swa_block_stride;
                        int64_t pstride = is_ex ? p.extra_pos_stride : p.swa_pos_stride;
                        const float *scale_base = is_ex ? p.extra_scale_ptr : p.swa_scale_ptr;
                        int64_t sbstride =
                            is_ex ? p.extra_scale_block_stride : p.swa_scale_block_stride;
                        int64_t spstride =
                            is_ex ? p.extra_scale_pos_stride : p.swa_scale_pos_stride;
                        int bsize = is_ex ? p.extra_block_size : p.swa_block_size;
                        int b = sidx / bsize, pos = sidx - b * bsize;
                        src = cache + (int64_t)b * bstride + (int64_t)pos * pstride;
                        sc = scale_base[(int64_t)b * sbstride + (int64_t)pos * spstride];
                    } else {
                        src = kv + ((int64_t)(is_ex ? swa_slots + sidx : sidx)) * HEAD_DIM;
                    }
                }
            }
            s.src_row_s[tile & 1][tid] = src;
            if constexpr (INT8_GATHER) s.scale8_s[tile & 1][tid] = sc;
        }
    };

    // cp.async the tile's rows into the ring. fp8 path: bf16 rows straight into the
    // mma ring (zero-fill invalid/pad rows). int8 path: raw int8 rows into the
    // staging ring (invalid/pad rows are left untouched — their scale of 0 makes the
    // dequant an exact zero row).
    auto issue_copies = [&](int tile) {
        int buf = tile & 1;
        if constexpr (INT8_GATHER) {
#pragma unroll 2
            for (int vp = tid; vp < BLOCK_N * (HEAD_DIM / 16); vp += NTHREADS) {
                int n = vp >> 5, c = (vp & 31) * 16;  // 16 int8 = 16B per chunk
                const int8_t *src = reinterpret_cast<const int8_t *>(s.src_row_s[buf][n]);
                if (src != nullptr) {
                    __pipeline_memcpy_async(&s.k8_s[buf][n][c], src + c, 16);
                }
            }
        } else {
#pragma unroll 4
            for (int vp = tid; vp < BLOCK_N * ROW_CHUNKS; vp += NTHREADS) {
                int n = vp >> 6, c = (vp & 63) * 8;  // 8 bf16 = 16B per chunk
                const __nv_bfloat16 *src =
                    reinterpret_cast<const __nv_bfloat16 *>(s.src_row_s[buf][n]);
                if (src != nullptr) {
                    __pipeline_memcpy_async(&s.k_s[buf][n][c], src + c, 16);
                } else {
                    *reinterpret_cast<uint4 *>(&s.k_s[buf][n][c]) = uint4{0, 0, 0, 0};
                }
            }
        }
        __pipeline_commit();
    };

    if (n_tiles > 0) {
        resolve(0);
        __syncthreads();
        issue_copies(0);
    }

    for (int i = 0; i < n_tiles; ++i) {
        const int buf = i & 1;
        const int n_valid = min(BLOCK_N, len - i * BLOCK_N);
        __pipeline_wait_prior(0);
        __syncthreads();  // ring[buf] ready for all; p_s/qk_red from prev tile consumed
        if constexpr (INT8_GATHER) {
            // dequant staging[buf] -> mma ring[buf]: bf16 = (float)int8 * rowwise scale,
            // identical rounding to the retired whole-cache pre-pass. Rows whose staging
            // bytes were never written carry scale 0 -> exact zero row.
#pragma unroll
            for (int e = tid; e < BLOCK_N * (HEAD_DIM / 4); e += NTHREADS) {
                int n = e >> 7, c = (e & 127) * 4;
                const float sc = s.scale8_s[buf][n];
                char4 v = *reinterpret_cast<const char4 *>(&s.k8_s[buf][n][c]);
                __nv_bfloat162 *dst = reinterpret_cast<__nv_bfloat162 *>(&s.k_s[buf][n][c]);
                dst[0] = __nv_bfloat162(__float2bfloat16((float)v.x * sc),
                                        __float2bfloat16((float)v.y * sc));
                dst[1] = __nv_bfloat162(__float2bfloat16((float)v.z * sc),
                                        __float2bfloat16((float)v.w * sc));
            }
        }
        if (i + 1 < n_tiles) resolve(i + 1);
        __syncthreads();  // src ptrs + dequanted ring[buf] visible (and resolve doesn't race issue below)
        if (i + 1 < n_tiles) issue_copies(i + 1);  // overlaps all math below

        // ---- QK: warp (wm, wn, wk) computes partial m16 x n8 over k half wk ----
        float sc[4];
        {
            float scc[2][4];
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int j = 0; j < 4; ++j) scc[c][j] = 0.f;
            // ldmatrix addresses: A lanes 0-15 rows, 16-31 the k+8 column half;
            // B (x2) lanes 0-7 rows at k0, 8-15 rows at k0+8.
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
            // kb via ldmatrix.trans: M0/M1 = (n0-7, n8-15) at d8 block, M2/M3 at d8+1.
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

    // ---- epilogue: broadcast 1/l from the softmax warps, direct write ----
    if (wk == 0 && wn == 0 && tig == 0) {
        s.inv_s[r0] = 1.f / fmaxf(l0, 1e-20f);
        s.inv_s[r1] = 1.f / fmaxf(l1, 1e-20f);
    }
    __syncthreads();
    const float inv0 = s.inv_s[r0];
    const float inv1 = s.inv_s[r1];
    const int dcol0 = wd * D_PER_WARP;
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

}  // namespace

// Fused tensor-core prefill. fp8_ds_mla: dequantize the whole cache once into a
// dense bf16 [total_slots, 512] buffer, then run the gather-bf16 mma.m16n8k16
// attention kernel. int8_ds_mla: NO pre-pass and NO pool-sized buffer — the same
// attention kernel gathers raw int8 rows and dequantizes them in smem
// (INT8_GATHER=true). This is the only sparse prefill path -- the legacy staged
// two-kernel gather path was removed (dead code, unreachable in production,
// ~4x slower at the true 16k footprint).
void run_sparse_mla_prefill(Sparse_mla_prefill_params &params,
                            cudaStream_t stream) {
    const int swa_slots = params.swa_num_blocks * params.swa_block_size;
    const int extra_slots = (params.extra_cache_ptr != nullptr)
        ? params.extra_num_blocks * params.extra_block_size : 0;
    const int total_slots = swa_slots + extra_slots;

    dim3 grid((params.num_heads + BLOCK_M - 1) / BLOCK_M, params.num_tokens);
    if (params.int8_cache) {
        int smem = (int)sizeof(SmemInt8);
        static bool attr_set_int8 = false;
        if (!attr_set_int8) {
            cudaFuncSetAttribute(sparse_prefill_fused_mma_kernel<true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            attr_set_int8 = true;
        }
        sparse_prefill_fused_mma_kernel<true><<<grid, NTHREADS, smem, stream>>>(
            params, swa_slots, total_slots);
        return;
    }

    {
        int64_t work = (int64_t)total_slots * (HEAD_DIM / 2);
        int threads = 256;
        int64_t blocks64 = (work + threads - 1) / threads;
        int blocks = blocks64 > 4096 ? 4096 : (int)blocks64;
        dequant_cache_kernel<<<blocks, threads, 0, stream>>>(params, swa_slots, total_slots);
    }

    int smem = (int)sizeof(Smem);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(sparse_prefill_fused_mma_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    sparse_prefill_fused_mma_kernel<false><<<grid, NTHREADS, smem, stream>>>(
        params, swa_slots, total_slots);
}
