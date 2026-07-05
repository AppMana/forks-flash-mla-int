#include <cuda_runtime.h>
#include <cstdint>

#include "flash_mla.h"

namespace {

__device__ __forceinline__ uint32_t pack_s8x4(int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
    return (static_cast<uint32_t>(static_cast<uint8_t>(x0))      ) |
           (static_cast<uint32_t>(static_cast<uint8_t>(x1)) <<  8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(x2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(x3)) << 24);
}

__device__ __forceinline__ void mma_m16n8k32_s8s8(int32_t c[4], const uint32_t a[4],
                                                  const uint32_t b[2]) {
    uint32_t d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(static_cast<uint32_t>(c[0])), "r"(static_cast<uint32_t>(c[1])),
          "r"(static_cast<uint32_t>(c[2])), "r"(static_cast<uint32_t>(c[3])));
    c[0] = static_cast<int32_t>(d0);
    c[1] = static_cast<int32_t>(d1);
    c[2] = static_cast<int32_t>(d2);
    c[3] = static_cast<int32_t>(d3);
}

__global__ void debug_imma_m16n8k32_s8s8_kernel(const int8_t *__restrict__ a,
                                                const int8_t *__restrict__ b,
                                                int32_t *__restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int thread_id = lane & 3;
    const int group_id = lane >> 2;

    uint32_t a_frag[4];
    #pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        int8_t vals[4];
        #pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const int e = reg * 4 + byte;
            const int row = group_id + (((e >= 4 && e < 8) || e >= 12) ? 8 : 0);
            const int col = thread_id * 4 + (e & 3) + (e >= 8 ? 16 : 0);
            vals[byte] = a[row * 32 + col];
        }
        a_frag[reg] = pack_s8x4(vals[0], vals[1], vals[2], vals[3]);
    }

    uint32_t b_frag[2];
    #pragma unroll
    for (int reg = 0; reg < 2; ++reg) {
        int8_t vals[4];
        #pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const int e = reg * 4 + byte;
            const int row = thread_id * 4 + (e & 3) + (e >= 4 ? 16 : 0);
            const int col = group_id;
            vals[byte] = b[col * 32 + row];
        }
        b_frag[reg] = pack_s8x4(vals[0], vals[1], vals[2], vals[3]);
    }

    int32_t c[4] = {0, 0, 0, 0};
    mma_m16n8k32_s8s8(c, a_frag, b_frag);

    #pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int row = group_id + (reg >= 2 ? 8 : 0);
        const int col = thread_id * 2 + (reg & 1);
        out[row * 8 + col] = c[reg];
    }
}

}  // namespace

void run_debug_imma_m16n8k32_s8s8(const int8_t *a, const int8_t *b,
                                  int32_t *out, cudaStream_t stream) {
    debug_imma_m16n8k32_s8s8_kernel<<<1, 32, 0, stream>>>(a, b, out);
}
