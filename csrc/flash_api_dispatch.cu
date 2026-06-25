// nvcc-compiled dispatch: names the cutlass element types so the host-compiled
// binding (flash_api.cpp) stays cutlass-free (it is built by c++, not nvcc, and
// host toolchains can't always resolve cutlass's libcudacxx includes).
#include <cutlass/numeric_types.h>

#include "flash_mla.h"

void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream,
                             int head_size, bool is_bf16, bool is_sm90, bool warp_spec) {
    if (is_bf16) {
        if (is_sm90) {
            mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576, true>::run(params, stream);
        } else if (warp_spec) {
            if (head_size == 512) {
                mha_fwd_splitkv_mla<cutlass::bfloat16_t, 512, false>::run(params, stream);
            } else {
                mha_fwd_splitkv_mla_ws<cutlass::bfloat16_t, 576>::run(params, stream);
            }
        } else {
            if (head_size == 512) {
                mha_fwd_splitkv_mla<cutlass::bfloat16_t, 512, false>::run(params, stream);
            } else {
                mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576, false>::run(params, stream);
            }
        }
    }
#ifndef FLASH_MLA_DISABLE_FP16
    else {
        // fp16 path is sm90-only (validated by the caller).
        mha_fwd_splitkv_mla<cutlass::half_t, 576, true>::run(params, stream);
    }
#endif
}
