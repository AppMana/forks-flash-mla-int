// Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Torch + Python STABLE ABI binding (abi3 / cp39, TORCH_STABLE_ONLY).
// No ATen / c10 / pybind: device properties come from the CUDA runtime, the
// CUDA stream from the AOTI shim, tensors via torch::stable. The wheel built
// from this is torch-version-independent for any torch >= 2.9 (stable floor).

#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <cutlass/fast_math.h>

#include "flash_mla.h"
#include "static_switch.h"

using torch::stable::Tensor;
using torch::headeronly::ScalarType;

// ------------------------------------------------------------------ helpers

namespace {

struct DevProps { int major; int minor; int sm_count; };

DevProps get_dev_props(int device) {
    DevProps p{};
    cudaDeviceGetAttribute(&p.major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&p.minor, cudaDevAttrComputeCapabilityMinor, device);
    cudaDeviceGetAttribute(&p.sm_count, cudaDevAttrMultiProcessorCount, device);
    return p;
}

cudaStream_t current_cuda_stream(int device) {
    void *s = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &s));
    return reinterpret_cast<cudaStream_t>(s);
}

bool shape_eq(const Tensor &t, const std::vector<int64_t> &s) {
    if (t.dim() != static_cast<int64_t>(s.size())) return false;
    for (size_t i = 0; i < s.size(); ++i) {
        if (t.size(static_cast<int64_t>(i)) != s[i]) return false;
    }
    return true;
}

}  // namespace

#define CHECK_DEVICE(x) STD_TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) STD_TORCH_CHECK(shape_eq(x, {__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ------------------------------------------------------------------ get_mla_metadata

std::vector<Tensor>
get_mla_metadata(
    Tensor seqlens_k,
    int64_t num_heads_per_head_k,
    int64_t num_heads_k
) {
    // This should match the logic in the MLA kernel.
    static constexpr int block_size_m = 32;
    static constexpr int fixed_overhead_num_blocks = 5;

    int device = torch::stable::accelerator::getCurrentDeviceIndex();
    auto props = get_dev_props(device);
    int block_size_n = props.major >= 9 ? 64 : 32;

    CHECK_DEVICE(seqlens_k);
    STD_TORCH_CHECK(seqlens_k.is_contiguous());
    STD_TORCH_CHECK(seqlens_k.scalar_type() == ScalarType::Int);

    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = reinterpret_cast<int *>(seqlens_k.data_ptr());

    int num_sm_parts = props.sm_count / static_cast<int>(num_heads_k)
                       / cutlass::ceil_div(static_cast<int>(num_heads_per_head_k), block_size_m);

    Tensor tile_scheduler_metadata = torch::stable::new_empty(seqlens_k, {num_sm_parts, TileSchedulerMetaDataSize});
    Tensor num_splits = torch::stable::new_empty(seqlens_k, {batch_size + 1});
    int *tile_scheduler_metadata_ptr = reinterpret_cast<int *>(tile_scheduler_metadata.data_ptr());
    int *num_splits_ptr = reinterpret_cast<int *>(num_splits.data_ptr());

    torch::stable::accelerator::DeviceGuard guard(device);
    cudaStream_t stream = current_cuda_stream(device);
    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    get_mla_metadata_func(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

// ------------------------------------------------------------------ fwd_kvcache_mla

std::vector<Tensor>
mha_fwd_kvcache_mla(
    Tensor q,                               // batch_size x seqlen_q x num_heads x head_size
    Tensor kcache,                          // num_blocks x page_block_size x num_heads_k x head_size
    std::optional<Tensor> vcache_,          // num_blocks x page_block_size x num_heads_k x head_size_v
    int64_t head_size_v,
    Tensor seqlens_k,                       // batch_size
    Tensor block_table,                     // batch_size x max_num_blocks_per_seq
    double softmax_scale,
    bool is_causal,
    Tensor tile_scheduler_metadata,         // num_sm_parts x TileSchedulerMetaDataSize
    Tensor num_splits,                      // batch_size + 1
    bool warp_spec                          // Use warp-specialized SM80 kernel
) {
    int device = torch::stable::accelerator::getCurrentDeviceIndex();
    auto props = get_dev_props(device);
    bool is_sm8x = props.major == 8 && props.minor >= 0;
    bool is_sm90 = props.major == 9 && props.minor == 0;
    STD_TORCH_CHECK(is_sm90 || is_sm8x, "Only sm80 to sm90 (inclusive) are supported");

    Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;

    STD_TORCH_CHECK(kcache.scalar_type() == q.scalar_type(), "query and key must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    STD_TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    STD_TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    STD_TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    CHECK_DEVICE(block_table);
    STD_TORCH_CHECK(block_table.scalar_type() == ScalarType::Int, "block_table must have dtype torch.int32");
    STD_TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    const int batch_size = q.size(0);
    const int seqlen_q_ori = q.size(1);
    const int num_heads_ori = q.size(2);
    const int head_size = q.size(3);
    STD_TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    STD_TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    STD_TORCH_CHECK(batch_size > 0, "batch size must be postive");
    STD_TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    // (b, sq_ori, hk, ng, d) -> transpose(2,3) -> reshape(b, sq, h, d)
    q = torch::stable::reshape(
            torch::stable::transpose(
                torch::stable::view(q, {batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}),
                2, 3),
            {batch_size, seqlen_q, num_heads, head_size});

    int head_size_k = head_size;
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    if (vcache_.has_value()) { CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, (int64_t)head_size_v); }
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);

    STD_TORCH_CHECK(seqlens_k.scalar_type() == ScalarType::Int, "seqlens_k must have dtype int32");
    CHECK_DEVICE(seqlens_k);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(seqlens_k, batch_size);

    torch::stable::accelerator::DeviceGuard guard(device);

    Tensor out = torch::stable::new_empty(q, {batch_size, seqlen_q, num_heads, (int64_t)head_size_v});
    Tensor softmax_lse = torch::stable::new_empty(q, {batch_size, num_heads, seqlen_q}, ScalarType::Float);

    Flash_fwd_mla_params params = {};
    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.cu_seqlens_k = reinterpret_cast<int *>(seqlens_k.data_ptr());
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = static_cast<float>(softmax_scale);
    params.scale_softmax_log2 = static_cast<float>(softmax_scale * M_LOG2E);
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.v_ptr = vcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.v_batch_stride = vcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(-3);
    params.v_row_stride = vcache.stride(-3);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(-2);
    params.v_head_stride = vcache.stride(-2);
    params.o_head_stride = out.stride(-2);

    params.block_table = reinterpret_cast<int *>(block_table.data_ptr());
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;

    STD_TORCH_CHECK(tile_scheduler_metadata.scalar_type() == ScalarType::Int, "tile_scheduler_metadata must have dtype int32");
    STD_TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    params.tile_scheduler_metadata_ptr = reinterpret_cast<int *>(tile_scheduler_metadata.data_ptr());
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    STD_TORCH_CHECK(num_splits.scalar_type() == ScalarType::Int, "num_splits must have dtype int32");
    CHECK_DEVICE(num_splits);
    CHECK_CONTIGUOUS(num_splits);
    params.num_splits_ptr = reinterpret_cast<int *>(num_splits.data_ptr());

    Tensor softmax_lse_accum = torch::stable::new_empty(q, {batch_size + params.num_sm_parts, num_heads, seqlen_q}, ScalarType::Float);
    Tensor out_accum = torch::stable::new_empty(q, {batch_size + params.num_sm_parts, num_heads, seqlen_q, (int64_t)head_size_v}, ScalarType::Float);
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();

    cudaStream_t stream = current_cuda_stream(device);

    STD_TORCH_CHECK(head_size == 576 || head_size == 512);

    if (q.scalar_type() == ScalarType::BFloat16) {
        if (is_sm90) {
            STD_TORCH_CHECK(head_size == 576, "sm90 path currently supports head_size=576 only");
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
    else if (q.scalar_type() == ScalarType::Half) {
        if (is_sm90) {
            mha_fwd_splitkv_mla<cutlass::half_t, 576, true>::run(params, stream);
        } else {
            STD_TORCH_CHECK(false, "sm80 only support bfloat16");
        }
    }
    #endif
    else {
        STD_TORCH_CHECK(false, "Unsupported tensor dtype for query");
    }

    // (b, sq_ori, ng, hk, dv) -> transpose(2,3) -> reshape(b, sq_ori, h_ori, dv)
    out = torch::stable::reshape(
            torch::stable::transpose(
                torch::stable::view(out, {batch_size, seqlen_q_ori, ngroups, num_heads_k, (int64_t)head_size_v}),
                2, 3),
            {batch_size, seqlen_q_ori, num_heads_ori, (int64_t)head_size_v});
    softmax_lse = torch::stable::reshape(
            torch::stable::transpose(
                torch::stable::view(softmax_lse, {batch_size, num_heads_k, seqlen_q_ori, ngroups}),
                2, 3),
            {batch_size, num_heads_ori, seqlen_q_ori});

    return {out, softmax_lse};
}

// ------------------------------------------------------------------ boxed wrappers + registration

void boxed_get_mla_metadata(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto seqlens_k = to<Tensor>(stack[0]);
    auto num_heads_per_head_k = to<int64_t>(stack[1]);
    auto num_heads_k = to<int64_t>(stack[2]);
    auto res = get_mla_metadata(seqlens_k, num_heads_per_head_k, num_heads_k);
    stack[0] = from(res[0]);
    stack[1] = from(res[1]);
}

void boxed_fwd_kvcache_mla(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto q = to<Tensor>(stack[0]);
    auto kcache = to<Tensor>(stack[1]);
    auto vcache = to<std::optional<Tensor>>(stack[2]);
    auto head_size_v = to<int64_t>(stack[3]);
    auto seqlens_k = to<Tensor>(stack[4]);
    auto block_table = to<Tensor>(stack[5]);
    auto softmax_scale = to<double>(stack[6]);
    auto is_causal = to<bool>(stack[7]);
    auto tile_scheduler_metadata = to<Tensor>(stack[8]);
    auto num_splits = to<Tensor>(stack[9]);
    auto warp_spec = to<bool>(stack[10]);
    auto res = mha_fwd_kvcache_mla(q, kcache, vcache, head_size_v, seqlens_k, block_table,
                                   softmax_scale, is_causal, tile_scheduler_metadata, num_splits, warp_spec);
    stack[0] = from(res[0]);
    stack[1] = from(res[1]);
}

STABLE_TORCH_LIBRARY(flash_mla, m) {
    m.def("get_mla_metadata(Tensor seqlens_k, int num_heads_per_head_k, int num_heads_k) -> (Tensor, Tensor)");
    m.def("fwd_kvcache_mla(Tensor q, Tensor kcache, Tensor? vcache, int head_size_v, "
          "Tensor seqlens_k, Tensor block_table, float softmax_scale, bool is_causal, "
          "Tensor tile_scheduler_metadata, Tensor num_splits, bool warp_spec) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flash_mla, CUDA, m) {
    m.impl("get_mla_metadata", &boxed_get_mla_metadata);
    m.impl("fwd_kvcache_mla", &boxed_fwd_kvcache_mla);
}

// Dummy module so `import flash_mla_cuda` loads the .so and runs the
// STABLE_TORCH_LIBRARY static initializers above (registering torch.ops.flash_mla.*).
extern "C" {
PyObject *PyInit_flash_mla_cuda(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "flash_mla_cuda",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}
}
