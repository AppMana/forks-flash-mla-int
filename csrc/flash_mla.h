#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_mla_params {
    using index_t = int64_t;

    int b, seqlen_q, d, d_v;
    int h, h_h_k_ratio, ngroups;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int *__restrict__ cu_seqlens_k;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // Fused int8 path (task #61): per-kv-token rowwise-symmetric scales for the int8
    // K/V cache. Q is quantized in-kernel (no param). nullptr on the bf16 path.
    float *__restrict__ k_scale_ptr;   // [num_blocks * page_block_size] (per kv token, K dequant)
    float *__restrict__ v_scale_ptr;   // [num_blocks * page_block_size] (per kv token, V dequant)
    index_t kv_scale_batch_stride;     // stride between paged blocks, in elements

    int *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    void *__restrict__ softmax_lseaccum_ptr;
    void *__restrict__ oaccum_ptr;
};

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool is_sm90>
struct mha_fwd_splitkv_mla {
    static void run(Flash_fwd_mla_params &params, cudaStream_t stream);
};

template<typename T, int Headdim>
struct mha_fwd_splitkv_mla_ws {
    static void run(Flash_fwd_mla_params &params, cudaStream_t stream);
};

// Plain (cutlass-type-free) dispatch entry so the host-compiled binding
// (flash_api.cpp, c++ not nvcc) never needs cutlass element types. Defined in
// flash_api_dispatch.cu (nvcc).
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream,
                             int head_size, bool is_bf16, bool is_sm90, bool warp_spec);

// Ampere sm_86 sparse-MLA decode (DeepSeek-V4-Flash absorbed form, head_dim 512, V==K).
struct Sparse_mla_decode_params {
    int num_tokens, num_heads, block_size;
    float scale_log2;                 // softmax_scale * log2(e)
    const void *q_ptr;                // bf16 [T, H, 512]
    int64_t q_token_stride, q_head_stride;   // in elements
    void *o_ptr;                      // bf16 [T, H, 512]
    int64_t out_token_stride, out_head_stride;
    const float *attn_sink_ptr;       // [H] or nullptr
    // swa stream
    const void *swa_cache_ptr;        // uint8 fp8_ds_mla [nb, bs, 584]
    int64_t swa_block_stride;         // bytes per paged block
    const int *swa_indices_ptr;       // [T, swa_topk]
    const int *swa_lens_ptr;          // [T]
    int swa_topk, swa_num_blocks;
    // optional extra (compressed) stream; *_cache_ptr nullptr when absent
    const void *extra_cache_ptr;
    int64_t extra_block_stride;
    const int *extra_indices_ptr;
    const int *extra_lens_ptr;
    int extra_topk, extra_num_blocks, extra_block_size;
    // split-KV (flash-decoding): partial buffers + split count, filled by the binding.
    void *oaccum_ptr;    // bf16 [T, H, num_splits, 512] un-normalized acc per split (halves combine traffic)
    float *mlse_ptr;     // [T, H, num_splits, 2]   {running max m, denom l} per split
    int num_splits;
};

void run_sparse_mla_decode(Sparse_mla_decode_params &params, cudaStream_t stream);

struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);
