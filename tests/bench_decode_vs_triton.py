"""Decode-latency baseline at real DSV4 single-user shape (A5000, sm_86).

Times, at h=64, d=576, dv=512, 512 KV tokens, 1 decode token:
  (1) the LIVE triton sparse-MLA decode (decode_sparse_attention_triton) -- the baseline
      the fused int8 CUDA kernel must beat.
  (2) the bf16 FlashMLA CUDA kernel (flash_mla_with_kvcache) -- the CUDA reference; int8
      should extend its lead by halving K/V memory traffic.

int8 itself is not built yet (task #61) -- this establishes what it is competing with.
"""
import math
import torch
import triton


def bench_us(fn, warmup=25, rep=100):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep) * 1e3  # ms -> us


def bench_flashmla_bf16(h_q, seqlen, device):
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache
    b, s_q, h_kv, d, dv = 1, 1, 1, 576, 512
    block_size = 32
    max_seqlen_pad = triton.cdiv(seqlen, 256) * 256
    cache_seqlens = torch.full((b,), seqlen, dtype=torch.int32, device=device)
    q = torch.randn(b, s_q, h_q, d, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32,
                               device=device).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, device=device, dtype=torch.bfloat16)
    meta, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def run():
        return flash_mla_with_kvcache(q, blocked_k, block_table, cache_seqlens, dv,
                                      meta, num_splits, causal=True, warp_spec=False)
    run()
    return bench_us(run)


def bench_triton_live(h_q, topk, device):
    from vllm.models.deepseek_v4.nvidia_sm86.triton_kernels import decode_sparse_attention_triton
    head_dim = 512
    block_size = 32
    num_blocks = triton.cdiv(topk, block_size) + 2
    scale = 1.0 / math.sqrt(576)  # DSV4 scales by full head_dim incl rope
    q = torch.randn(1, h_q, head_dim, device=device, dtype=torch.bfloat16)
    swa_cache = torch.zeros(num_blocks, block_size, 576, dtype=torch.uint8, device=device)
    # valid token-slot scale bytes so dequant produces finite values (UE8M0 ~ 2^0)
    swa_cache[..., 568:576] = 127
    idx = torch.randint(0, num_blocks * block_size, (1, topk), dtype=torch.int32, device=device)
    swa_lens = torch.full((1,), topk, dtype=torch.int32, device=device)
    out = torch.empty(1, h_q, head_dim, device=device, dtype=torch.bfloat16)

    def run():
        decode_sparse_attention_triton(
            q=q, swa_cache=swa_cache, swa_indices=idx, swa_lens=swa_lens,
            scale=scale, attn_sink=None, out=out,
            extra_cache=None, extra_indices=None, extra_lens=None,
        )
    run()
    return bench_us(run)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    print("DSV4 single-user decode (b=1, s_q=1, d=576, dv=512), A5000 sm_86\n")
    print(f"{'shape':<26}{'triton-live (us)':>18}{'bf16 FlashMLA (us)':>20}{'CUDA speedup':>14}")
    for h_q in [64]:
        for sk in [512, 1024, 4096]:
            try:
                t_tri = bench_triton_live(h_q, sk, dev)
            except Exception as e:
                t_tri = float("nan")
                print(f"  triton failed h={h_q} sk={sk}: {type(e).__name__}: {str(e)[:120]}")
            try:
                t_fm = bench_flashmla_bf16(h_q, sk, dev)
            except Exception as e:
                t_fm = float("nan")
                print(f"  flashmla failed h={h_q} sk={sk}: {type(e).__name__}: {str(e)[:120]}")
            spd = (t_tri / t_fm) if (t_fm == t_fm and t_fm > 0) else float("nan")
            print(f"{'h=%d sk=%d'%(h_q,sk):<26}{t_tri:>18.1f}{t_fm:>20.1f}{spd:>13.2f}x")
    print("\nint8 (task #61, not yet built) targets: <= bf16 FlashMLA latency, "
          "with ~2x less K/V memory traffic -> expected further gain on this memory-bound decode.")
