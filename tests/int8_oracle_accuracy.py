"""De-risk gate for the fused int8 FlashMLA sm_86 kernel (task #61).

Question this answers BEFORE any CUDA is written:
  Does int8-QK (s8 IMMA, rowwise-symmetric Q and K) + int8-KV cache (rowwise V),
  with PV kept in bf16, preserve MLA *decode* accuracy at the real DSV4 head dims
  (d=576, dv=512, h=64)?

It computes three references on identical inputs:
  - O_fp32 : full-precision SDPA (the ground truth, what the bf16 kernel matches).
  - O_int8 : the EXACT math the fused int8 kernel will do
             (q_i8 @ k_i8 -> int32 -> *q_scale*k_scale -> softmax -> p @ dequant(v_i8)).
  - O_bf16ref : bf16 SDPA (so we can see int8 error vs the bf16 baseline error).

If O_int8 tracks O_fp32 about as well as O_bf16ref does, int8 is viable and we build
the kernel. If int8 is materially worse, the scheme changes (per-group K scales, or
int8-KV-only) -- better to learn that now than after writing the kernel.
"""
import math
import torch


def quant_rowwise_sym_int8(x, dim):
    """Symmetric per-row int8: scale = amax/127, q = round(x/scale).clamp(-127,127)."""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def rmse(x, y):
    x, y = x.double(), y.double()
    return ((x - y) ** 2).mean().sqrt().item()


@torch.inference_mode()
def run(b, s_q, seqlen, h_q, d, dv, device):
    # decode shapes: q (b, s_q, h, d); single kv head (MLA), kv (seqlen, d)
    q = torch.randn(b, s_q, h_q, d, device=device, dtype=torch.float32)
    k = torch.randn(b, seqlen, d, device=device, dtype=torch.float32)   # h_kv=1
    v = k[..., :dv]                                                     # MLA: V is a slice of K
    softmax_scale = 1.0 / math.sqrt(d)

    # ---- O_fp32 : ground-truth dense MLA attention (decode, attends to all seqlen) ----
    def sdpa(qq, kk, vv):
        # qq (b, s_q, h, d), kk (b, seqlen, d), vv (b, seqlen, dv)
        attn = torch.einsum("bqhd,bkd->bhqk", qq, kk) * softmax_scale
        p = torch.softmax(attn, dim=-1)
        return torch.einsum("bhqk,bkv->bqhv", p, vv)

    O_fp32 = sdpa(q, k, v)
    O_bf16 = sdpa(q.bfloat16().float(), k.bfloat16().float(), v.bfloat16().float())

    # ---- O_int8 : EXACT fused-int8-kernel math ----
    # Q quantized per (b,s_q,h) row over d ; K/V quantized per kv-token row.
    q_i8, q_scale = quant_rowwise_sym_int8(q, dim=-1)          # scale (b,s_q,h,1)
    k_i8, k_scale = quant_rowwise_sym_int8(k, dim=-1)          # scale (b,seqlen,1)
    v_i8, v_scale = quant_rowwise_sym_int8(v, dim=-1)          # scale (b,seqlen,1)

    # QK^T as int32 IMMA, then dequant by outer product of scales (this is what s8s8s32 yields)
    qk_int = torch.einsum("bqhd,bkd->bhqk", q_i8.float(), k_i8.float())   # == s32 accum
    qsc = q_scale.squeeze(-1).permute(0, 2, 1).unsqueeze(-1)              # (b,h,s_q,1)
    ksc = k_scale.squeeze(-1).unsqueeze(1).unsqueeze(1)                   # (b,1,1,seqlen)
    qk = qk_int * qsc * ksc * softmax_scale
    p = torch.softmax(qk, dim=-1)                                        # bf16 in kernel; fp here
    p_bf16 = p.bfloat16().float()                                        # PV runs in bf16
    v_deq = (v_i8.float() * v_scale).bfloat16().float()                  # V dequant int8->bf16
    O_int8 = torch.einsum("bhqk,bkv->bqhv", p_bf16, v_deq)

    return {
        "int8_vs_fp32_cos": cos_diff(O_int8, O_fp32),
        "bf16_vs_fp32_cos": cos_diff(O_bf16, O_fp32),
        "int8_vs_fp32_rmse": rmse(O_int8, O_fp32),
        "bf16_vs_fp32_rmse": rmse(O_bf16, O_fp32),
        "int8_vs_bf16_cos": cos_diff(O_int8, O_bf16),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    d, dv = 576, 512
    print(f"{'config':<34}{'int8/fp32 cos':>15}{'bf16/fp32 cos':>15}{'int8/fp32 rmse':>16}{'bf16/fp32 rmse':>16}")
    for h_q in [64, 128]:
        for seqlen in [512, 4096]:
            for s_q in [1, 2]:
                r = run(16, s_q, seqlen, h_q, d, dv, device)
                cfg = f"h={h_q} sk={seqlen} s_q={s_q}"
                print(f"{cfg:<34}{r['int8_vs_fp32_cos']:>15.2e}{r['bf16_vs_fp32_cos']:>15.2e}"
                      f"{r['int8_vs_fp32_rmse']:>16.2e}{r['bf16_vs_fp32_rmse']:>16.2e}")
    print("\nViability: int8/fp32 cos within ~1 order of magnitude of bf16/fp32 cos => int8 scheme is sound.")
