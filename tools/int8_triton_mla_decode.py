"""Fused int8 MLA decode (Triton, sm_86) + head-to-head bench vs bf16 FlashMLA CUDA.

This is the deployable int8-matmul path for DSV4 decode on Ampere (task #61):
  - Q is quantized to int8 ROWWISE in-kernel (per query-head, over the 576 head dim).
  - QK^T runs on the int8 IMMA tensor cores: tl.dot(q_i8, k_i8.T, out_dtype=int32),
    then dequantized by q_scale * k_scale (outer product) -- exactly the int8 IMMA
    pattern already used by the DSV4 indexer in nvidia_sm86/triton_kernels.py.
  - K/V cache is int8 (rowwise-symmetric per kv token); V is dequantized int8->bf16
    in-kernel and PV runs in bf16. Fully fused: one kernel, no separate quant/dequant.

Run on a cluster RTX 3090 (or local A5000), both sm_86:
  python tools/int8_triton_mla_decode.py
Compares, at the real DSV4 decode shape (h=64, d=576, dv=512):
  fp32 reference  vs  int8-triton  vs  bf16 FlashMLA CUDA (torch.ops.flash_mla)
reporting cos_diff (correctness) and latency (us) for each.
"""
import math
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _int8_mla_decode_kernel(
    q_ptr, k_ptr, kscale_ptr, vscale_ptr, out_ptr,
    seqlen,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_sb, stride_sn,
    stride_ob, stride_oh, stride_od,
    sm_scale_log2: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    D_PAD: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    hblk = tl.program_id(1)
    heads = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = heads < H
    offs_d = tl.arange(0, D_PAD)        # padded to pow2; D..D_PAD masked to 0 (no dot contrib)
    mask_d = offs_d < D
    offs_dv = tl.arange(0, DV)          # DV=512 is already pow2

    # load q [BLOCK_H, D_PAD] bf16, quantize to int8 rowwise (per head over D)
    q = tl.load(q_ptr + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    q_amax = tl.maximum(tl.max(tl.abs(q), 1), 1e-12)
    q_scale = q_amax / 127.0
    qn = q / q_scale[:, None]
    # round-half-away-from-zero (symmetric); (x+0.5).to(int8) truncates toward zero
    # and mis-rounds negatives, biasing the quant.
    q_i8 = (qn + tl.where(qn >= 0, 0.5, -0.5)).to(tl.int8)

    e_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, DV), dtype=tl.float32)

    for start in range(0, seqlen, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen
        # K tile [BLOCK_N, D_PAD] int8  (h_kv = 1, shared by all heads); pad cols masked to 0
        k_i8 = tl.load(k_ptr + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                       mask=mask_n[:, None] & mask_d[None, :], other=0).to(tl.int8)
        # one scale per kv token: V shares K's int8 storage (v_i8 == k_i8[:, :dv]),
        # so V must dequant with the SAME k_scale that produced those int8 values.
        kscale = tl.load(kscale_ptr + b * stride_sb + offs_n * stride_sn, mask=mask_n, other=0.0)

        # int8 IMMA: [BLOCK_H, D] @ [D, BLOCK_N] -> int32
        qk_i32 = tl.dot(q_i8, tl.trans(k_i8), out_dtype=tl.int32)
        qk = qk_i32.to(tl.float32) * (q_scale[:, None] * kscale[None, :]) * sm_scale_log2
        qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, -float("inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        p = tl.where(mask_h[:, None] & mask_n[None, :], p, 0.0)

        # V tile [BLOCK_N, DV] int8 = K[:, :DV]; dequant int8->bf16, PV in bf16
        v_i8 = tl.load(k_ptr + b * stride_kb + offs_n[:, None] * stride_kn + offs_dv[None, :] * stride_kd,
                       mask=mask_n[:, None], other=0).to(tl.float32)
        v_bf16 = (v_i8 * kscale[:, None]).to(tl.bfloat16)

        acc = acc * re[:, None] + tl.dot(p.to(tl.bfloat16), v_bf16)
        e_sum = e_sum * re + tl.sum(p, 1)
        e_max = n_e_max

    acc = acc / tl.maximum(e_sum, 1e-20)[:, None]
    tl.store(out_ptr + b * stride_ob + heads[:, None] * stride_oh + offs_dv[None, :] * stride_od,
             acc.to(tl.bfloat16), mask=mask_h[:, None])


def int8_mla_decode(q_bf16, k_i8, k_scale, v_scale, dv, sm_scale, BLOCK_H=16, BLOCK_N=32):
    """q_bf16 [B,H,D]; k_i8 [B,S,D] int8; k_scale/v_scale [B,S]; -> out [B,H,DV] bf16.

    Small head/kv tiles + num_stages=1 keep the [BLOCK_H,DV] accumulator and the
    D_PAD-wide K tile inside sm_86's ~100KB shared-memory budget.
    """
    B, H, D = q_bf16.shape
    S = k_i8.shape[1]
    out = torch.empty(B, H, dv, device=q_bf16.device, dtype=torch.bfloat16)
    grid = (B, triton.cdiv(H, BLOCK_H))
    _int8_mla_decode_kernel[grid](
        q_bf16, k_i8, k_scale, v_scale, out, S,
        q_bf16.stride(0), q_bf16.stride(1), q_bf16.stride(2),
        k_i8.stride(0), k_i8.stride(1), k_i8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        sm_scale_log2=sm_scale * LOG2E,
        H=H, D=D, DV=dv, D_PAD=triton.next_power_of_2(D), BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=1,
    )
    return out


# ----------------------------------------------------------------- references + bench

def quant_rowwise_sym_int8(x, dim):
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(dim)


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def fp32_ref(q, k, dv, sm_scale):
    attn = torch.einsum("bhd,bkd->bhk", q.float(), k.float()) * sm_scale
    p = torch.softmax(attn, dim=-1)
    return torch.einsum("bhk,bkv->bhv", p, k[..., :dv].float())


def bench_us(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    print("DEVICE:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    d, dv = 576, 512
    sm_scale = 1.0 / math.sqrt(d)

    have_flashmla = False
    try:
        from flash_mla import get_mla_metadata, flash_mla_with_kvcache
        have_flashmla = True
    except Exception as e:
        print("flash_mla CUDA not available:", e)

    print(f"\n{'shape':<22}{'int8 cos':>12}{'int8 us':>10}{'bf16FM cos':>12}{'bf16FM us':>11}{'int8 vs bf16':>14}")
    for h in [64]:
        for sk in [512, 1024, 4096]:
            B = 1
            q = torch.randn(B, h, d, device=dev, dtype=torch.bfloat16)
            k = torch.randn(B, sk, d, device=dev, dtype=torch.bfloat16)
            k_i8, k_scale = quant_rowwise_sym_int8(k.float(), dim=-1)
            _, v_scale = quant_rowwise_sym_int8(k[..., :dv].float(), dim=-1)
            k_scale = k_scale.contiguous(); v_scale = v_scale.contiguous()

            O_ref = fp32_ref(q, k, dv, sm_scale)
            O_int8 = int8_mla_decode(q, k_i8, k_scale, v_scale, dv, sm_scale)
            c_int8 = cos_diff(O_int8.float(), O_ref)
            t_int8 = bench_us(lambda: int8_mla_decode(q, k_i8, k_scale, v_scale, dv, sm_scale))

            c_fm = float("nan"); t_fm = float("nan"); c_cmp = float("nan")
            if have_flashmla:
                bs = 32
                pad = triton.cdiv(sk, 256) * 256
                cseq = torch.full((B,), sk, dtype=torch.int32, device=dev)
                qf = q.view(B, 1, h, d)
                bt = torch.arange(B * pad // bs, dtype=torch.int32, device=dev).view(B, pad // bs)
                bk = torch.empty(bt.numel(), bs, 1, d, device=dev, dtype=torch.bfloat16)
                bk.view(B, pad, 1, d)[:, :sk, 0, :] = k
                meta, nsp = get_mla_metadata(cseq, 1 * h // 1, 1)
                O_fm = flash_mla_with_kvcache(qf, bk, bt, cseq, dv, meta, nsp, causal=False, warp_spec=False)[0]
                O_fm = O_fm.view(B, h, dv)
                c_fm = cos_diff(O_fm.float(), O_ref)
                t_fm = bench_us(lambda: flash_mla_with_kvcache(qf, bk, bt, cseq, dv, meta, nsp, causal=False, warp_spec=False))
                c_cmp = cos_diff(O_int8.float(), O_fm.float())
            print(f"{'h=%d sk=%d'%(h,sk):<22}{c_int8:>12.2e}{t_int8:>10.1f}{c_fm:>12.2e}{t_fm:>11.1f}{c_cmp:>14.2e}")
    print("\ncos_diff < 8e-5 = correct vs fp32. int8 vs bf16 = output agreement of the two kernels.")
