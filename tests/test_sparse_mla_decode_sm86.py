"""Red-first oracle + parity test for the Ampere (sm_86) CUDA sparse-MLA decode kernel.

Builds a realistic fp8_ds_mla cache (the exact DSV4 byte layout), random top-k slot
indices, computes the fp32 reference (absorbed MLA: V == K, head_dim 512), and asserts
the CUDA kernel `flash_sparse_mla_decode` matches.

RED until the kernel + binding exist. Run:
  CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_sparse_mla_decode_sm86.py -x -q
"""
import math
import torch
import pytest

# fp8_ds_mla byte layout (must match nvidia_sm86/triton_kernels.py)
_FP8_DIM = 448           # NoPE dims, fp8e4m3 (1 byte each)
_ROPE_DIM = 64           # RoPE dims, bf16 (2 bytes each)
_SCALE_DIM = 8           # per-token scale bytes; SCALE_GROUP=64 -> 7 used + 1 pad
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2     # 576 data bytes/token
_SCALE_GROUP = 64
_HEAD_DIM = 512          # 448 NoPE + 64 RoPE


def _write_fp8_ds_mla_token(k_cache, slot, block_size):
    """Write one token into the paged fp8_ds_mla cache; return its expected bf16 K row [512]."""
    block_idx = slot // block_size
    block_offset = slot % block_size
    values = ((torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17) - 8) / 16.0
    values = values + float(slot) / 32.0
    scale_exponents = torch.tensor([-2, -1, 0, 1, 2, -2, 1], device=k_cache.device, dtype=torch.float32)
    scales = torch.exp2(scale_exponents)
    scale_per_dim = scales.repeat_interleave(_SCALE_GROUP)
    fp8_values = (values / scale_per_dim).to(torch.float8_e4m3fn)
    expected_nope = fp8_values.float() * scale_per_dim
    rope = (torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device) + float(slot) / 16.0).to(torch.bfloat16)

    flat = k_cache[block_idx].view(-1)
    data_start = block_offset * _TOKEN_DATA_SIZE
    scale_start = block_size * _TOKEN_DATA_SIZE + block_offset * _SCALE_DIM
    flat[data_start : data_start + _FP8_DIM] = fp8_values.view(torch.uint8)
    flat[data_start + _FP8_DIM : data_start + _TOKEN_DATA_SIZE] = rope.view(torch.uint8)
    enc = (scale_exponents.to(torch.int32) + 127).to(torch.uint8)
    flat[scale_start : scale_start + enc.numel()] = enc
    flat[scale_start + enc.numel() : scale_start + _SCALE_DIM] = 127
    return torch.cat([expected_nope, rope.float()]).to(torch.bfloat16)


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def _fp32_ref(q, K_by_slot, indices, lens, scale, attn_sink):
    """q [T,H,512]; K_by_slot [num_slots,512]; indices [T,topk]; lens [T]; -> [T,H,512]."""
    T, H, D = q.shape
    out = torch.zeros(T, H, D, device=q.device, dtype=torch.float32)
    for t in range(T):
        n = int(lens[t].item())
        if n == 0:
            continue
        sl = indices[t, :n].long()
        K = K_by_slot[sl].float()                       # [n, 512]
        scores = (q[t].float() @ K.t()) * scale         # [H, n]
        if attn_sink is not None:
            sink = attn_sink[:, None].float()
            m = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
            ex = torch.exp(scores - m)
            denom = ex.sum(-1, keepdim=True) + torch.exp(sink - m)
            p = ex / denom
        else:
            p = torch.softmax(scores, dim=-1)
        out[t] = p @ K
    return out


@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("topk", [256, 512])
def test_sparse_mla_decode_parity(H, topk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("sm_86 sparse-MLA decode kernel requires Ampere (capability 8.x)")
    from flash_mla import flash_sparse_mla_decode

    torch.manual_seed(0)
    dev = "cuda"
    T = 2
    block_size = 32
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    num_slots = topk + 64
    num_blocks = (num_slots + block_size - 1) // block_size

    swa_cache = torch.zeros(num_blocks, block_size, _TOKEN_DATA_SIZE + _SCALE_DIM,
                            dtype=torch.uint8, device=dev)
    K_by_slot = torch.zeros(num_blocks * block_size, _HEAD_DIM, dtype=torch.bfloat16, device=dev)
    for slot in range(num_slots):
        K_by_slot[slot] = _write_fp8_ds_mla_token(swa_cache, slot, block_size)

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    lens = torch.tensor([topk, topk - 7], dtype=torch.int32, device=dev)[:T]
    indices = torch.stack([torch.randperm(num_slots, device=dev)[:topk].to(torch.int32) for _ in range(T)])
    attn_sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _fp32_ref(q, K_by_slot, indices, lens, scale, attn_sink)
    out = flash_sparse_mla_decode(
        q=q, swa_cache=swa_cache, swa_indices=indices, swa_lens=lens,
        scale=scale, attn_sink=attn_sink,
    )
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"sparse-MLA decode cos_diff={cd:.2e} (H={H} topk={topk})"


def _build_cache(num_slots, block_size, dev):
    nb = (num_slots + block_size - 1) // block_size
    cache = torch.zeros(nb, block_size, _TOKEN_DATA_SIZE + _SCALE_DIM, dtype=torch.uint8, device=dev)
    K = torch.zeros(nb * block_size, _HEAD_DIM, dtype=torch.bfloat16, device=dev)
    for slot in range(num_slots):
        K[slot] = _write_fp8_ds_mla_token(cache, slot, block_size)
    return cache, K


@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("s_q", [1, 2])  # MTP: >1 decode token per sequence
def test_sparse_mla_decode_two_stream(H, s_q):
    """swa + extra streams (different block sizes), attn_sink, multi-token (MTP)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere (capability 8.x)")
    from flash_mla import flash_sparse_mla_decode

    torch.manual_seed(1)
    dev = "cuda"
    T = 3 * s_q
    swa_topk, extra_topk = 384, 256
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    # swa block_size 32, extra (compressed) block_size 16 — exercises distinct strides
    swa_cache, swa_K = _build_cache(swa_topk + 48, 32, dev)
    extra_cache, extra_K = _build_cache(extra_topk + 48, 16, dev)
    swa_slots, extra_slots = swa_K.shape[0], extra_K.shape[0]

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    swa_lens = torch.randint(swa_topk // 2, swa_topk + 1, (T,), dtype=torch.int32, device=dev)
    extra_lens = torch.randint(extra_topk // 2, extra_topk + 1, (T,), dtype=torch.int32, device=dev)
    swa_idx = torch.stack([torch.randperm(swa_slots, device=dev)[:swa_topk].to(torch.int32) for _ in range(T)])
    extra_idx = torch.stack([torch.randperm(extra_slots, device=dev)[:extra_topk].to(torch.int32) for _ in range(T)])
    attn_sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    # fp32 ref: attend over concatenated [swa selected ; extra selected]
    O_ref = torch.zeros(T, H, _HEAD_DIM, device=dev, dtype=torch.float32)
    for t in range(T):
        ns, ne = int(swa_lens[t]), int(extra_lens[t])
        K = torch.cat([swa_K[swa_idx[t, :ns].long()], extra_K[extra_idx[t, :ne].long()]]).float()
        sc = (q[t].float() @ K.t()) * scale
        sink = attn_sink[:, None].float()
        mx = torch.maximum(sc.max(-1, keepdim=True).values, sink)
        ex = torch.exp(sc - mx)
        O_ref[t] = (ex @ K) / (ex.sum(-1, keepdim=True) + torch.exp(sink - mx))

    out = flash_sparse_mla_decode(
        q=q, swa_cache=swa_cache, swa_indices=swa_idx, swa_lens=swa_lens,
        scale=scale, attn_sink=attn_sink,
        extra_cache=extra_cache, extra_indices=extra_idx, extra_lens=extra_lens,
    )
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"two-stream sparse-MLA cos_diff={cd:.2e} (H={H} s_q={s_q})"


@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("T", [256, 1024])
def test_sparse_mla_prefill_parity(H, T):
    """Prefill: many query tokens, each with its own selection. Exercises the num_splits==1
    fast path (direct in-kernel normalize + sink, no split-KV partials/combine)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 8:
        pytest.skip("requires Ampere (capability 8.x)")
    from flash_mla import flash_sparse_mla_prefill

    torch.manual_seed(3)
    dev = "cuda"
    topk = 512
    block_size = 32
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    num_slots = topk + 128
    swa_cache, K_by_slot = _build_cache(num_slots, block_size, dev)

    q = torch.randn(T, H, _HEAD_DIM, device=dev, dtype=torch.bfloat16)
    # variable per-query lens (incl. some short rows) -> exercises masking at large T
    lens = torch.randint(topk - 64, topk + 1, (T,), dtype=torch.int32, device=dev)
    lens[0] = 1
    lens[1] = topk
    indices = torch.stack([torch.randperm(num_slots, device=dev)[:topk].to(torch.int32) for _ in range(T)])
    attn_sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.1

    O_ref = _fp32_ref(q, K_by_slot, indices, lens, scale, attn_sink)
    out = flash_sparse_mla_prefill(
        q=q, swa_cache=swa_cache, swa_indices=indices, swa_lens=lens,
        scale=scale, attn_sink=attn_sink,
    )
    cd = cos_diff(out.float(), O_ref)
    assert cd < 8e-5, f"sparse-MLA prefill cos_diff={cd:.2e} (H={H} T={T})"
