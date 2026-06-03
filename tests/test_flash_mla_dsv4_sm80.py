import random

import pytest
import torch

import tests.test_flash_mla_sm80 as sm80_test
from flash_mla import flash_mla_with_kvcache, get_mla_metadata


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8,
    reason="DSV4 SM80 MLA regression requires an Ampere-or-newer CUDA GPU",
)


@pytest.mark.parametrize(
    ("batch", "seqlen", "heads"),
    [
        (2, 128, 4),
        (4, 256, 8),
    ],
)
def test_dsv4_head_dim_512_sm80(batch, seqlen, heads):
    sm80_test.dtype = torch.bfloat16
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    random.seed(0)

    sm80_test.test_flash_mla(
        batch,
        1,
        seqlen,
        heads,
        1,
        512,
        512,
        True,
        False,
    )


def test_dsv4_head_dim_512_warp_spec_routes_to_sm80_kernel():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(1)

    batch, seqlen, heads, head_dim = 2, 128, 4, 512
    q = torch.randn(batch, 1, heads, head_dim)
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32)
    block_size = 32
    block_table = torch.arange(batch * seqlen // block_size, dtype=torch.int32).view(
        batch, seqlen // block_size
    )
    blocked_k = torch.randn(block_table.numel(), block_size, 1, head_dim)
    metadata, num_splits = get_mla_metadata(cache_seqlens, heads, 1)

    out_fallback, lse_fallback = flash_mla_with_kvcache(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        head_dim,
        metadata,
        num_splits,
        causal=True,
        warp_spec=False,
    )
    out_requested, lse_requested = flash_mla_with_kvcache(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        head_dim,
        metadata,
        num_splits,
        causal=True,
        warp_spec=True,
    )

    torch.testing.assert_close(out_requested, out_fallback)
    torch.testing.assert_close(lse_requested, lse_fallback)
