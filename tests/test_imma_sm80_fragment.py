"""RED-first tests for the standalone SM80 s8 IMMA fragment.

This deliberately does not touch the sparse decode kernel. The prefill work
needs a proven m16n8k32 int8 tensor-core primitive before it is composed with
softmax, sparse indices, or an int8 KV-cache layout.
"""

import pytest
import torch

import flash_mla  # noqa: F401  # loads flash_mla_cuda and registers torch.ops.flash_mla


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_debug_imma_m16n8k32_s8s8_matches_torch(seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(seed)
    a = torch.randint(-17, 18, (16, 32), device="cuda", dtype=torch.int8)
    b = torch.randint(-17, 18, (8, 32), device="cuda", dtype=torch.int8)

    got = torch.ops.flash_mla.debug_imma_m16n8k32_s8s8(a, b)
    ref = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).t()

    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0)
