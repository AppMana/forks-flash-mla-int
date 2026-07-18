__version__ = "1.1.0"

from .flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_int8,
    # Harmonized role-named kernels (appmana config symbols).
    sparse_mla_decode_fp8,
    sparse_mla_prefill,
    # Deprecated aliases.
    flash_sparse_mla_decode,
    flash_sparse_mla_prefill,
)
from .int8_sparse_mla import (
    quantize_int8_ds_mla_rows,
    # Harmonized role-named kernels.
    sparse_mla_decode_int8,
    sparse_mla_decode_int8_triton,
    sparse_mla_prefill_int8,
    # Deprecated aliases.
    sparse_int8_mla_decode,
    sparse_int8_mla_prefill,
    triton_sparse_int8_mla_decode,
)
