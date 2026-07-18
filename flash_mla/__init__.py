__version__ = "2.0.0"

from .flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_int8,
    sparse_mla_decode_fp8,
    sparse_mla_prefill,
)
from .int8_sparse_mla import (
    quantize_int8_ds_mla_rows,
    sparse_mla_decode_int8,
    sparse_mla_decode_int8_triton,
    sparse_mla_prefill_int8,
)
