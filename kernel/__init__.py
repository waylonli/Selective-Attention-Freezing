"""Custom Triton kernels for the dynamic frozen-head attention project.

Import surface (used by nanogpt/model.py):

    from kernel import frozen_pv, frozen_pv_usable

`frozen_pv(patt, v, fz_idx, T)` computes, per frozen head h in fz_idx,
`y[b, f] = patt[h, :T, :T] @ v[b, h]` with a causal-tiled Triton GEMM
(autograd-aware: backward produces dV = P^T @ dY, zero grad elsewhere).

`frozen_pv_usable(v)` says whether the Triton path can run for tensor v
(CUDA + triton importable); callers fall back to dense torch.matmul otherwise.
"""
from .frozen_attn import frozen_pv, frozen_pv_usable, HAVE_TRITON
from .indexed_linear import linear_rows, linear_rows2, linear_permuted_cols
from .fused_attn import (fused_mixed_attn, fused_attn_usable, decomposed_logz,
                         decomp_logits, make_rho_band, BAND_ROWS)

__all__ = ["frozen_pv", "frozen_pv_usable", "HAVE_TRITON",
           "linear_rows", "linear_rows2", "linear_permuted_cols",
           "fused_mixed_attn", "fused_attn_usable", "decomposed_logz",
           "decomp_logits", "make_rho_band", "BAND_ROWS"]
