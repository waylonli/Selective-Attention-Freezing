"""
Triton kernel for frozen-head attention: y = P @ V.

A frozen head replaces softmax(QK^T) with a FIXED causal, row-stochastic pattern P
(its prior). Its output is y = P[:T, :T] @ V, and the only trainable input is V
(dV = P^T @ dY; Q/K are never touched, so their projections get exactly zero grad).

This module replaces the dense `torch.matmul(pattern, v)` in
CausalSelfAttention._attend_cheap, which was slower than plain Flash because of
per-step pattern casts/gathers and a dense (non-causal) GEMM. Here:

- V (and dY/dV in the backward) are COMPACT tensors holding only the frozen heads,
  (B, nf, T, hd), indexed by program id; the pattern buffer keeps its canonical
  (n_head, S, S) layout and is indexed through `fz_idx`. So the model glue never
  gathers or scatters activations for this op.
- P is read DIRECTLY from the registered `dyn_pattern` buffer (any float dtype,
  cast to V's dtype in registers) -> no per-step cast/gather copies, and nothing
  extra is saved for backward (the buffer itself is enough).
- P is causal (zero above the diagonal), so the forward skips j-tiles above the
  i-tile diagonal and the backward skips i-tiles below the j-tile diagonal:
  half the FLOPs and half the pattern bandwidth of a dense GEMM.
- P is shared across the batch; consecutive programs differ only in the batch
  index, so pattern tiles are served from L2 for all but the first program.
- fp32 inputs use IEEE-precision dots (no TF32): the fp32 correctness contract
  vs the eager oracle (`_attend_dynamic`, ~1e-5) holds. bf16/fp16 inputs use
  tensor cores as usual.

Shapes:
    patt   : (n_head, S, S) buffer, S = block_size, T <= S (sliced via strides)
    v      : (B, nf, T, hd) compact frozen-head values (any strides)
    fz_idx : (nf,) int32 CUDA tensor; fz_idx[f] = pattern row of compact head f
    out    : (B, nf, T, hd) compact, in v.dtype
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # CPU-only install; callers use the dense torch fallback
    HAVE_TRITON = False


def frozen_pv_usable(v):
    """True if the Triton path can run for this tensor (CUDA + triton present)."""
    return HAVE_TRITON and v.is_cuda


if HAVE_TRITON:

    @triton.jit
    def _pv_fwd_kernel(P, V, Y, FZ,
                       sph, spi, spj,
                       svb, svf, svt, svd,
                       syb, syf, syt, syd,
                       T, HD,
                       IEEE: tl.constexpr,
                       BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
                       BLOCK_D: tl.constexpr):
        # one program: BLOCK_I query rows x full head_dim, for (batch b, frozen head f)
        pid_b = tl.program_id(0)   # fastest-varying: same P tiles hit L2 across batch
        pid_i = tl.program_id(1)
        pid_f = tl.program_id(2)
        h = tl.load(FZ + pid_f).to(tl.int64)   # pattern row for this compact head
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_d = tl.arange(0, BLOCK_D)
        mask_i = offs_i < T
        mask_d = offs_d < HD
        p_base = P + h * sph
        v_base = V + pid_b * svb + pid_f * svf
        acc = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)
        # causal: P[i, j] = 0 for j > i, so only walk j-tiles up to this i-tile
        hi = tl.minimum((pid_i + 1) * BLOCK_I, T)
        for j0 in range(0, hi, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            mask_j = offs_j < T
            v_tile = tl.load(v_base + offs_j[:, None] * svt + offs_d[None, :] * svd,
                             mask=mask_j[:, None] & mask_d[None, :], other=0.0)
            p_tile = tl.load(p_base + offs_i[:, None] * spi + offs_j[None, :] * spj,
                             mask=mask_i[:, None] & mask_j[None, :], other=0.0)
            p_tile = p_tile.to(v_tile.dtype)
            if IEEE:
                acc = tl.dot(p_tile, v_tile, acc, input_precision="ieee")
            else:
                acc = tl.dot(p_tile, v_tile, acc)
        y_ptr = Y + pid_b * syb + pid_f * syf \
            + offs_i[:, None] * syt + offs_d[None, :] * syd
        tl.store(y_ptr, acc.to(Y.dtype.element_ty),
                 mask=mask_i[:, None] & mask_d[None, :])

    @triton.jit
    def _pv_bwd_kernel(P, DY, DV, FZ,
                       sph, spi, spj,
                       sdyb, sdyf, sdyt, sdyd,
                       sdvb, sdvf, sdvt, sdvd,
                       T, HD,
                       IEEE: tl.constexpr,
                       BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
                       BLOCK_D: tl.constexpr):
        # dV[b, f, j, :] = sum_{i >= j} P[h, i, j] * dY[b, f, i, :]
        pid_b = tl.program_id(0)
        pid_j = tl.program_id(1)
        pid_f = tl.program_id(2)
        h = tl.load(FZ + pid_f).to(tl.int64)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        offs_d = tl.arange(0, BLOCK_D)
        mask_j = offs_j < T
        mask_d = offs_d < HD
        p_base = P + h * sph
        dy_base = DY + pid_b * sdyb + pid_f * sdyf
        acc = tl.zeros((BLOCK_J, BLOCK_D), dtype=tl.float32)
        # causal: P[i, j] = 0 for i < j, so start i-tiles at this j-tile
        for i0 in range(pid_j * BLOCK_J, T, BLOCK_I):
            offs_i = i0 + tl.arange(0, BLOCK_I)
            mask_i = offs_i < T
            dy_tile = tl.load(dy_base + offs_i[:, None] * sdyt + offs_d[None, :] * sdyd,
                              mask=mask_i[:, None] & mask_d[None, :], other=0.0)
            p_tile = tl.load(p_base + offs_i[:, None] * spi + offs_j[None, :] * spj,
                             mask=mask_i[:, None] & mask_j[None, :], other=0.0)
            p_t = tl.trans(p_tile).to(dy_tile.dtype)
            if IEEE:
                acc = tl.dot(p_t, dy_tile, acc, input_precision="ieee")
            else:
                acc = tl.dot(p_t, dy_tile, acc)
        dv_ptr = DV + pid_b * sdvb + pid_f * sdvf \
            + offs_j[:, None] * sdvt + offs_d[None, :] * sdvd
        tl.store(dv_ptr, acc.to(DV.dtype.element_ty),
                 mask=mask_j[:, None] & mask_d[None, :])

    _BLOCK_I = 64
    _BLOCK_J = 64

    def _launch_fwd(patt, v, fz_idx, T):
        B, nf, Tv, hd = v.shape
        yf = torch.empty((B, nf, T, hd), device=v.device, dtype=v.dtype)
        grid = (B, triton.cdiv(T, _BLOCK_I), nf)
        _pv_fwd_kernel[grid](
            patt, v, yf, fz_idx,
            patt.stride(0), patt.stride(1), patt.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            yf.stride(0), yf.stride(1), yf.stride(2), yf.stride(3),
            T, hd,
            IEEE=(v.dtype == torch.float32),
            BLOCK_I=_BLOCK_I, BLOCK_J=_BLOCK_J,
            BLOCK_D=max(16, triton.next_power_of_2(hd)),
            num_warps=4, num_stages=3,
        )
        return yf

    def _launch_bwd(patt, dyf, fz_idx, v_shape, v_dtype, T):
        B, nf, Tv, hd = v_shape
        dv = torch.empty(v_shape, device=dyf.device, dtype=v_dtype)
        grid = (B, triton.cdiv(T, _BLOCK_J), nf)
        _pv_bwd_kernel[grid](
            patt, dyf, dv, fz_idx,
            patt.stride(0), patt.stride(1), patt.stride(2),
            dyf.stride(0), dyf.stride(1), dyf.stride(2), dyf.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            T, hd,
            IEEE=(v_dtype == torch.float32),
            BLOCK_I=_BLOCK_I, BLOCK_J=_BLOCK_J,
            BLOCK_D=max(16, triton.next_power_of_2(hd)),
            num_warps=4, num_stages=3,
        )
        return dv


class _FrozenPV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, patt, v, fz_idx, T):
        ctx.save_for_backward(patt, fz_idx)
        ctx.v_shape = v.shape
        ctx.v_dtype = v.dtype
        ctx.T = T
        return _launch_fwd(patt, v, fz_idx, T)

    @staticmethod
    def backward(ctx, dyf):
        patt, fz_idx = ctx.saved_tensors
        dv = _launch_bwd(patt, dyf, fz_idx, ctx.v_shape, ctx.v_dtype, ctx.T)
        return None, dv, None, None


def frozen_pv(patt, v, fz_idx, T):
    """y[b, f] = patt[fz_idx[f], :T, :T] @ v[b, f]  ->  (B, nf, T, hd).

    patt: (n_head, S, S) fixed pattern buffer (no grad), S >= T.
    v: (B, nf, T, hd) COMPACT frozen-head values, the only differentiable input.
    fz_idx: (nf,) int32 CUDA tensor mapping compact index f -> pattern row.
    """
    return _FrozenPV.apply(patt, v, fz_idx, T)
