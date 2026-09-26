"""
Linear layers over row-sliced / column-permuted weights, with cheap backwards.

The cheap frozen-head path computes per-head-group projections by slicing rows of
the Q/K/V weights and permuting columns of c_proj's weight. Autograd's default
backward for `weight[index]` is an atomic index_add (it must assume duplicate
indices) which profiles at >100us per call. Our indices are always UNIQUE
(head slices / a full permutation), so the weight gradient can be scattered with
index_copy_ instead — a plain permutation copy, no atomics.

Both ops keep the canonical weight as the autograd leaf: optimizer state,
checkpoints and the eager oracle never see a permuted weight.
"""
import torch
import torch.nn.functional as F


def _autocast_args(x, weight, bias):
    dev = "cuda" if x.is_cuda else "cpu"
    if torch.is_autocast_enabled(dev):
        dt = torch.get_autocast_dtype(dev)
        x = x.to(dt)
        weight = weight.to(dt)
        bias = None if bias is None else bias.to(dt)
    return x, weight, bias


class _RowSlicedLinear(torch.autograd.Function):
    """F.linear(x, weight[rows], bias[rows]) with a no-atomics weight backward.
    Rows not selected get exactly zero weight/bias gradient."""

    @staticmethod
    def forward(ctx, x, weight, bias, rows):
        w = weight.index_select(0, rows)
        b = None if bias is None else bias.index_select(0, rows)
        ctx.save_for_backward(x, weight, rows)
        ctx.has_bias = bias is not None
        return F.linear(x, w, b)

    @staticmethod
    def backward(ctx, dout):
        x, weight, rows = ctx.saved_tensors
        dout2 = dout.reshape(-1, dout.size(-1))
        x2 = x.reshape(-1, x.size(-1))
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            w = weight.index_select(0, rows)
            dx = (dout2 @ w).view(x.shape)
        if ctx.needs_input_grad[1]:
            dw = torch.zeros_like(weight)
            dw.index_copy_(0, rows, dout2.t() @ x2)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = torch.zeros(weight.size(0), device=dout.device, dtype=dout.dtype)
            db.index_copy_(0, rows, dout2.sum(0))
        return dx, dw, db, None


class _ColPermutedLinear(torch.autograd.Function):
    """F.linear(y, weight[:, cols], bias) where cols is a FULL permutation of the
    input features; weight grad is scattered back with index_copy_ (no atomics)."""

    @staticmethod
    def forward(ctx, y, weight, bias, cols):
        w = weight.index_select(1, cols)
        ctx.save_for_backward(y, weight, cols)
        ctx.has_bias = bias is not None
        return F.linear(y, w, bias)

    @staticmethod
    def backward(ctx, dout):
        y, weight, cols = ctx.saved_tensors
        dout2 = dout.reshape(-1, dout.size(-1))
        y2 = y.reshape(-1, y.size(-1))
        dy = dw = db = None
        if ctx.needs_input_grad[0]:
            w = weight.index_select(1, cols)
            dy = (dout2 @ w).view(y.shape)
        if ctx.needs_input_grad[1]:
            dw = torch.empty_like(weight)          # full permutation: every col written
            dw.index_copy_(1, cols, dout2.t() @ y2)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = dout2.sum(0)
        return dy, dw, db, None


class _TwoWeightRowSlicedLinear(torch.autograd.Function):
    """One GEMM computing [F.linear(x, w1[rows]), F.linear(x, w2[rows])] stacked on
    the feature dim (used to fuse the Q and K projections of the unfrozen heads).
    Same no-atomics index_copy_ weight backward as _RowSlicedLinear."""

    @staticmethod
    def forward(ctx, x, w1, w2, b1, b2, rows):
        w = torch.cat([w1.index_select(0, rows), w2.index_select(0, rows)])
        b = None if b1 is None else \
            torch.cat([b1.index_select(0, rows), b2.index_select(0, rows)])
        ctx.save_for_backward(x, w1, w2, rows)
        ctx.has_bias = b1 is not None
        return F.linear(x, w, b)   # (..., 2R)

    @staticmethod
    def backward(ctx, dout):
        x, w1, w2, rows = ctx.saved_tensors
        dout2 = dout.reshape(-1, dout.size(-1))
        x2 = x.reshape(-1, x.size(-1))
        R = dout.size(-1) // 2
        dx = dw1 = dw2 = db1 = db2 = None
        if ctx.needs_input_grad[0]:
            w = torch.cat([w1.index_select(0, rows), w2.index_select(0, rows)])
            dx = (dout2 @ w).view(x.shape)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            dw = dout2.t() @ x2                     # (2R, in) in one GEMM
            dw1 = torch.zeros_like(w1)
            dw1.index_copy_(0, rows, dw[:R])
            dw2 = torch.zeros_like(w2)
            dw2.index_copy_(0, rows, dw[R:])
        if ctx.has_bias and ctx.needs_input_grad[3]:
            db = dout2.sum(0)
            db1 = torch.zeros(w1.size(0), device=dout.device, dtype=dout.dtype)
            db1.index_copy_(0, rows, db[:R])
            db2 = torch.zeros(w2.size(0), device=dout.device, dtype=dout.dtype)
            db2.index_copy_(0, rows, db[R:])
        return dx, dw1, dw2, db1, db2, None


def linear_rows(x, weight, bias, rows):
    """Linear over a unique-row slice of `weight` (per-head-group projection)."""
    x, weight, bias = _autocast_args(x, weight, bias)
    return _RowSlicedLinear.apply(x, weight, bias, rows)


def linear_rows2(x, w1, w2, b1, b2, rows):
    """Fused two-weight row-sliced linear: returns (..., 2R) = [x@w1[rows]^T, x@w2[rows]^T]."""
    x, w1, b1 = _autocast_args(x, w1, b1)
    _, w2, b2 = _autocast_args(x, w2, b2)
    return _TwoWeightRowSlicedLinear.apply(x, w1, w2, b1, b2, rows)


def linear_permuted_cols(y, weight, bias, cols):
    """Linear over a column-permuted `weight` (head-order fix-up inside c_proj)."""
    y, weight, bias = _autocast_args(y, weight, bias)
    return _ColPermutedLinear.apply(y, weight, bias, cols)
