"""Parameter gradients for the existing fused absolute-plus-relative operator."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalisers(A, R, Z, T: tl.constexpr, S: tl.constexpr, K: tl.constexpr):
    i, h = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, K)
    a = tl.load(A + h * S + j, j < T, other=-float('inf'))
    r = tl.load(R + h * S + i - j, j <= i, other=-float('inf'))
    logits = a + r
    maximum = tl.max(logits, 0)
    z = maximum + tl.log(tl.sum(tl.exp(logits - maximum), 0))
    tl.store(Z + h * S + i, z)


@triton.jit
def _parameter_backward(V, G, O, A, R, Z, STATE, DA, DR,
                        T: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                        D: tl.constexpr, C: tl.constexpr, M: tl.constexpr,
                        N: tl.constexpr, NT: tl.constexpr):
    tile, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    it, jt = tile // NT, tile % NT
    state = tl.load(STATE + h)
    if state == 2 and jt <= it:
        i = it * M + tl.arange(0, M)
        j = jt * N + tl.arange(0, N)
        d = tl.arange(0, C)
        base = (b * H + h) * T * D
        g = tl.load(G + base + i[:, None] * D + d[None, :],
                    (i[:, None] < T) & (d[None, :] < D), other=0)
        v = tl.load(V + base + j[:, None] * D + d[None, :],
                    (j[:, None] < T) & (d[None, :] < D), other=0)
        o = tl.load(O + base + i[:, None] * D + d[None, :],
                    (i[:, None] < T) & (d[None, :] < D), other=0)
        delta = tl.sum(g.to(tl.float32) * o.to(tl.float32), 1)
        dp = tl.dot(g, tl.trans(v), input_precision='ieee')
        a = tl.load(A + h * S + j, j < T, other=0)
        distance = i[:, None] - j[None, :]
        mask = (i[:, None] < T) & (j[None, :] < T) & (distance >= 0)
        r = tl.load(R + h * S + distance, mask, other=0)
        z = tl.load(Z + h * S + i, i < T, other=0)
        p = tl.where(mask, tl.exp(a[None, :] + r - z[:, None]), 0.)
        ds = p * (dp - delta[:, None])
        tl.atomic_add(DA + h * S + j, tl.sum(ds, 0), j < T, sem='relaxed')
        # Sum diagonals in the tile before atomics; no T-by-T gradient buffer.
        local_delta = tl.arange(0, M + N) - (N - 1)
        column = tl.arange(0, M)[:, None] - local_delta[None, :]
        diagonal = tl.gather(ds, tl.minimum(tl.maximum(column, 0), N - 1), axis=1)
        diagonal = tl.where((column >= 0) & (column < N), diagonal, 0.)
        rho_index = it * M - jt * N + local_delta
        tl.atomic_add(DR + h * S + rho_index, tl.sum(diagonal, 0),
                      (rho_index >= 0) & (rho_index < T), sem='relaxed')


def normalisers(alpha, rho, length):
    z = torch.zeros_like(alpha)
    _normalisers[(length, alpha.shape[0])](
        alpha, rho, z, length, alpha.shape[1], triton.next_power_of_2(length))
    return z


class PatternGradient(torch.autograd.Function):
    """Keep fused dV and add exact softmax parameter derivatives."""

    @staticmethod
    def forward(ctx, out, value, alpha, rho, z, states):
        ctx.save_for_backward(out, value, alpha, rho, z, states)
        return out

    @staticmethod
    def backward(ctx, grad):
        out, value, alpha, rho, z, states = ctx.saved_tensors
        batch, heads, length, dim = value.shape
        da, dr = torch.zeros_like(alpha), torch.zeros_like(rho)
        block = 32
        count = triton.cdiv(length, block)
        _parameter_backward[(count * count, heads, batch)](
            value.contiguous(), grad.contiguous(), out.contiguous(), alpha, rho, z,
            states, da, dr, length, alpha.shape[1], heads, dim,
            triton.next_power_of_2(dim), block, block, count)
        return grad, None, da, dr, None, None
