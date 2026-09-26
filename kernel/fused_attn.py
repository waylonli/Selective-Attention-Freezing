"""
Fused mixed-head attention: frozen and unfrozen heads in ONE kernel launch.

`frozen_attn.py` computes only the frozen heads' `P @ V`, leaving the unfrozen
heads to SDPA. That split costs a `cat` of the two head groups, extra transposes,
and a permutation of c_proj's weight columns — profiling at 124M/T=1024/25%
showed ~2.1 ms/iter of such glue against only ~1.3 ms of real saving.

Here a single kernel walks ALL heads and branches per program on the head state
(the branch is uniform within a program, so there is no warp divergence):

    state 0  unfrozen         ->  y = softmax(QK^T / sqrt(d)) V, causal (flash)
    state 1  frozen-dense     ->  y_i = sum_{j<=i} P[h, i, j] V_j   (no Q/K read)
    state 2  frozen-decomposed->  y_i = sum_{j<=i} P_i,j V_j  with the pattern
             rebuilt IN REGISTERS from two per-head vectors:
                 P[i, j] = exp2((alpha[j] + rho[i-j] - z[i]) * log2(e))
             where z[i] = logsumexp_{j<=i}(alpha[j] + rho[i-j]) is precomputed
             once at freeze time (`decomposed_logz`). Storage per head is O(T)
             instead of O(T^2) and the frozen path reads ZERO pattern bandwidth.

State 2 only exists when the caller passes alpha/rho/z; otherwise the constexpr
`HAS_DECOMP` compiles the decomposed branch OUT and the binaries are the same
as before this feature existed — dense-mode efficiency is untouched.

Heads keep their NATURAL order, so the output needs no concatenation and c_proj
runs on its own unpermuted weight. Q/K are only ever read for unfrozen heads and
their gradients are only ever written there, so frozen heads keep exactly zero
Q/K gradient (the tensors are pre-zeroed and those programs never touch them).

Backward follows the standard FlashAttention-2 two-kernel split:
  * `_bwd_dkdv`: one program per key block; unfrozen -> dK, dV; frozen -> dV only
    (dV[j] = sum_{i>=j} P[h,i,j] dO[i], with P read (dense) or rebuilt in
    registers (decomposed)). alpha/rho/z are buffers and receive no gradient.
  * `_bwd_dq`:   one program per query block, unfrozen heads only.

Precision: fp32 inputs use IEEE dots (the ~1e-5 eager-oracle contract);
bf16/fp16 use tensor cores. Softmax statistics are always fp32.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

RCP_LN2 = 1.4426950408889634   # 1/ln(2), for exp2-based softmax

# Rows of the pre-expanded rho band == the kernel's BLOCK_M row tile. A
# Toeplitz tile of rho[i-j] shifts its base address by 1 per row, so it can
# never be 16-byte aligned and its loads serialize (measured ~2x the whole
# branch cost at T=16K). The band stores BAND_ROWS shifted copies,
# band[r, u] = rho[r - u + (S - BAND_ROWS)], turning every tile access into a
# standard aligned 2-D block load (all address terms are multiples of the
# block size). Storage is BAND_ROWS * S per head — 1/256 of the dense T^2
# pattern at BLOCK 64 with T=16K.
BAND_ROWS = 64


def fused_attn_usable(q):
    return HAVE_TRITON and q.is_cuda


def make_rho_band(rho):
    """(H, S) rho vectors -> (H, BAND_ROWS, S) aligned band for the kernel.
    band[h, r, u] = rho[h, r - u + (S - BAND_ROWS)], and -inf outside [0, S).
    The out-of-range condition is EXACTLY the non-causal condition (i - j < 0),
    so the -inf fill makes exp2 zero those cells for free — the kernel needs no
    causal mask or where() in the decomposed branch at all."""
    H, S = rho.shape
    assert S % BAND_ROWS == 0, \
        f"decomposed prior buffer length {S} must be a multiple of {BAND_ROWS}"
    r = torch.arange(BAND_ROWS, device=rho.device)[:, None]
    u = torch.arange(S, device=rho.device)[None, :]
    src = r - u + (S - BAND_ROWS)
    band = rho[:, src.clamp(0, S - 1)]
    return torch.where((src >= 0) & (src < S), band,
                       torch.full((), float("-inf"), device=rho.device,
                                  dtype=rho.dtype)).contiguous()


def decomp_logits(alpha, rho, rows=None):
    """Masked decomposed logits: L[h, i, j] = alpha[h, j] + rho[h, i-j], and
    -inf where j > i. alpha, rho: (H, S); rows=(r0, r1) selects a row slice
    (default all rows). Returns (H, r1-r0, S).

    This is THE definition of the decomposed pattern in logit space — the
    normalizers (decomposed_logz), the materialized patterns (model /
    controller), and the Triton kernel's in-register rebuild all follow it;
    change it here and nowhere else."""
    H, S = alpha.shape
    r0, r1 = (0, S) if rows is None else rows
    i = torch.arange(r0, r1, device=alpha.device)
    j = torch.arange(S, device=alpha.device)
    dist = i[:, None] - j[None, :]
    lg = alpha[:, None, :] + rho[:, dist.clamp(min=0)]
    return lg.masked_fill(dist < 0, float("-inf"))


@torch.no_grad()
def decomposed_logz(alpha, rho, chunk=None):
    """z[h, i] = logsumexp_{j<=i}(alpha[h, j] + rho[h, i-j]).

    The row normalizers of the decomposed pattern, precomputed once at freeze
    time so the kernel never normalizes online. alpha, rho: (H, S) fp32.
    Row-chunked adaptively so the workspace stays ~0.5 GB regardless of H, S.
    """
    H, S = alpha.shape
    if chunk is None:
        chunk = max(1, int(2 ** 27 // max(H * S, 1)))
    z = torch.empty_like(alpha)
    for i0 in range(0, S, chunk):
        i1 = min(i0 + chunk, S)
        z[:, i0:i1] = decomp_logits(alpha, rho, rows=(i0, i1)).logsumexp(-1)
    return z


if HAVE_TRITON:

    _RCP_LN2 = tl.constexpr(1.4426950408889634)

    # ---- per-head-state forward accumulators (always inlined by triton) ----

    @triton.jit
    def _acc_dense(P, v_base, hp, sph, spi, spj, svn, svd,
                   offs_m, offs_d, mask_m, mask_d, hi, T, acc,
                   IEEE: tl.constexpr, BLOCK_N: tl.constexpr):
        # h*sph can exceed int32 at long context (sph = S^2): force 64-bit
        p_base = P + hp.to(tl.int64) * sph
        for j0 in range(0, hi, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T
            v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
                        mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            p = tl.load(p_base + offs_m[:, None] * spi + offs_n[None, :] * spj,
                        mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            p = p.to(v.dtype)
            if IEEE:
                acc = tl.dot(p, v, acc, input_precision="ieee")
            else:
                acc = tl.dot(p, v, acc)
        return acc

    @triton.jit
    def _acc_decomp(A, RB, Zp, v_base, h, sah, sas, srbh, srbr, srbu, szh, szs,
                    SB, svn, svd, i0, offs_m, offs_d, mask_m, mask_d, hi, T, acc,
                    IEEE: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr):
        a_base = A + h * sah
        # RB is the pre-expanded rho band (see make_rho_band):
        #   rho[i-j] = RB[i - i0, (SB - BLOCK_M) - i0 + j]
        # i.e. every tile is a standard ALIGNED 2-D block load (all address
        # terms are multiples of the tile size), where a direct Toeplitz load
        # shifts by one element per row and can never vectorize. The index
        # range is in-bounds by construction, so the load needs no mask.
        rb_row = RB + h * srbh + tl.arange(0, BLOCK_M)[:, None] * srbr
        # softmax statistics always in fp32 (buffers may live in bf16 models).
        # Masked (i >= T) rows get z = +inf so their whole logit row is -inf ->
        # exp2 gives exact zeros: no garbage can overflow into inf*0 = NaN.
        z = tl.load(Zp + h * szh + offs_m * szs, mask=mask_m,
                    other=float("inf")).to(tl.float32)
        for j0 in range(0, hi, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T
            al = tl.load(a_base + offs_n * sas, mask=mask_n, other=0.0).to(tl.float32)
            # band cells outside the causal region are -inf -> exp2 gives exact
            # zeros: no causal mask, no where(). (Rows >= T produce garbage that
            # the output store / zero dO rows already discard.)
            rho = tl.load(rb_row + ((SB - BLOCK_M) - i0 + offs_n)[None, :] * srbu
                          ).to(tl.float32)
            p = tl.math.exp2((al[None, :] + rho - z[:, None]) * _RCP_LN2)
            v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
                        mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            p = p.to(v.dtype)
            if IEEE:
                acc = tl.dot(p, v, acc, input_precision="ieee")
            else:
                acc = tl.dot(p, v, acc)
        return acc

    @triton.jit
    def _acc_flash(Q, K, HIDX, LSE, b, h,
                   sqb, sqh, sqm, sqd, skb, skh, skn, skd,
                   v_base, svn, svd, slb, slh, slm, sm_scale,
                   offs_m, offs_d, mask_m, mask_d, hi, T, acc,
                   IEEE: tl.constexpr, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr):
        hq = tl.load(HIDX + h).to(tl.int64)   # compact slot in Q/K
        q_base = Q + b * sqb + hq * sqh
        k_base = K + b * skb + hq * skh
        q = tl.load(q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        qk_scale = sm_scale * _RCP_LN2
        for j0 in range(0, hi, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T
            k = tl.load(k_base + offs_n[None, :] * skn + offs_d[:, None] * skd,
                        mask=mask_n[None, :] & mask_d[:, None], other=0.0)
            if IEEE:
                qk = tl.dot(q, k, input_precision="ieee") * qk_scale
            else:
                qk = tl.dot(q, k) * qk_scale
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
            p = tl.math.exp2(qk - m_safe[:, None])
            alpha = tl.math.exp2(tl.where(m_i == -float("inf"), -float("inf"),
                                          m_i - m_safe))
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
                        mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            if IEEE:
                acc = tl.dot(p, v.to(tl.float32), acc, input_precision="ieee")
            else:
                acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_ij
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]
        m_store = tl.where(m_i == -float("inf"), 0.0, m_i)
        tl.store(LSE + b * slb + h * slh + offs_m * slm,
                 m_store + tl.math.log2(l_safe), mask=mask_m)
        return acc

    @triton.jit
    def _fwd_kernel(Q, K, V, P, A, RB, Zp, FROZEN, HIDX, PIDX, Out, LSE,
                    sqb, sqh, sqm, sqd,
                    skb, skh, skn, skd,
                    svb, svh, svn, svd,
                    sph, spi, spj,
                    sah, sas, srbh, srbr, srbu, szh, szs, SB,
                    sob, soh, som, sod,
                    slb, slh, slm,
                    sm_scale, H, T, HD,
                    IEEE: tl.constexpr, HAS_DECOMP: tl.constexpr,
                    P_IDENTITY: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_D: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < T
        mask_d = offs_d < HD
        v_base = V + b * svb + h * svh
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
        is_frozen = tl.load(FROZEN + h)
        if P_IDENTITY:
            hp = h
        else:
            hp = tl.load(PIDX + h).to(tl.int64)

        if HAS_DECOMP:
            if is_frozen == 2:
                acc = _acc_decomp(A, RB, Zp, v_base, h, sah, sas,
                                  srbh, srbr, srbu, szh, szs, SB, svn, svd,
                                  pid_m * BLOCK_M, offs_m, offs_d,
                                  mask_m, mask_d, hi, T, acc,
                                  IEEE, BLOCK_M, BLOCK_N)
            elif is_frozen == 1:
                acc = _acc_dense(P, v_base, hp, sph, spi, spj, svn, svd,
                                 offs_m, offs_d, mask_m, mask_d, hi, T, acc,
                                 IEEE, BLOCK_N)
            else:
                acc = _acc_flash(Q, K, HIDX, LSE, b, h,
                                 sqb, sqh, sqm, sqd, skb, skh, skn, skd,
                                 v_base, svn, svd, slb, slh, slm, sm_scale,
                                 offs_m, offs_d, mask_m, mask_d, hi, T, acc,
                                 IEEE, BLOCK_M, BLOCK_N)
        else:
            if is_frozen:
                acc = _acc_dense(P, v_base, hp, sph, spi, spj, svn, svd,
                                 offs_m, offs_d, mask_m, mask_d, hi, T, acc,
                                 IEEE, BLOCK_N)
            else:
                acc = _acc_flash(Q, K, HIDX, LSE, b, h,
                                 sqb, sqh, sqm, sqd, skb, skh, skn, skd,
                                 v_base, svn, svd, slb, slh, slm, sm_scale,
                                 offs_m, offs_d, mask_m, mask_d, hi, T, acc,
                                 IEEE, BLOCK_M, BLOCK_N)

        tl.store(Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
                 acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

    @triton.jit
    def _bwd_preprocess(O, DO, Delta,
                        sob, soh, som, sod,
                        sdb, sdh, sdm, sdd,
                        slb, slh, slm,
                        H, T, HD,
                        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_m[:, None] < T) & (offs_d[None, :] < HD)
        o = tl.load(O + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
                    mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdm + offs_d[None, :] * sdd,
                     mask=mask, other=0.0).to(tl.float32)
        tl.store(Delta + b * slb + h * slh + offs_m * slm, tl.sum(o * do, 1),
                 mask=offs_m < T)

    # ---- per-head-state backward accumulators for dV (and dK, unfrozen) ----

    @triton.jit
    def _dv_dense(P, do_base, hp, sph, spi, spj, sdom, sdod,
                  offs_n, offs_d, mask_n, mask_d, lo, T, dv,
                  IEEE: tl.constexpr, BLOCK_M: tl.constexpr):
        # dV[j] = sum_{i >= j} P[h, i, j] dO[i];  no dK, no dQ
        p_base = P + hp.to(tl.int64) * sph     # 64-bit: sph = S^2 overflows int32
        for i0 in range(lo, T, BLOCK_M):
            offs_m = i0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < T
            do = tl.load(do_base + offs_m[:, None] * sdom + offs_d[None, :] * sdod,
                         mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            p = tl.load(p_base + offs_m[:, None] * spi + offs_n[None, :] * spj,
                        mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            pt = tl.trans(p).to(do.dtype)
            if IEEE:
                dv = tl.dot(pt, do, dv, input_precision="ieee")
            else:
                dv = tl.dot(pt, do, dv)
        return dv

    @triton.jit
    def _dv_decomp(A, RB, Zp, do_base, h, sah, sas, srbh, srbr, srbu, szh, szs,
                   SB, sdom, sdod, offs_n, offs_d, mask_n, mask_d, lo, T, dv,
                   IEEE: tl.constexpr, BLOCK_M: tl.constexpr):
        # same in-register pattern rebuild as the forward, via the aligned rho
        # band (lo == this program's j0); alpha/rho/z receive no grads
        al = tl.load(A + h * sah + offs_n * sas, mask=mask_n, other=0.0).to(tl.float32)
        rb_row = RB + h * srbh + tl.arange(0, BLOCK_M)[:, None] * srbr
        z_base = Zp + h * szh
        for i0 in range(lo, T, BLOCK_M):
            offs_m = i0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < T
            z = tl.load(z_base + offs_m * szs, mask=mask_m,
                        other=float("inf")).to(tl.float32)
            # -inf band fill zeroes non-causal cells through exp2, and the
            # z = +inf fill zeroes rows i >= T exactly (no mask, no where)
            rho = tl.load(rb_row + ((SB - BLOCK_M - i0) + offs_n)[None, :] * srbu) \
                .to(tl.float32)
            p = tl.math.exp2((al[None, :] + rho - z[:, None]) * _RCP_LN2)
            do = tl.load(do_base + offs_m[:, None] * sdom + offs_d[None, :] * sdod,
                         mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            pt = tl.trans(p).to(do.dtype)
            if IEEE:
                dv = tl.dot(pt, do, dv, input_precision="ieee")
            else:
                dv = tl.dot(pt, do, dv)
        return dv

    @triton.jit
    def _dkdv_flash(Q, K, HIDX, DK, do_base, LSE, Delta, b, h,
                    sqb, sqh, sqm, sqd, skb, skh, skn, skd,
                    v_base, svn, svd, sdom, sdod,
                    sdkb, sdkh, sdkn, sdkd, slb, slh, slm, sm_scale,
                    offs_n, offs_d, mask_n, mask_d, lo, T, dv,
                    IEEE: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
        hq = tl.load(HIDX + h).to(tl.int64)
        q_base = Q + b * sqb + hq * sqh
        k_base = K + b * skb + hq * skh
        k = tl.load(k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        qk_scale = sm_scale * _RCP_LN2
        for i0 in range(lo, T, BLOCK_M):
            offs_m = i0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < T
            q = tl.load(q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                        mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            lse = tl.load(LSE + b * slb + h * slh + offs_m * slm,
                          mask=mask_m, other=0.0)
            if IEEE:
                qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
            else:
                qk = tl.dot(q, tl.trans(k)) * qk_scale
            p = tl.math.exp2(qk - lse[:, None])
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
            do = tl.load(do_base + offs_m[:, None] * sdom + offs_d[None, :] * sdod,
                         mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            if IEEE:
                dv = tl.dot(tl.trans(p), do.to(tl.float32), dv,
                            input_precision="ieee")
                dp = tl.dot(do.to(tl.float32), tl.trans(v).to(tl.float32),
                            input_precision="ieee")
            else:
                dv = tl.dot(tl.trans(p).to(do.dtype), do, dv)
                dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            delta = tl.load(Delta + b * slb + h * slh + offs_m * slm,
                            mask=mask_m, other=0.0)
            ds = p * (dp - delta[:, None]) * sm_scale
            if IEEE:
                dk = tl.dot(tl.trans(ds), q.to(tl.float32), dk,
                            input_precision="ieee")
            else:
                dk = tl.dot(tl.trans(ds).to(q.dtype), q, dk)
        tl.store(DK + b * sdkb + hq * sdkh + offs_n[:, None] * sdkn + offs_d[None, :] * sdkd,
                 dk.to(DK.dtype.element_ty),
                 mask=mask_n[:, None] & mask_d[None, :])
        return dv

    @triton.jit
    def _bwd_dkdv(Q, K, V, P, A, RB, Zp, FROZEN, HIDX, PIDX,
                  DO, DK, DV, LSE, Delta,
                  sqb, sqh, sqm, sqd,
                  skb, skh, skn, skd,
                  svb, svh, svn, svd,
                  sph, spi, spj,
                  sah, sas, srbh, srbr, srbu, szh, szs, SB,
                  sdob, sdoh, sdom, sdod,
                  sdkb, sdkh, sdkn, sdkd,
                  sdvb, sdvh, sdvn, sdvd,
                  slb, slh, slm,
                  sm_scale, H, T, HD,
                  IEEE: tl.constexpr, HAS_DECOMP: tl.constexpr,
                  P_IDENTITY: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_D: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < T
        mask_d = offs_d < HD
        do_base = DO + b * sdob + h * sdoh
        v_base = V + b * svb + h * svh
        dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        is_frozen = tl.load(FROZEN + h)
        lo = pid_n * BLOCK_N
        if P_IDENTITY:
            hp = h
        else:
            hp = tl.load(PIDX + h).to(tl.int64)

        if HAS_DECOMP:
            if is_frozen == 2:
                dv = _dv_decomp(A, RB, Zp, do_base, h, sah, sas,
                                srbh, srbr, srbu, szh, szs, SB, sdom, sdod,
                                offs_n, offs_d, mask_n, mask_d, lo, T, dv,
                                IEEE, BLOCK_M)
            elif is_frozen == 1:
                dv = _dv_dense(P, do_base, hp, sph, spi, spj, sdom, sdod,
                               offs_n, offs_d, mask_n, mask_d, lo, T, dv,
                               IEEE, BLOCK_M)
            else:
                dv = _dkdv_flash(Q, K, HIDX, DK, do_base, LSE, Delta, b, h,
                                 sqb, sqh, sqm, sqd, skb, skh, skn, skd,
                                 v_base, svn, svd, sdom, sdod,
                                 sdkb, sdkh, sdkn, sdkd, slb, slh, slm,
                                 sm_scale, offs_n, offs_d, mask_n, mask_d,
                                 lo, T, dv, IEEE, BLOCK_M, BLOCK_N, BLOCK_D)
        else:
            if is_frozen:
                dv = _dv_dense(P, do_base, hp, sph, spi, spj, sdom, sdod,
                               offs_n, offs_d, mask_n, mask_d, lo, T, dv,
                               IEEE, BLOCK_M)
            else:
                dv = _dkdv_flash(Q, K, HIDX, DK, do_base, LSE, Delta, b, h,
                                 sqb, sqh, sqm, sqd, skb, skh, skn, skd,
                                 v_base, svn, svd, sdom, sdod,
                                 sdkb, sdkh, sdkn, sdkd, slb, slh, slm,
                                 sm_scale, offs_n, offs_d, mask_n, mask_d,
                                 lo, T, dv, IEEE, BLOCK_M, BLOCK_N, BLOCK_D)

        tl.store(DV + b * sdvb + h * sdvh + offs_n[:, None] * sdvn + offs_d[None, :] * sdvd,
                 dv.to(DV.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])

    @triton.jit
    def _bwd_dq(Q, K, V, FROZEN, HIDX, DO, DQ, LSE, Delta,
                sqb, sqh, sqm, sqd,
                skb, skh, skn, skd,
                svb, svh, svn, svd,
                sdob, sdoh, sdom, sdod,
                sdqb, sdqh, sdqm, sdqd,
                slb, slh, slm,
                sm_scale, H, T, HD,
                IEEE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_D: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        if tl.load(FROZEN + h):
            return                      # frozen heads (dense or decomposed): dQ stays zero
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < T
        mask_d = offs_d < HD
        hq = tl.load(HIDX + h).to(tl.int64)
        q_base = Q + b * sqb + hq * sqh
        k_base = K + b * skb + hq * skh
        v_base = V + b * svb + h * svh
        q = tl.load(q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do = tl.load(DO + b * sdob + h * sdoh + offs_m[:, None] * sdom + offs_d[None, :] * sdod,
                     mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        lse = tl.load(LSE + b * slb + h * slh + offs_m * slm, mask=mask_m, other=0.0)
        delta = tl.load(Delta + b * slb + h * slh + offs_m * slm, mask=mask_m, other=0.0)
        dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        qk_scale = sm_scale * _RCP_LN2
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
        for j0 in range(0, hi, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T
            k = tl.load(k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
                        mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
                        mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            if IEEE:
                qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
                dp = tl.dot(do.to(tl.float32), tl.trans(v).to(tl.float32),
                            input_precision="ieee")
            else:
                qk = tl.dot(q, tl.trans(k)) * qk_scale
                dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            p = tl.math.exp2(qk - lse[:, None])
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
            ds = p * (dp - delta[:, None]) * sm_scale
            if IEEE:
                dq = tl.dot(ds, k.to(tl.float32), dq, input_precision="ieee")
            else:
                dq = tl.dot(ds.to(k.dtype), k, dq)
        tl.store(DQ + b * sdqb + hq * sdqh + offs_m[:, None] * sdqm + offs_d[None, :] * sdqd,
                 dq.to(DQ.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

    _BM, _BN = 64, 64

    def _decomp_args(v, patt, alpha, rho_band, z):
        """Kernel-arg plumbing for the two frozen representations.

        Either buffer may be absent (dense-only or decomposed-only layers);
        absent ones are replaced by a dead pointer + zero strides. The
        HAS_DECOMP constexpr compiles the decomposed branch out entirely when
        alpha is None, so dense-mode binaries do not change.
        """
        if patt is None:
            pt, sp = v, (0, 0, 0)
        else:
            pt, sp = patt, tuple(patt.stride())
        if alpha is None:
            a = r = zz = v
            sa = sz = (0, 0)
            sr = (0, 0, 0)
            sb = 0
        else:
            sb = alpha.size(1)
            # the band's in-bounds-without-mask addressing needs both of these
            assert sb % _BM == 0 and sb >= v.size(2), \
                f"decomposed buffer length {sb} must be a multiple of {_BM} and >= T"
            assert rho_band.shape == (alpha.size(0), _BM, sb), \
                "rho_band must be make_rho_band(rho) with matching block size"
            a, r, zz = alpha, rho_band, z
            sa, sr, sz = tuple(alpha.stride()), tuple(rho_band.stride()), tuple(z.stride())
        return pt, a, r, zz, sp, sa, sr, sz, sb, alpha is not None

    def _fwd(q, k, v, patt, alpha, rho_band, z, frozen, hidx, pidx,
             sm_scale, p_identity):
        B, H, T, HD = v.shape
        out = torch.empty_like(v)
        # no zero-init: unfrozen heads always write their LSE rows and frozen
        # heads never read them (fwd or bwd)
        lse = torch.empty((B, H, T), device=v.device, dtype=torch.float32)
        BD = max(16, triton.next_power_of_2(HD))
        pt, a, r, zz, sp, sa, sr, sz, sb, has_dec = _decomp_args(v, patt, alpha, rho_band, z)
        _fwd_kernel[(triton.cdiv(T, _BM), B * H)](
            q, k, v, pt, a, r, zz, frozen, hidx, pidx, out, lse,
            *q.stride(), *k.stride(), *v.stride(),
            *sp, *sa, *sr, *sz, sb,
            *out.stride(), *lse.stride(),
            sm_scale, H, T, HD,
            IEEE=(v.dtype == torch.float32), HAS_DECOMP=has_dec,
            P_IDENTITY=p_identity,
            BLOCK_M=_BM, BLOCK_N=_BN, BLOCK_D=BD,
            num_warps=4, num_stages=2)
        return out, lse

    def _bwd(q, k, v, patt, alpha, rho_band, z, frozen, hidx, pidx,
             out, do, lse, sm_scale, p_identity):
        B, H, T, HD = v.shape
        do = do.contiguous()
        dq = torch.zeros_like(q)        # frozen heads keep exactly zero
        dk = torch.zeros_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty((B, H, T), device=v.device, dtype=torch.float32)
        BD = max(16, triton.next_power_of_2(HD))
        pt, a, r, zz, sp, sa, sr, sz, sb, has_dec = _decomp_args(v, patt, alpha, rho_band, z)
        _bwd_preprocess[(triton.cdiv(T, _BM), B * H)](
            out, do, delta, *out.stride(), *do.stride(), *delta.stride(),
            H, T, HD, BLOCK_M=_BM, BLOCK_D=BD, num_warps=4)
        _bwd_dkdv[(triton.cdiv(T, _BN), B * H)](
            q, k, v, pt, a, r, zz, frozen, hidx, pidx,
            do, dk, dv, lse, delta,
            *q.stride(), *k.stride(), *v.stride(),
            *sp, *sa, *sr, *sz, sb,
            *do.stride(), *dk.stride(), *dv.stride(), *delta.stride(),
            sm_scale, H, T, HD,
            IEEE=(v.dtype == torch.float32), HAS_DECOMP=has_dec,
            P_IDENTITY=p_identity,
            BLOCK_M=_BM, BLOCK_N=_BN, BLOCK_D=BD, num_warps=4, num_stages=2)
        _bwd_dq[(triton.cdiv(T, _BM), B * H)](
            q, k, v, frozen, hidx, do, dq, lse, delta,
            *q.stride(), *k.stride(), *v.stride(), *do.stride(), *dq.stride(),
            *delta.stride(),
            sm_scale, H, T, HD,
            IEEE=(v.dtype == torch.float32),
            BLOCK_M=_BM, BLOCK_N=_BN, BLOCK_D=BD, num_warps=4, num_stages=2)
        return dq, dk, dv


class _FusedMixedAttn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, patt, alpha, rho_band, z, frozen, hidx, pidx,
                sm_scale, p_identity):
        out, lse = _fwd(q, k, v, patt, alpha, rho_band, z, frozen, hidx,
                        pidx, sm_scale, p_identity)
        ctx.save_for_backward(q, k, v, patt, alpha, rho_band, z, frozen,
                              hidx, pidx, out, lse)
        ctx.sm_scale = sm_scale
        ctx.p_identity = p_identity
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, patt, alpha, rho_band, z, frozen, hidx, pidx, out, lse = \
            ctx.saved_tensors
        dq, dk, dv = _bwd(q, k, v, patt, alpha, rho_band, z, frozen,
                          hidx, pidx, out, do, lse, ctx.sm_scale,
                          ctx.p_identity)
        # patt / alpha / rho / z are fixed buffers: no gradient, ever
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def fused_mixed_attn(q, k, v, patt, frozen, hidx=None, pidx=None,
                     sm_scale=None, alpha=None, rho=None, z=None,
                     rho_band=None, validate=True):
    """Causal attention over all heads at once, frozen heads using a fixed prior.

    v       : (B, H, T, HD) — every head needs V.
    q, k    : (B, U, T, HD) — ONLY the unfrozen heads (U = H - n_frozen), so the
              caller never computes a projection it will not use. `hidx[h]` gives
              head h's slot in q/k; frozen entries are never read. Pass U == H
              with hidx=None for the identity mapping.
    patt    : (P, S, S) fixed pattern buffer, S >= T (only dense-frozen rows are
              read). With pidx=None, P >= H and pattern row h belongs to head h.
              Otherwise pidx[h] selects a compact pattern row, which is useful
              for adapters that store only the frozen heads. May be None when
              no head is dense-frozen.
    frozen  : (H,) int32 CUDA tensor per-head state:
              0 = live, 1 = frozen on the dense pattern `patt`,
              2 = frozen on the decomposed pattern softmax(alpha[j] + rho[i-j]).
    alpha, rho, z : (H, S) — only when decomposed heads exist (state 2).
              z[i] = logsumexp_{j<=i}(alpha[j] + rho[i-j]) precomputed via
              `decomposed_logz` (pass z explicitly to avoid recomputing).
              When alpha is None, state 2 must not appear in `frozen`.
    rho_band: (H, 64, S) output of `make_rho_band(rho)` — the aligned layout
              the kernel actually reads (a raw Toeplitz tile load cannot
              vectorize). Precompute it once at freeze time; if omitted it is
              derived here per call.
    pidx    : optional (H,) int32 compact pattern-row mapping. It is ignored by
              live and decomposed heads.
    Returns (B, H, T, HD) in natural head order. dQ/dK have q/k's compact shape,
    so frozen heads cannot receive Q/K gradient at all; patt/alpha/rho/z get none.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (v.size(-1) ** 0.5)
    if hidx is None:
        hidx = torch.arange(v.size(1), device=v.device, dtype=torch.int32)
    p_identity = pidx is None
    if p_identity:
        # A valid dead pointer is still required by the Triton launch. The
        # P_IDENTITY constexpr compiles every load from this pointer away.
        pidx = frozen
    if alpha is not None:
        if z is None:
            z = decomposed_logz(alpha, rho)
        if rho_band is None:
            rho_band = make_rho_band(rho)
    if validate:
        assert q.ndim == k.ndim == v.ndim == 4, "q, k, v must be 4-D"
        assert q.shape == k.shape, "q and k must have the same compact shape"
        B, H, T, HD = v.shape
        assert q.shape[0] == B and q.shape[2:] == (T, HD), \
            "q/k batch, sequence, and head dimensions must match v"
        assert frozen.shape == hidx.shape == pidx.shape == (H,), \
            "frozen, hidx, and pidx must have one entry per output head"
        assert frozen.dtype == hidx.dtype == pidx.dtype == torch.int32, \
            "frozen, hidx, and pidx must be int32"
        assert frozen.device == hidx.device == pidx.device == v.device, \
            "head-state and index tensors must be on v.device"
        assert q.device == k.device == v.device, "q, k, v must share a device"
        if patt is not None:
            assert patt.ndim == 3 and patt.shape[1] >= T and patt.shape[2] >= T, \
                "patt must have shape (P, S, S) with S >= T"
            assert patt.device == v.device, "patt must be on v.device"
            if p_identity:
                assert patt.shape[0] >= H, \
                    "identity pattern mapping requires at least H pattern rows"
            elif bool((frozen == 1).any()):
                dense_pidx = pidx[frozen == 1]
                assert int(dense_pidx.min()) >= 0 and \
                    int(dense_pidx.max()) < patt.shape[0], \
                    "pidx for dense-frozen heads must index patt"
        else:
            assert not bool((frozen == 1).any()), \
                "dense-frozen heads require patt"
        if alpha is not None:
            assert rho is not None and alpha.shape == rho.shape == z.shape, \
                "alpha, rho, and z must have matching (H, S) shapes"
            assert alpha.shape[0] == H and alpha.shape[1] >= T, \
                "decomposed buffers must have shape (H, S) with S >= T"
            assert alpha.device == rho.device == z.device == v.device, \
                "decomposed buffers must be on v.device"
        else:
            assert not bool((frozen == 2).any()), \
                "decomposed-frozen heads require alpha/rho/z"
    return _FusedMixedAttn.apply(q, k, v, patt, alpha, rho_band, z, frozen,
                                 hidx, pidx, sm_scale, p_identity)
