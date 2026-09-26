"""
Unit test for the fused mixed-head kernel (kernel/fused_attn.py).

Reference = the eager oracle's semantics: unfrozen heads use causal
softmax(QK^T/sqrt(d)) V; frozen heads use pattern @ V, where the pattern is
either the dense (H, S, S) buffer or — for decomposed heads — the causal
softmax(alpha[j] + rho[i-j]) MATERIALIZED here in torch and compared against
the kernel's in-register reconstruction. Checks fwd, dQ, dK, dV, that frozen
heads receive EXACTLY zero dQ/dK, and that alpha/rho/z receive no grad.

Run:  python kernel/test_fused_attn.py   (needs a CUDA GPU)
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.fused_attn import (fused_mixed_attn, fused_attn_usable,  # noqa: E402
                               decomposed_logz)


def make_pattern(nh, S, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.rand(nh, S, S, generator=g).to(device=device, dtype=torch.float32).tril()
    return p / p.sum(-1, keepdim=True).clamp(min=1e-9)


def make_decomp(nh, S, device, seed=0, scale=0.8):
    """Random per-head alpha/rho vectors + their precomputed normalizers.
    scale >> 1 gives near-deterministic (sharp) patterns with huge logit
    magnitudes — stresses the garbage-row/-inf zeroing paths of the kernel."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    alpha = (torch.randn(nh, S, generator=g) * scale).to(device=device, dtype=torch.float32)
    rho = (torch.randn(nh, S, generator=g) * scale).to(device=device, dtype=torch.float32)
    return alpha, rho, decomposed_logz(alpha, rho)


def decomp_materialize(alpha, rho, T, z=None):
    """(H, T, T) causal softmax(alpha[j] + rho[i-j]) — the torch reference.
    With z given, uses exp(alpha + rho - z) exactly like the kernel does (so a
    rounded/bf16 z stays bit-consistent between kernel and reference)."""
    ar = torch.arange(T, device=alpha.device)
    dist = ar[:, None] - ar[None, :]
    lg = alpha[:, None, :T] + rho[:, dist.clamp(min=0)]
    if z is None:
        return lg.masked_fill(dist < 0, float("-inf")).softmax(-1)
    lg = lg - z[:, :T, None]
    return lg.masked_fill(dist < 0, float("-inf")).exp()


def ref(q, k, v, patt, fz_list, T):
    """Eager oracle: per-head softmax attention, frozen heads overridden."""
    B, H, _, hd = v.shape
    causal = torch.tril(torch.ones(T, T, device=v.device, dtype=torch.bool))
    att = (q @ k.transpose(-2, -1)) * (1.0 / (hd ** 0.5))
    att = att.masked_fill(~causal, float("-inf")).softmax(-1)
    fp = patt[:, :T, :T].to(att.dtype).unsqueeze(0)
    m = torch.zeros(H, dtype=torch.bool, device=v.device)
    m[fz_list] = True
    att = torch.where(m.view(1, H, 1, 1), fp.expand_as(att), att)
    return att @ v


def one_case(B, H, T, hd, S, fz_list, dtype, device, tol_f, tol_b,
             strided=False, dec_list=(), no_patt=False, buf_bf16=False,
             dec_scale=0.8, compact_patt=False):
    """fz_list: dense-frozen heads (state 1); dec_list: decomposed (state 2).
    buf_bf16 stores alpha/rho/z in bf16 (a fully-bf16 model casts its buffers;
    the kernel must upcast to fp32 for the exp2)."""
    dec_list = list(dec_list)
    patt = make_pattern(H, S, device, seed=H * S + T)
    alpha, rho, z = make_decomp(H, S, device, seed=H + S + T, scale=dec_scale)
    if buf_bf16:
        alpha = alpha.to(torch.bfloat16)
        rho = rho.to(torch.bfloat16)
        z = z.to(torch.bfloat16)
    frozen = torch.zeros(H, dtype=torch.int32, device=device)
    frozen[fz_list] = 1
    frozen[dec_list] = 2
    all_fz = sorted(set(fz_list) | set(dec_list))
    uf = [h for h in range(H) if h not in all_fz]
    # hidx[h] = head h's slot in the COMPACT q/k (frozen entries never read)
    hidx = torch.zeros(H, dtype=torch.int32, device=device)
    for slot, h in enumerate(uf):
        hidx[h] = slot

    # reference pattern buffer: decomposed heads use their materialized softmax
    ref_patt = patt.clone()
    if dec_list:
        ref_patt[dec_list, :T, :T] = decomp_materialize(
            alpha[dec_list].float(), rho[dec_list].float(), T,
            z=z[dec_list].float())
    has_dec = len(dec_list) > 0
    pidx = None
    if compact_patt:
        # Store only dense-frozen patterns, as the Qwen adapter does. This also
        # exercises compact indexing inside the HAS_DECOMP three-state branch.
        pidx = torch.zeros(H, dtype=torch.int32, device=device)
        for slot, h in enumerate(fz_list):
            pidx[h] = slot
        kern_patt = None if no_patt else patt[fz_list].contiguous()
    else:
        kern_patt = None if no_patt else patt
    kern_a = alpha if has_dec else None
    kern_r = rho if has_dec else None
    kern_z = z if has_dec else None
    if has_dec:
        alpha.requires_grad_(True)   # must come back with .grad == None
        rho.requires_grad_(True)

    g = torch.Generator(device="cpu").manual_seed(T * hd + B)
    base = [torch.randn(B, H, T, hd, generator=g).to(device=device, dtype=dtype)
            for _ in range(3)]
    # kernel gets compact q/k (only unfrozen heads) + full v
    if strided:
        # exactly what the model hands us: q/k are non-dense views into one fused
        # QK projection, so their grads are allocated CONTIGUOUS (different strides)
        qk = torch.stack([base[0][:, uf], base[1][:, uf]], dim=2) \
            .transpose(1, 2).contiguous().transpose(1, 2)   # (B, U, 2, T, hd)
        qk = qk.detach().requires_grad_(True)
        tri_q, tri_k = qk[:, :, 0], qk[:, :, 1]
        leaf_q = leaf_k = qk
    else:
        tri_q = base[0][:, uf].clone().requires_grad_(True)
        tri_k = base[1][:, uf].clone().requires_grad_(True)
        leaf_q, leaf_k = tri_q, tri_k
    tri_v = base[2].clone().requires_grad_(True)
    rf = [t.clone().requires_grad_(True) for t in base]

    y_t = fused_mixed_attn(tri_q, tri_k, tri_v, kern_patt, frozen, hidx, pidx,
                           alpha=kern_a, rho=kern_r, z=kern_z)
    y_r = ref(rf[0], rf[1], rf[2], ref_patt, all_fz, T)
    d_f = (y_t.float() - y_r.float()).abs().max().item()

    w = torch.randn_like(y_r, dtype=torch.float32)
    (y_t.float() * w).sum().backward()
    (y_r.float() * w).sum().backward()
    dv = (tri_v.grad.float() - rf[2].grad.float()).abs().max().item()
    if uf:
        gq = leaf_q.grad[:, :, 0] if strided else leaf_q.grad
        gk = leaf_k.grad[:, :, 1] if strided else leaf_k.grad
        dq = (gq.float() - rf[0].grad[:, uf].float()).abs().max().item()
        dk = (gk.float() - rf[1].grad[:, uf].float()).abs().max().item()
    else:
        dq = dk = 0.0
    # frozen heads have no q/k slot at all -> they cannot receive Q/K gradient;
    # alpha/rho are fixed buffers -> autograd must hand them no grad at all
    no_ar_grad = (not has_dec) or (alpha.grad is None and rho.grad is None)
    ok = (d_f <= tol_f and dq <= tol_b and dk <= tol_b and dv <= tol_b
          and no_ar_grad)
    tag = (f"nf={len(fz_list):2d} nd={len(dec_list):2d}" +
           (" compactP" if compact_patt else ""))
    print(f"  [{'ok ' if ok else 'FAIL'}] B{B} H{H} T{T:5d} d{hd:3d} {tag} "
          f"{'strided' if strided else 'contig ':7s} {str(dtype):15s} "
          f"fwd={d_f:.1e} dq={dq:.1e} dk={dk:.1e} dv={dv:.1e}"
          f"{'' if no_ar_grad else '  ALPHA/RHO GOT GRAD'}")
    return ok


def main():
    if not torch.cuda.is_available() or not fused_attn_usable(torch.zeros(1, device="cuda")):
        print("needs a CUDA GPU with triton"); sys.exit(1)
    dev = "cuda"
    cases = [
        (2, 12, 1024, 64, 1024, [0, 1, 2]),
        (2, 12, 1024, 64, 1024, []),            # nothing frozen: pure flash
        (2, 12, 1024, 64, 1024, list(range(12))),  # everything frozen
        (2, 12, 1024, 64, 1024, [3, 7, 11]),    # non-contiguous freeze set
        (2, 12,  700, 64, 1024, [0, 5, 11]),    # unaligned T, buffer S > T
        (2, 12,   37, 64, 1024, [0, 5]),
        (1, 16, 2048, 96, 2048, [0, 7, 8, 15]),  # hd not a power of two
        (4, 12, 2048, 64, 2048, [0, 1, 2]),
    ]
    # dense-frozen (state 1) + decomposed-frozen (state 2) + live, mixed
    dec_cases = [
        # (B, H, T, hd, S, fz_list, dec_list, no_patt)
        (2, 12, 1024, 64, 1024, [0, 1], [2, 3], False),      # both kinds + live
        (2, 12, 1024, 64, 1024, [], [0, 5, 11], True),       # decomposed only, patt=None
        (2, 12, 1024, 64, 1024, [], list(range(12)), True),  # all heads decomposed
        (2, 12,  700, 64, 1024, [3], [0, 7], False),         # unaligned T, S > T
        (2, 12,   37, 64, 1024, [5], [0, 2], False),         # tiny T
        (1, 16, 2048, 96, 2048, [0, 8], [7, 15], False),     # hd not power of two
        (4, 12, 2048, 64, 2048, [1], [0, 2], False),
    ]
    ok = True
    print("fp32 (IEEE dots; eager-oracle contract ~1e-5):")
    for c in cases:
        ok &= one_case(*c, torch.float32, dev, 2e-5, 2e-4)
    print("bf16 (tensor cores):")
    for c in cases:
        ok &= one_case(*c, torch.bfloat16, dev, 6e-2, 2e-1)
    # q/k as NON-DENSE views of one fused QK projection (what the model passes):
    # their grads get contiguous strides, so DQ/DK must not reuse Q/K's strides.
    print("non-dense q/k views (grad strides differ from input strides):")
    for c in cases:
        ok &= one_case(*c, torch.float32, dev, 2e-5, 2e-4, strided=True)
        ok &= one_case(*c, torch.bfloat16, dev, 6e-2, 2e-1, strided=True)
    print("decomposed heads (in-register softmax(alpha[j]+rho[i-j]) vs materialized):")
    for (B, H, T, hd, S, fz, dec, np_) in dec_cases:
        ok &= one_case(B, H, T, hd, S, fz, torch.float32, dev, 2e-5, 2e-4,
                       dec_list=dec, no_patt=np_)
        ok &= one_case(B, H, T, hd, S, fz, torch.bfloat16, dev, 6e-2, 2e-1,
                       dec_list=dec, no_patt=np_)
        ok &= one_case(B, H, T, hd, S, fz, torch.float32, dev, 2e-5, 2e-4,
                       strided=True, dec_list=dec, no_patt=np_)
    # fully-bf16 model: alpha/rho/z buffers themselves are bf16 (kernel upcasts)
    print("decomposed heads with bf16 alpha/rho/z buffers:")
    for (B, H, T, hd, S, fz, dec, np_) in dec_cases[:3]:
        ok &= one_case(B, H, T, hd, S, fz, torch.bfloat16, dev, 6e-2, 2e-1,
                       dec_list=dec, no_patt=np_, buf_bf16=True)
    # EXTREME logits (near-deterministic patterns, |logit| up to ~150) on
    # unaligned T: fwd/bwd garbage rows must stay exactly zeroed (no inf/NaN)
    print("decomposed heads with extreme sharp logits (overflow guard):")
    for (B, H, T, hd, S, fz, dec, np_) in [
            (2, 12,  700, 64, 1024, [3], [0, 7], False),
            (2, 12,   37, 64, 1024, [], [0, 2, 5], True),
            (2, 12, 1000, 64, 1024, [], list(range(12)), True)]:
        ok &= one_case(B, H, T, hd, S, fz, torch.float32, dev, 2e-5, 2e-4,
                       dec_list=dec, no_patt=np_, dec_scale=40.0)
        ok &= one_case(B, H, T, hd, S, fz, torch.bfloat16, dev, 6e-2, 2e-1,
                       dec_list=dec, no_patt=np_, dec_scale=40.0)
    print("decomposed + compact dense-pattern mapping (Qwen compatibility):")
    for dtype, tf, tb in ((torch.float32, 2e-5, 2e-4),
                          (torch.bfloat16, 6e-2, 2e-1)):
        ok &= one_case(2, 12, 700, 64, 1024, [3, 9], dtype, dev, tf, tb,
                       strided=True, dec_list=[0, 7], compact_patt=True)
    print("\n=>", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
