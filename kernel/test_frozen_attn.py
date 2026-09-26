"""
Standalone unit test for the frozen-head Triton kernel (kernel/frozen_attn.py).

Checks forward and backward against the dense torch.matmul reference over a grid of
shapes (unaligned T, head_dim != power of two, pattern buffer larger than T, single
frozen head, all heads frozen) in fp32 (tight tolerance, IEEE dots) and bf16.

Run:  python kernel/test_frozen_attn.py   (needs a CUDA GPU; exits 0 on PASS)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel import frozen_pv, frozen_pv_usable  # noqa: E402


def make_pattern(nh, S, device, dtype=torch.float32, seed=0):
    """Causal, row-stochastic pattern buffer, like a real prior."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.rand(nh, S, S, generator=g).to(device=device, dtype=dtype)
    p = p.tril()
    return p / p.sum(-1, keepdim=True).clamp(min=1e-9)


def ref_fwd(patt, vf, fz, T):
    return torch.matmul(patt[fz, :T, :T].to(vf.dtype).unsqueeze(0), vf)


def one_case(B, nh, T, hd, S, fz_list, dtype, device, tol_fwd, tol_bwd):
    patt = make_pattern(nh, S, device, seed=nh * S + T)
    fz = torch.tensor(fz_list, dtype=torch.long, device=device)
    fz32 = fz.to(torch.int32)

    g = torch.Generator(device="cpu").manual_seed(T * hd + B)
    base = torch.randn(B, len(fz_list), T, hd, generator=g).to(device=device, dtype=dtype)
    v_tri = base.clone().requires_grad_(True)
    v_ref = base.clone().requires_grad_(True)

    y_tri = frozen_pv(patt, v_tri, fz32, T)
    y_ref = ref_fwd(patt, v_ref, fz, T)
    d_fwd = (y_tri.float() - y_ref.float()).abs().max().item()

    w = torch.randn_like(y_ref, dtype=torch.float32)
    (y_tri.float() * w).sum().backward()
    (y_ref.float() * w).sum().backward()
    d_bwd = (v_tri.grad.float() - v_ref.grad.float()).abs().max().item()

    ok = d_fwd <= tol_fwd and d_bwd <= tol_bwd
    tag = "ok " if ok else "FAIL"
    print(f"  [{tag}] B={B} nh={nh} T={T:5d} hd={hd:3d} S={S:5d} nf={len(fz_list)} "
          f"{str(dtype):15s} fwd={d_fwd:.2e} bwd={d_bwd:.2e}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("needs a CUDA GPU"); sys.exit(1)
    dummy = torch.zeros(1, device="cuda")
    if not frozen_pv_usable(dummy):
        print("triton not available"); sys.exit(1)
    device = "cuda"

    cases = [
        # B, nh, T,    hd,  S,    frozen heads
        (2, 12, 1024,  64, 1024, [0, 1, 2]),
        (2, 12, 1024,  64, 1024, [3]),
        (2, 12, 1024,  64, 1024, list(range(12))),
        (2, 12,  700,  64, 1024, [0, 5, 11]),          # T unaligned, buffer S > T
        (2, 12,    1,  64, 1024, [0, 5, 11]),          # generation step
        (2, 12,   37,  64, 1024, [0, 5, 11]),
        (1, 16, 2048,  96, 2048, [0, 7, 8, 15]),       # hd not a power of two
        (3,  8,  256, 128, 512,  [1, 6]),
        (8, 12, 1024,  64, 1024, [0, 1, 2]),           # bench-like batch
    ]
    all_ok = True
    print("fp32 (IEEE dots; contract tolerance ~1e-5):")
    for c in cases:
        all_ok &= one_case(*c, torch.float32, device, tol_fwd=1e-5, tol_bwd=1e-4)
    print("bf16 (tensor cores; kernel-noise tolerance):")
    for c in cases:
        all_ok &= one_case(*c, torch.bfloat16, device, tol_fwd=5e-2, tol_bwd=5e-2)

    print("\n=>", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
