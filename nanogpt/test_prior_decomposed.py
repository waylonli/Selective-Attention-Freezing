"""
End-to-end integration test for the DECOMPOSED frozen-prior mode
(rand_attn_prior_repr="decomposed": priors stored as alpha[j] + rho[i-j] logit
vectors, O(T) per head, rebuilt in-register by the fused kernel).

Checks, on a fresh dynamic model:
  1. no (n_head, T, T) dyn_pattern buffer is allocated in decomposed mode;
  2. freeze_heads_decomposed + (optionally) dense freeze_heads coexist, and the
     cheap fused path matches the eager capture oracle in fp32 (~1e-3 e2e);
  3. backward: q/k projection rows of frozen heads get EXACTLY zero gradient,
     alpha/rho buffers get no gradient, live-head rows get real gradient;
  4. the bf16 autocast forward runs (unfrozen heads on SDPA/fused flash).

    python nanogpt/test_prior_decomposed.py                # T=512 (full-QK gate)
    python nanogpt/test_prior_decomposed.py --T 1024       # compact-QK gate
    python nanogpt/test_prior_decomposed.py --dense_also 0 # decomposed-only
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT                                 # noqa: E402


def attn_modules(model):
    mods = []
    for blk in model.transformer.h:
        m = blk.main_block
        if hasattr(m, "block"):
            m = m.block
        mods.append(m)
    return mods


def make_alpha_rho(n_head, T):
    """Realistic decomposed logits: local decay (rho) + attention sink (alpha)."""
    k = torch.arange(T).float()
    alpha = torch.zeros(n_head, T)
    rho = torch.empty(n_head, T)
    for h in range(n_head):
        rho[h] = -k / (2.0 + 3.0 * (h % 5))     # per-head decay scale
        alpha[h, 0] = 1.5 + 0.5 * (h % 3)       # per-head sink strength
    return alpha, rho


def make_dense_prior(n_head, T):
    i = torch.arange(T).view(T, 1).float()
    d = (i - torch.arange(T).view(1, T).float())
    w = torch.exp(-d / 7.0).masked_fill(d < 0, 0.0)
    return (w / w.sum(-1, keepdim=True).clamp(min=1e-9)).expand(n_head, T, T).clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dense_also", type=int, default=1,
                    help="1: also dense-freeze one head/layer (mixed states)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device={device}  {args.n_layer}L/{args.n_head}H/{args.n_embd}d  "
          f"T={args.T}  dense_also={args.dense_also}")

    cfg = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    block_size=args.T, vocab_size=50304, dropout=0.0, bias=False,
                    rand_attn_dynamic=True, rand_attn_prior_repr="decomposed")
    model = GPT(cfg).to(device)
    mods = attn_modules(model)
    ok = True

    # 1) memory contract: no T^2 pattern buffer in decomposed mode
    no_patt = all(m.dyn_pattern is None for m in mods)
    print(f"  [buffers] dyn_pattern is None on all layers: {no_patt}")
    ok &= no_patt

    # 2) freeze: 2 decomposed heads / layer (+ optionally 1 dense head / layer)
    alpha, rho = make_alpha_rho(args.n_head, args.T)
    dense_prior = make_dense_prior(args.n_head, args.T)
    dec_idx, dense_idx = [0, 3], ([5] if args.dense_also else [])
    for m in mods:
        m.freeze_heads_decomposed(dec_idx, alpha[dec_idx].to(device),
                                  rho[dec_idx].to(device))
        if dense_idx:
            m.freeze_heads(dense_idx, dense_prior[dense_idx].to(device))
    n_fz = sum(int(m.dyn_frozen.sum()) for m in mods)
    still_no_patt = all(m.dyn_pattern is None for m in mods) if not dense_idx \
        else all(m.dyn_pattern is not None for m in mods)
    print(f"  [freeze] {n_fz} heads frozen "
          f"({len(dec_idx)} decomposed + {len(dense_idx)} dense per layer); "
          f"lazy pattern alloc consistent: {still_no_patt}")
    ok &= still_no_patt

    x = torch.randint(0, 50304, (args.batch, args.T), device=device)

    def set_capture(flag):
        for m in mods:
            m.set_capture(flag)

    # 3) fp32 oracle: eager capture path (materialized patterns) vs cheap path
    model.eval()
    with torch.no_grad():
        set_capture(True)
        ye, _ = model(x, x)
        set_capture(False)
        yc, _ = model(x, x)
    diff = (ye.float() - yc.float()).abs().max().item()
    match = diff < 1e-3
    print(f"  [forward fp32] cheap vs eager max|Δ| = {diff:.2e}  ->  "
          f"{'MATCH' if match else 'MISMATCH'}")
    ok &= match

    # 4) backward: frozen heads' q/k rows exactly zero; alpha/rho get no grad;
    #    live rows get real gradient
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    hd = mods[0].head_dim
    frozen_rows_zero = live_rows_nonzero = True
    for m in mods:
        for h in range(args.n_head):
            for w in (m.q_proj.weight, m.k_proj.weight):
                g = 0.0 if w.grad is None else \
                    w.grad[h * hd:(h + 1) * hd].abs().max().item()
                if h in dec_idx or h in dense_idx:
                    frozen_rows_zero &= (g == 0.0)
                else:
                    live_rows_nonzero &= (g > 0.0)
    no_ar_grad = all(m.dyn_alpha.grad is None and m.dyn_rho.grad is None
                     for m in mods)
    print(f"  [backward] frozen q/k rows all-zero: {frozen_rows_zero}; "
          f"live rows nonzero: {live_rows_nonzero}; alpha/rho grad-free: {no_ar_grad}")
    ok &= frozen_rows_zero and live_rows_nonzero and no_ar_grad

    # 5) bf16 autocast forward runs end to end
    if device == "cuda":
        model.eval()
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x, x)
            print("  [bf16] autocast forward: OK")
        except Exception as e:
            print(f"  [bf16] autocast forward FAILED: {type(e).__name__}: {e}")
            ok = False

    # 6) rollback: unfreeze everything, cheap path == plain model again
    for m in mods:
        m.unfreeze_heads(list(range(args.n_head)))
    model.eval()
    with torch.no_grad():
        set_capture(True)
        ye2, _ = model(x, x)
        set_capture(False)
        yc2, _ = model(x, x)
    d2 = (ye2.float() - yc2.float()).abs().max().item()
    print(f"  [unfreeze fp32] cheap vs eager max|Δ| = {d2:.2e}")
    ok &= d2 < 1e-3

    print(f"\n=> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
