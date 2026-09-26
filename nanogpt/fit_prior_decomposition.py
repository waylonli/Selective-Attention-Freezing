"""
Can the measured T x T attention prior be decomposed into an ABSOLUTE-position
component plus a RELATIVE-position component?

Model:  P_hat[i, j] = softmax_{j<=i}( alpha[j] + rho[i-j] )
        alpha : (T,)  absolute-key logit  (its matrix is constant per column;
                captures attention-sink / begin-of-sequence structure)
        rho   : (T,)  relative-distance logit (its matrix is Toeplitz;
                captures local decay)
Storage 2T vs T^2; P_hat @ V is computable as two causal convolutions.

Pipeline (all on one GPU, ~5 minutes):
 1. train a small char-GPT (6L/6H/384d, T=256) on shakespeare_char;
 2. capture real attention means over K batches (the controller's measurement);
 3. per head, fit alpha+rho by Adam on masked-softmax KL; ablations: rho-only,
    alpha-only, SVD rank-2 (same-order parameter count), uniform causal;
 4. reconstruction table (KL per head);
 5. THE test that matters: freeze the lowest-variance heads with each
    representation and compare val loss against the dense-mean freeze.

Usage: python nanogpt/fit_prior_decomposition.py [--iters 2000] [--rate 0.25]
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "shakespeare_char")


def get_batch(split, B, T, device):
    d = np.memmap(os.path.join(DATA, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(d) - T - 1, (B,))
    x = torch.stack([torch.from_numpy(d[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def attn_modules(model):
    return [getattr(b.main_block, "block", b.main_block) for b in model.transformer.h]


def train(model, iters, B, T, device):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99),
                            weight_decay=0.1)
    model.train()
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    for it in range(iters):
        x, y = get_batch("train", B, T, device)
        with amp:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == iters - 1:
            print(f"  iter {it:5d}  train loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def capture_means(model, K, B, T, device):
    """Real measured priors: mean post-softmax attention over K*B sequences,
    plus the controller's per-head variance (for head selection)."""
    mods = attn_modules(model)
    model.eval()
    for m in mods:
        m.set_capture(True)
    s1 = [torch.zeros(m.n_head, T, T, device=device) for m in mods]
    s2 = [torch.zeros(m.n_head, T, T, device=device) for m in mods]
    n = 0
    for _ in range(K):
        x, y = get_batch("train", B, T, device)
        model(x, y)
        for li, m in enumerate(mods):
            a = m._last_attn.float()          # (B, nh, T, T)
            s1[li] += a.sum(0)
            s2[li] += (a * a).sum(0)
            n_add = a.shape[0]
        n += n_add
    for m in mods:
        m.set_capture(False)
    means = [s / n for s in s1]
    var = [torch.clamp(s2[li] / n - means[li] ** 2, min=0) for li in range(len(mods))]
    head_var = torch.stack([v.mean(dim=(1, 2)) for v in var])   # (L, nh)
    return means, head_var


def kl_rows(P, Q, causal, eps=1e-9):
    """Mean over rows of KL(P_i || Q_i), restricted to the causal region.
    Masked cells are excluded via where() (0 * log 0 would poison the sum)."""
    m = causal.bool()
    Pn = P.clamp_min(eps) * causal
    Pn = Pn / Pn.sum(-1, keepdim=True)
    Qn = Q.clamp_min(eps) * causal
    Qn = Qn / Qn.sum(-1, keepdim=True)
    contrib = torch.where(
        m, Pn.clamp_min(eps) * (Pn.clamp_min(eps).log() - Qn.clamp_min(eps).log()),
        torch.zeros_like(Pn))
    return contrib.sum(-1).mean().item()


def fit_alpha_rho(P, causal, use_alpha=True, use_rho=True, steps=600, lr=0.05):
    """Fit P_hat = row-softmax(alpha[j] + rho[i-j]) to P by KL, on the GPU."""
    T = P.size(0)
    device = P.device
    idx = torch.arange(T, device=device)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0)     # (T,T) i-j
    # init rho from the diagonal profile of log P (good starting point)
    logP = P.clamp_min(1e-9).log()
    rho0 = torch.stack([logP.diagonal(-d).mean() for d in range(T)])
    alpha = torch.zeros(T, device=device, requires_grad=use_alpha)
    rho = (rho0.clone() if use_rho else torch.zeros(T, device=device)) \
        .requires_grad_(use_rho)
    params = [p for p, u in ((alpha, use_alpha), (rho, use_rho)) if u]
    opt = torch.optim.Adam(params, lr=lr)
    Pn = (P.clamp_min(1e-9) * causal)
    Pn = Pn / Pn.sum(-1, keepdim=True)
    for _ in range(steps):
        L = alpha.view(1, -1) + rho[dist]
        L = L.masked_fill(~causal.bool(), float("-inf"))
        logQ = F.log_softmax(L, dim=-1)
        loss = -(Pn * logQ).sum(-1).mean()      # cross-entropy == KL + const
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        L = alpha.view(1, -1) + rho[dist]
        L = L.masked_fill(~causal.bool(), float("-inf"))
        Q = F.softmax(L, dim=-1)
    return Q.detach(), alpha.detach(), rho.detach()


def svd_rank2(P, causal):
    """Rank-2 SVD reconstruction, clamped to the simplex (parameter count 4T,
    same order as alpha+rho's 2T)."""
    U, S, Vh = torch.linalg.svd(P)
    R = (U[:, :2] * S[:2]) @ Vh[:2]
    R = R.clamp_min(0) * causal
    R = R / R.sum(-1, keepdim=True).clamp_min(1e-9)
    return R


@torch.no_grad()
def val_loss(model, batches):
    model.eval()
    tot = 0.0
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    for x, y in batches:
        with amp:
            _, l = model(x, y)
        tot += float(l)
    return tot / len(batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--K", type=int, default=16, help="measurement batches")
    ap.add_argument("--rate", type=float, default=0.25, help="freeze rate for the val test")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    T = args.T

    meta = pickle.load(open(os.path.join(DATA, "meta.pkl"), "rb"))
    cfg = GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=T,
                    vocab_size=meta["vocab_size"], dropout=0.0, bias=False,
                    rand_attn_dynamic=True)
    model = GPT(cfg).to(device)
    print(f"training 6L/6H/384d T={T} on shakespeare_char ({args.iters} iters)...")
    train(model, args.iters, args.batch, T, device)

    print(f"capturing attention means over {args.K}x{args.batch} sequences...")
    means, head_var = capture_means(model, args.K, args.batch, T, device)
    causal = torch.tril(torch.ones(T, T, device=device))

    # ---- per-head reconstruction quality ----
    L_, H_ = cfg.n_layer, cfg.n_head
    print("\nreconstruction KL(P_mean || P_hat), nats/row  (lower = better; "
          "'unif' = KL to uniform-causal, the no-structure reference):")
    print(f"{'head':>8} | {'a+r':>7} {'r only':>7} {'a only':>7} {'svd r2':>7} {'unif':>7}")
    fits = {}
    tot = {k: [] for k in ("ar", "r", "a", "svd", "unif")}
    for li in range(L_):
        for h in range(H_):
            P = means[li][h]
            Qar, alpha, rho = fit_alpha_rho(P, causal, True, True)
            Qr, _, _ = fit_alpha_rho(P, causal, False, True)
            Qa, _, _ = fit_alpha_rho(P, causal, True, False)
            Qs = svd_rank2(P, causal)
            U = causal / causal.sum(-1, keepdim=True)
            r = {k: kl_rows(P, Q, causal) for k, Q in
                 (("ar", Qar), ("r", Qr), ("a", Qa), ("svd", Qs), ("unif", U))}
            fits[(li, h)] = {"ar": Qar, "r": Qr, "svd": Qs, "uniform": U,
                             "dense": P}
            for k in tot:
                tot[k].append(r[k])
            print(f"  L{li}H{h:>2}  | {r['ar']:7.3f} {r['r']:7.3f} {r['a']:7.3f} "
                  f"{r['svd']:7.3f} {r['unif']:7.3f}")
    print(f"{'MEAN':>8} | " + " ".join(f"{np.mean(tot[k]):7.3f}"
                                       for k in ("ar", "r", "a", "svd", "unif")))

    # ---- the test that matters: freeze with each representation, compare val ----
    k = max(1, int(round(args.rate * H_)))
    sel = [head_var[li].argsort()[:k].tolist() for li in range(L_)]  # lowest-var heads
    print(f"\nfreezing the {k} lowest-variance heads per layer "
          f"(rate={args.rate:.0%}), val loss over 32 fixed batches:")
    torch.manual_seed(args.seed + 1)
    vb = [get_batch("val", args.batch, T, device) for _ in range(32)]
    mods = attn_modules(model)

    def freeze_all(kind):
        for li, m in enumerate(mods):
            m.unfreeze_heads(list(range(H_)))
            if kind is None:
                continue
            pats = torch.stack([fits[(li, h)][kind] for h in sel[li]])
            m.freeze_heads(sel[li], pats)

    freeze_all(None)
    base = val_loss(model, vb)
    print(f"  {'no freeze':>14}: {base:.4f}")
    for kind, label in [("dense", "dense mean"), ("ar", "alpha+rho"),
                        ("r", "rho only"), ("svd", "svd rank-2"),
                        ("uniform", "uniform")]:
        freeze_all(kind)
        v = val_loss(model, vb)
        print(f"  {label:>14}: {v:.4f}  (delta vs no-freeze {v-base:+.4f}, "
              f"vs dense {'--' if kind=='dense' else f'{v-dense_v:+.4f}'})")
        if kind == "dense":
            dense_v = v


if __name__ == "__main__":
    main()


def fit_alpha_rho_batched(Ps, causal, use_alpha=True, use_rho=True,
                          steps=600, lr=0.05):
    """Batched fit over N heads at once: Ps (N,T,T) -> Q (N,T,T), alpha/rho (N,T).
    One optimization loop for all heads (the per-head loop is host-bound on busy
    nodes); identical math to fit_alpha_rho."""
    N, T, _ = Ps.shape
    device = Ps.device
    idx = torch.arange(T, device=device)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0)          # (T,T)
    m = causal.bool()
    logP = Ps.clamp_min(1e-9).log()
    # rho init = per-head mean of logP along each diagonal (vectorized)
    valid = m.view(-1)
    didx = dist.view(-1)[valid]                                       # (nnz,)
    vals = logP.view(N, -1)[:, valid]                                 # (N,nnz)
    dsum = torch.zeros(N, T, device=device).index_add_(1, didx, vals)
    dcnt = torch.bincount(didx, minlength=T).clamp_min(1).float()
    rho0 = dsum / dcnt
    alpha = torch.zeros(N, T, device=device, requires_grad=use_alpha)
    rho = (rho0.clone() if use_rho else torch.zeros(N, T, device=device)) \
        .requires_grad_(use_rho)
    params = [p_ for p_, u in ((alpha, use_alpha), (rho, use_rho)) if u]
    opt = torch.optim.Adam(params, lr=lr)
    Pn = Ps.clamp_min(1e-9) * causal
    Pn = Pn / Pn.sum(-1, keepdim=True)
    for _ in range(steps):
        L = alpha[:, None, :] + rho[:, dist]
        L = L.masked_fill(~m, float("-inf"))
        logQ = F.log_softmax(L, dim=-1)
        loss = -(Pn * logQ).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        L = alpha[:, None, :] + rho[:, dist]
        L = L.masked_fill(~m, float("-inf"))
        Q = F.softmax(L, dim=-1)
    return Q.detach(), alpha.detach(), rho.detach()


