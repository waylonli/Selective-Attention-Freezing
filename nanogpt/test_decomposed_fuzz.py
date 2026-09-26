"""
State-transition fuzz for the dynamic-freeze machinery (decomposed mode).

The riskiest surface of this feature is not any single kernel but the STATE
WEB: three per-head states (live / frozen-dense / frozen-decomposed) x lazy
buffers (dyn_pattern, dyn_rho_band) x derived caches (_fz_* indices, _hidx,
_dec_args, _all_patt, _fz_patt, inference weight cache keyed on weight
versions) x four execution paths (fused kernel, split fallback, eager capture
oracle, no-grad inference). This test drives a RANDOM sequence of mutations
and forwards and, after EVERY step, checks the cheap path against the eager
oracle in fp32 and (on train steps) the structural gradient invariants.

Run:  python nanogpt/test_decomposed_fuzz.py [--ops 150] [--seed 0]
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT                                 # noqa: E402

H, NL, NE, S = 4, 2, 128, 128


def attn_modules(model):
    return [getattr(b.main_block, "block", b.main_block) for b in model.transformer.h]


def rand_dense_patterns(k, T, device, rng):
    g = torch.Generator(device="cpu").manual_seed(rng.randrange(1 << 30))
    p = torch.rand(k, T, T, generator=g).to(device).tril()
    return p / p.sum(-1, keepdim=True).clamp(min=1e-9)


def rand_vectors(k, T, device, rng, scale):
    g = torch.Generator(device="cpu").manual_seed(rng.randrange(1 << 30))
    a = (torch.randn(k, T, generator=g) * scale).to(device)
    r = (torch.randn(k, T, generator=g) * scale).to(device)
    return a, r


def oracle_check(model, mods, x, tag, tol=1e-3):
    was = model.training
    model.eval()
    with torch.no_grad():
        for m in mods:
            m.set_capture(True)
        ye, _ = model(x, x)
        for m in mods:
            m.set_capture(False)
        yc, _ = model(x, x)
    model.train(was)
    d = (ye - yc).abs().max().item()
    assert d < tol, f"{tag}: cheap vs eager |Δ|={d:.2e}"
    return d


def grad_check(model, mods, x, tag):
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    hd = mods[0].head_dim
    for m in mods:
        for h in range(H):
            for w in (m.q_proj.weight, m.k_proj.weight):
                g = 0.0 if w.grad is None else \
                    w.grad[h * hd:(h + 1) * hd].abs().max().item()
                if bool(m.dyn_frozen[h]):
                    assert g == 0.0, f"{tag}: frozen head {h} got q/k grad {g:.1e}"
                else:
                    assert g > 0.0, f"{tag}: live head {h} got zero q/k grad"
    model.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    cfg = GPTConfig(n_layer=NL, n_head=H, n_embd=NE, block_size=S,
                    vocab_size=512, dropout=0.0, bias=False,
                    rand_attn_dynamic=True, rand_attn_prior_repr="decomposed")
    model = GPT(cfg).to(device)
    mods = attn_modules(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trace = []
    worst = 0.0
    for step in range(args.ops):
        op = rng.choice(["freeze_dense", "freeze_decomp", "freeze_decomp",
                         "unfreeze", "train_step", "infer", "short_T", "grads"])
        li = rng.randrange(NL)
        m = mods[li]
        heads = rng.sample(range(H), rng.randint(1, H))
        T = S
        if op == "freeze_dense":
            m.freeze_heads(heads, rand_dense_patterns(len(heads), S, device, rng))
        elif op == "freeze_decomp":
            scale = rng.choice([0.5, 2.0, 20.0])   # incl. sharp logits
            a, r = rand_vectors(len(heads), S, device, rng, scale)
            m.freeze_heads_decomposed(heads, a, r)
        elif op == "unfreeze":
            m.unfreeze_heads(heads)
        elif op == "train_step":
            model.train()
            x = torch.randint(0, 512, (args.batch, S), device=device)
            model.zero_grad(set_to_none=True)
            _, loss = model(x, x)
            loss.backward()
            opt.step()          # bumps weight._version -> infer cache must refresh
            model.zero_grad(set_to_none=True)
        elif op == "infer":
            model.eval()
            with torch.no_grad():
                x = torch.randint(0, 512, (args.batch, S), device=device)
                model(x, x)     # exercises the pre-gathered weight cache
        elif op == "short_T":
            T = rng.choice([31, 64, 97])            # < block_size, incl. unaligned
        elif op == "grads":
            x = torch.randint(0, 512, (args.batch, S), device=device)
            grad_check(model, mods, x, f"op{step}:{op}")
        trace.append(f"{op}(L{li},{heads})" if op.startswith(("freeze", "unfreeze"))
                     else op)
        x = torch.randint(0, 512, (args.batch, T), device=device)
        try:
            d = oracle_check(model, mods, x, f"op{step}:{op} T={T}")
        except AssertionError:
            print("TRACE:", " -> ".join(trace[-12:]))
            raise
        worst = max(worst, d)

    states = [[int(m.dyn_frozen[h]) + int(m.dyn_decomp[h]) for h in range(H)]
              for m in mods]
    print(f"fuzz: {args.ops} ops OK on {device}; worst oracle |Δ| = {worst:.2e}; "
          f"final states {states}")
    print("=> PASS")


if __name__ == "__main__":
    main()
