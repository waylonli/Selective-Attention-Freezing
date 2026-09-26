"""
Fit alpha+rho for the heads chosen by a given selection signal, reusing the
sufficient statistics already extracted by prior_corpus_matrix.py.

The ext256 checkpoint only carries fits for the KL-selected heads, so a
variance-selected run needs its own fit file or the unfitted heads would
silently freeze to a uniform prior (which is the content control, not the
selector control).

  python nanogpt/fit_for_selector.py --signal phv --rate 0.3
"""
from __future__ import annotations
import argparse, os, sys, time
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_prior_pretrained as TP                       # noqa: E402
from prior_corpus_matrix import fit_source               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="results/prior_corpus/matrix_kl_ext256")
    ap.add_argument("--source", default="fineweb_edu")
    ap.add_argument("--signal", default="phv", choices=["phv", "kl"])
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--fit_steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda"

    stats = torch.load(os.path.join(args.ckpt_dir, "stats_ckpt.pt"),
                       map_location="cpu", weights_only=False)[args.source]
    L, H = stats["kl"].shape
    rng = torch.Generator().manual_seed(args.seed)
    masks = TP.build_head_masks(args.rate, L, H, "variance_guided", rng,
                                per_head_var=stats[args.signal])
    heads = [(li, h) for li in range(L) for h in range(H) if masks[li][h]]
    print(f"fitting {len(heads)} heads selected by '{args.signal}' @ {args.rate:.0%}",
          flush=True)
    t0 = time.time()
    decomp, fit_kl = fit_source(stats, heads, args.seq_len, device,
                                args.fit_steps, lambda m: print(m, flush=True))
    out = args.out or os.path.join(args.ckpt_dir,
                                   f"fit_{args.source}_{args.signal}.pt")
    torch.save({"decomp": decomp, "fit_kl": fit_kl, "signal": args.signal,
                "rate": args.rate, "n_heads": len(heads)}, out)
    kl = torch.tensor(list(fit_kl.values()))
    print(f"saved {out} in {time.time()-t0:.0f}s; fit KL median {kl.median():.3f} "
          f"max {kl.max():.3f} nats/row", flush=True)


if __name__ == "__main__":
    main()
