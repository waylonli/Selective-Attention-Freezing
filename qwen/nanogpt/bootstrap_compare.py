"""
Bootstrap confidence interval for per-token log-prob differences between two
runs (Shay's scheme).

Inputs are .pt files saved by test_prior_pretrained.py with
--save_token_log_probs. Each file is a dict {rate (float): 1-D tensor of
per-token log-probs over the eval set}, in the same token order.

Algorithm (per Shay):
  For j = 1..B:
      sample M indices a_1..a_M with replacement (M = total tokens)
      sys1_j = mean(log_prob_sys1[a_i])
      sys2_j = mean(log_prob_sys2[a_i])
      delta_j = sys1_j - sys2_j
  Sort {delta_j}. If 0 lies inside the central (1 - alpha) interval the
  difference is NOT significant; otherwise it IS significant at level alpha
  (two-sided).

Note: we sample the SAME indices for sys1 and sys2 each iteration, so the
two systems are evaluated on the same bootstrap sample (paired bootstrap).
This is more powerful than an unpaired comparison and matches Shay's
pseudocode (one index set per iteration).

Output:
  - prints CI bounds, observed mean diff, p-value (two-sided), verdict
  - optionally writes a JSON summary
"""

import argparse
import json
import os

import numpy as np
import torch


def load_lp(path: str, key=None) -> torch.Tensor:
    """Load a per-token log-prob tensor from a .pt file.

    The file may be:
      - a bare tensor  → returned directly (key ignored)
      - a dict with one entry → that entry is returned (key ignored)
      - a dict with several entries → `key` selects one. Accepts an exact
        match (e.g. a string label) or, for float rate keys, an approximate
        numeric match.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, torch.Tensor):
        return d.float()
    if not isinstance(d, dict):
        raise TypeError(f"unexpected object in {path}: {type(d)}")
    if len(d) == 1:
        return next(iter(d.values())).float()
    if key is None:
        raise KeyError(f"{path} holds multiple entries {list(d.keys())}; "
                       f"specify which one")
    if key in d:
        return d[key].float()
    # Approximate numeric match for float rate keys.
    try:
        for k in d.keys():
            if abs(float(k) - float(key)) < 1e-6:
                return d[k].float()
    except (TypeError, ValueError):
        pass
    raise KeyError(f"key {key!r} not in {path}; available: {list(d.keys())}")


def bootstrap_diff(lp1: np.ndarray, lp2: np.ndarray,
                   B: int, alpha: float, seed: int):
    assert lp1.shape == lp2.shape, \
        f"shape mismatch: sys1={lp1.shape} vs sys2={lp2.shape}"
    M = lp1.shape[0]
    rng = np.random.default_rng(seed)
    deltas = np.empty(B, dtype=np.float64)
    # Process in chunks to avoid materialising an (B, M) index array.
    chunk = max(1, min(B, max(1, 200_000_000 // max(M, 1))))
    j = 0
    while j < B:
        b = min(chunk, B - j)
        idx = rng.integers(0, M, size=(b, M))  # (b, M)
        s1 = lp1[idx].mean(axis=1)
        s2 = lp2[idx].mean(axis=1)
        deltas[j:j+b] = s1 - s2
        j += b
    deltas.sort()
    lo = deltas[int(np.floor((alpha / 2.0) * B))]
    hi = deltas[int(np.ceil((1.0 - alpha / 2.0) * B)) - 1]
    observed = float(lp1.mean() - lp2.mean())
    # Two-sided bootstrap p-value: fraction of deltas at least as extreme as 0
    # under H0 that the true mean is 0. Use the centered deltas trick.
    centered = deltas - observed
    p_two_sided = float(
        (np.sum(centered >= abs(observed)) + np.sum(centered <= -abs(observed)))
        / B
    )
    return {
        "M": int(M),
        "B": int(B),
        "alpha": float(alpha),
        "observed_mean_diff": observed,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_two_sided": p_two_sided,
        "significant": bool(lo > 0 or hi < 0),
    }


def ppl_from_meanlp(mean_lp: float) -> float:
    # mean_lp here is mean of log p(x), so NLL = -mean_lp, ppl = exp(NLL)
    return float(np.exp(-mean_lp))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sys1", required=True,
                    help="Path to .pt with sys1 token log-probs")
    ap.add_argument("--rate1", type=float, default=None,
                    help="Replacement-rate key inside sys1 file (Qwen runs)")
    ap.add_argument("--key1", default=None,
                    help="String key inside sys1 file (nanoGPT runs); "
                         "overrides --rate1. Optional if file has one entry.")
    ap.add_argument("--sys2", required=True,
                    help="Path to .pt with sys2 token log-probs")
    ap.add_argument("--rate2", type=float, default=None,
                    help="Replacement-rate key inside sys2 file (Qwen runs)")
    ap.add_argument("--key2", default=None,
                    help="String key inside sys2 file (nanoGPT runs); "
                         "overrides --rate2. Optional if file has one entry.")
    ap.add_argument("--B", type=int, default=1000,
                    help="Number of bootstrap resamples")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Two-sided significance level (0.05 → 95% CI)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default=None,
                    help="Optional path to write the result summary as JSON")
    args = ap.parse_args()

    sel1 = args.key1 if args.key1 is not None else args.rate1
    sel2 = args.key2 if args.key2 is not None else args.rate2
    lp1 = load_lp(args.sys1, sel1).numpy()
    lp2 = load_lp(args.sys2, sel2).numpy()

    print(f"sys1: {args.sys1} @ {sel1}")
    print(f"       mean log_p = {lp1.mean():.6f}  →  ppl = {ppl_from_meanlp(lp1.mean()):.4f}")
    print(f"       N tokens = {lp1.shape[0]}")
    print(f"sys2: {args.sys2} @ {sel2}")
    print(f"       mean log_p = {lp2.mean():.6f}  →  ppl = {ppl_from_meanlp(lp2.mean()):.4f}")
    print(f"       N tokens = {lp2.shape[0]}")

    res = bootstrap_diff(lp1, lp2, args.B, args.alpha, args.seed)
    res["sys1_path"] = args.sys1
    res["sys1_sel"] = sel1
    res["sys2_path"] = args.sys2
    res["sys2_sel"] = sel2

    print(f"\nBootstrap: B={res['B']}, alpha={res['alpha']:.3f} "
          f"({(1-res['alpha'])*100:.1f}% CI)")
    print(f"  observed Δmean log_p (sys1 - sys2) = {res['observed_mean_diff']:+.6f}")
    print(f"  CI = [{res['ci_lo']:+.6f}, {res['ci_hi']:+.6f}]")
    print(f"  two-sided p-value = {res['p_two_sided']:.4f}")
    verdict = "DIFFERENT" if res["significant"] else "NOT DIFFERENT"
    print(f"  verdict: sys1 and sys2 are {verdict} "
          f"(at α={res['alpha']:.3f})")
    if res["observed_mean_diff"] > 0:
        print(f"  → sys1 has higher mean log_p (lower NLL, lower PPL)")
    else:
        print(f"  → sys2 has higher mean log_p (lower NLL, lower PPL)")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote summary to {args.out_json}")


if __name__ == "__main__":
    main()
