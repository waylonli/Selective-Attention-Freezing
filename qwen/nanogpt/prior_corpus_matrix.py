"""
Prior-corpus transfer matrix on a pretrained GQA model (Qwen3-4B) with
DECOMPOSED (alpha + rho) frozen priors.

Question: which text is the right one to measure attention priors on?
For each PRIOR SOURCE s in {NL, math, code, uniform mixture of the three}
and each EVAL CORPUS e in {NL, math, code}, freeze the lowest-variance
heads (selected with s's own variance) to s's fitted alpha+rho prior at
rates 10 / 20 / 30 %, and report perplexity on e against e's own rate-0
baseline. Output: a 3 (eval) x 4 (source) table with 4 sub-scores per cell.

Everything streams (nanogpt/prior_corpus_data.py): each corpus is a manifest
of shard URLs; extraction reads the first `--extract_batches` windows and
evaluation the following `--eval_batches` windows of the same iterator, so
the spans are disjoint by construction.

The mixed source never re-extracts: the alpha+rho fit only needs the
sufficient statistics c (column mass) and d (diagonal mass), which are linear
in the samples, so the mixture's statistics are the mean of the three; its
head-selection variance follows from the law of total variance.

Usage:
  python nanogpt/prior_corpus_matrix.py --model Qwen/Qwen3-4B \
      --corpora fineweb_edu,openwebmath,the_stack --seq_len 2048 \
      --rates 0,0.1,0.2,0.3 --out_dir results/prior_corpus/matrix
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import test_prior_pretrained as TP                                 # noqa: E402
from hybrid_prior_attention import convert_targets_to_hybrid, restore_originals  # noqa: E402
from dynamic_freeze import decomp_stats, fit_alpha_rho_from_stats  # noqa: E402

MIXED = "__mixed__"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_spans(name, tokenizer_name, seq_len, batch_size, n_extract, n_eval):
    """(extract_windows, eval_windows): lists of (B, seq_len+1) int64 CPU
    tensors, streamed from the corpus in manifest order; disjoint."""
    from prior_corpus_data import corpus_batches
    it = corpus_batches(name, tokenizer_name, seq_len, batch_size)
    ext = [w.clone() for w in itertools.islice(it, n_extract)]
    ev = [w.clone() for w in itertools.islice(it, n_eval)]
    if len(ext) < n_extract or len(ev) < n_eval:
        raise RuntimeError(f"{name}: stream ended early ({len(ext)}/{n_extract} extract, "
                           f"{len(ev)}/{n_eval} eval windows)")
    return ext, ev


# ---------------------------------------------------------------------------
# extraction -> per-corpus sufficient statistics (no T x T kept)
# ---------------------------------------------------------------------------

def _fits_on_gpu(nbytes, margin=8 * 2 ** 30):
    if not torch.cuda.is_available():
        return False
    # blocks cached by the allocator (e.g. the previous corpus' accumulators)
    # are invisible to mem_get_info — release them before deciding, or every
    # corpus after the first falls to the ~25x slower CPU accumulation path
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    return free - nbytes > margin


@torch.no_grad()
def extract_stats(model, targets, windows, seq_len, device, keep_mean=False):
    """Run the eager capture over `windows` and reduce each layer's attention
    to what the decomposed fit and the head selection need (one pass):
      c, d   : (L, H, T) column / diagonal masses of the row-normalized mean
      negH   : (L, H)   mean_i sum_j Pn ln Pn of the MEAN pattern (fit KL term)
      phv    : (L, H)   per-head mean cell variance (E[P^2] - Pbar^2)
      kl     : (L, H)   mean_s KL(P_s || Pbar) = E_s[-H(P_s)] - (-H(Pbar)):
               a scale-free input-dependence signal (cell variance shrinks
               with 1/T^2 for diffuse heads even when they are input-dependent)
      S1, S2 : (L, H, T, T) sums for the mixture (mean and E[P^2])
    The (L, H, T, T) tensors live on CPU only until the caller drops them.
    """
    caps = [TP._AttnCapture(t) for t in targets]
    L = len(targets)
    mean = sq = None
    negH_s = None
    n = 0
    try:
        for w in windows:
            x = w[:, :seq_len].to(device)
            model(x, output_attentions=True, use_cache=False)
            for li, cap in enumerate(caps):
                P = cap.last.float()                              # (B, H, T, T)
                if mean is None:
                    H = P.shape[1]
                    # accumulate on the GPU when it fits (2 x L*H*T*T fp32 =
                    # 38 GB at 36x32x2048 on an H200): the CPU adds were the
                    # bottleneck (~12 s per batch), not the model forward
                    acc_dev = device if _fits_on_gpu(2 * L * H * seq_len * seq_len * 4) else "cpu"
                    mean = torch.zeros(L, H, seq_len, seq_len, device=acc_dev)
                    sq = torch.zeros(L, H, seq_len, seq_len, device=acc_dev)
                    negH_s = torch.zeros(L, H)
                mean[li] += P.sum(0).to(acc_dev)
                sq[li] += (P * P).sum(0).to(acc_dev)
                negH_s[li] += (P.clamp_min(1e-9) * P.clamp_min(1e-9).log()) \
                    .sum(-1).mean(-1).sum(0).cpu()
            n += x.size(0)
    finally:
        for cap in caps:
            cap.remove()
    mean = mean / n
    sq = sq / n
    negH_s /= n
    phv = (sq - mean * mean).clamp_min(0).mean(dim=(-1, -2)).cpu()
    msq = sq.mean(dim=(-1, -2)).cpu()      # per-head E[P^2] cell mean (for the mixture)
    del sq                                  # never keep a second (L,H,T,T) tensor
    mean = mean.cpu()
    L, H, T, _ = mean.shape
    causal = torch.tril(torch.ones(T, T, device=device))
    m = causal.bool()
    c = torch.empty(L, H, T); d = torch.empty(L, H, T); negH = torch.empty(L, H)
    for li in range(L):
        P = mean[li].to(device)
        cc, dd = decomp_stats(P, causal)
        Pn = P.clamp_min(1e-9) * causal
        Pn = Pn / Pn.sum(-1, keepdim=True)
        negH[li] = torch.where(m, Pn * Pn.clamp_min(1e-9).log(),
                               torch.zeros_like(Pn)).sum((-1, -2)).cpu() / T
        c[li], d[li] = cc.cpu(), dd.cpu()
        del P, Pn
    out = dict(c=c, d=d, negH=negH, phv=phv, kl=(negH_s - negH).clamp_min(0),
               negH_s=negH_s, msq=msq, S1=mean)
    if keep_mean:
        out["mean"] = mean
    return out


def mixture_stats(stats_list, n_corpora):
    """Statistics of the uniform mixture of the corpora (law of total variance
    for the selection signal; c, d are linear so they average)."""
    L, H, T, _ = stats_list[0]["S1"].shape
    S1 = stats_list[0]["S1"].clone()
    for s in stats_list[1:]:
        S1 += s["S1"]
    S1 /= n_corpora                                              # mixture mean
    msq = sum(s["msq"] for s in stats_list) / n_corpora          # mean_cells E[P^2]
    negH_s = sum(s["negH_s"] for s in stats_list) / n_corpora   # E_s[-H(P_s)] over the union
    device = "cuda" if torch.cuda.is_available() else "cpu"
    causal = torch.tril(torch.ones(T, T, device=device)); m = causal.bool()
    c = torch.empty(L, H, T); d = torch.empty(L, H, T); negH = torch.empty(L, H)
    m1sq = torch.empty(L, H)
    for li in range(L):
        P = S1[li].to(device)
        cc, dd = decomp_stats(P, causal)
        m1sq[li] = (P * P).mean(dim=(-1, -2)).cpu()
        Pn = P.clamp_min(1e-9) * causal
        Pn = Pn / Pn.sum(-1, keepdim=True)
        negH[li] = torch.where(m, Pn * Pn.clamp_min(1e-9).log(),
                               torch.zeros_like(Pn)).sum((-1, -2)).cpu() / T
        c[li], d[li] = cc.cpu(), dd.cpu()
    phv = (msq - m1sq).clamp_min(0)          # mean_cells Var = E[P^2] - Pbar^2
    return dict(c=c, d=d, negH=negH, phv=phv.float(), kl=(negH_s - negH).clamp_min(0),
                mean=S1)


# ---------------------------------------------------------------------------
# fitting: alpha + rho per selected head, batched, from (c, d) only
# ---------------------------------------------------------------------------

def fit_source(stats, heads, T, device, steps, log):
    """heads: list of (li, h). Returns per-layer decomp list [(alpha, rho)]
    with fitted rows for the given heads (zeros elsewhere), plus fit KLs."""
    L, H = stats["phv"].shape
    alpha = torch.zeros(L, H, T); rho = torch.zeros(L, H, T)
    kls = {}
    causal = torch.tril(torch.ones(T, T, device=device))
    chunk = max(1, int(2 ** 28 // (T * T)))
    t0 = time.time()
    for c0 in range(0, len(heads), chunk):
        part = heads[c0:c0 + chunk]
        li_ = [p[0] for p in part]; h_ = [p[1] for p in part]
        c = stats["c"][li_, h_].to(device); d = stats["d"][li_, h_].to(device)
        a, r, z = fit_alpha_rho_from_stats(c, d, causal, steps=steps)
        ce = (z.sum(-1) - (c * a).sum(-1) - (d * r).sum(-1)) / T
        kl = (stats["negH"][li_, h_].to(device) + ce).cpu()
        alpha[li_, h_] = a.cpu(); rho[li_, h_] = r.cpu()
        for j, (li, h) in enumerate(part):
            kls[(li, h)] = float(kl[j])
    log(f"    fitted {len(heads)} heads in {time.time()-t0:.0f}s; fit KL "
        f"median {torch.tensor(list(kls.values())).median():.3f} "
        f"max {max(kls.values()):.3f} nats/row")
    return [(alpha[li], rho[li]) for li in range(L)], kls


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(model, windows, seq_len, device):
    """Mean NLL / PPL over the eval windows; returns per-token log-probs too."""
    lps = []
    for w in windows:
        w = w.to(device)
        x, y = w[:, :seq_len], w[:, 1:seq_len + 1]
        logits = model(x, use_cache=False).logits.float()
        lp = torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        lps.append(lp.reshape(-1).cpu())
    lp = torch.cat(lps)
    nll = -lp.mean().item()
    return nll, math.exp(nll), lp


def set_attn_impl(model, impl):
    model.config._attn_implementation = impl
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = impl


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--corpora", default="fineweb_edu,openwebmath,the_stack")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--extract_batches", type=int, default=32)
    ap.add_argument("--eval_batches", type=int, default=64)
    ap.add_argument("--rates", default="0,0.1,0.2,0.3")
    ap.add_argument("--repr", default="decomposed", choices=["decomposed", "dense"])
    ap.add_argument("--placement", default="variance_guided",
                    choices=["variance_guided", "kl_guided", "uniform"],
                    help="head selection: lowest cell variance (repo default), lowest "
                         "per-sample KL(P_s||Pbar) (scale-free input-dependence), or random")
    ap.add_argument("--fit_steps", type=int, default=400)
    ap.add_argument("--no_mixed", action="store_true")
    ap.add_argument("--content", default="prior", choices=["prior", "uniform"],
                    help="prior: the fitted corpus prior; uniform: alpha = rho = 0 "
                         "(generic causal-uniform pattern) on the SAME selected heads — "
                         "the control for whether corpus-specific content matters")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000, help="0 disables the CI")
    ap.add_argument("--out_dir", default="results/prior_corpus/matrix")
    args = ap.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    rates = [float(r) for r in args.rates.split(",")]
    names = args.corpora.split(",")
    os.makedirs(args.out_dir, exist_ok=True)
    log_f = open(os.path.join(args.out_dir, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    log(f"== prior-corpus matrix: {args.model} repr={args.repr} T={args.seq_len} "
        f"B={args.batch_size} extract={args.extract_batches} eval={args.eval_batches} "
        f"rates={rates} corpora={names}")

    # ---- data (streamed; disjoint spans) ----
    spans = {}
    for n in names:
        t0 = time.time()
        spans[n] = load_spans(n, args.model, args.seq_len, args.batch_size,
                              args.extract_batches, args.eval_batches)
        log(f"  {n}: {args.extract_batches} extract + {args.eval_batches} eval windows "
            f"of {args.seq_len} tokens streamed in {time.time()-t0:.0f}s")

    # ---- model (eager for extraction) ----
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager").to(device).eval()
    model.config.use_cache = False
    targets = TP.discover_softmax_attention(model)
    L, H = len(targets), model.config.num_attention_heads
    T = args.seq_len
    log(f"  {L} attention layers x {H} heads")

    # ---- extraction -> stats per corpus (checkpointed: a killed run resumes
    # without repeating the expensive capture passes; the T^2 matrices are
    # never saved, only the O(T) sufficient statistics) ----
    sources = list(names) + ([] if args.no_mixed else [MIXED])
    ckpt_stats = os.path.join(args.out_dir, "stats_ckpt.pt")
    stats = {}
    if args.repr == "decomposed" and os.path.exists(ckpt_stats):
        stats = torch.load(ckpt_stats, map_location="cpu", weights_only=False)
        log(f"  resumed extraction stats for {list(stats)} from {ckpt_stats}")
    if any(s not in stats for s in sources):
        for n in names:
            t0 = time.time()
            stats[n] = extract_stats(model, targets, spans[n][0], T, device,
                                     keep_mean=(args.repr == "dense"))
            log(f"  extracted {n} in {time.time()-t0:.0f}s: per-head var median "
                f"{stats[n]['phv'].median():.2e}, per-sample KL(P_s||Pbar) median "
                f"{stats[n]['kl'].median():.3f}")
        if not args.no_mixed:
            stats[MIXED] = mixture_stats([stats[n] for n in names], len(names))
        for n in names:                              # free the T^2 mean matrices
            stats[n].pop("S1", None)
        if args.repr == "decomposed":
            torch.save({s: {k: v for k, v in st.items()
                            if k not in ("S1", "mean")} for s, st in stats.items()},
                       ckpt_stats)
            log(f"  saved extraction stats checkpoint to {ckpt_stats}")

    # ---- head masks per (source, rate): variance-guided, nested in rate ----
    rng = torch.Generator().manual_seed(args.seed)
    sig_key = {"variance_guided": "phv", "kl_guided": "kl"}.get(args.placement)
    masks = {(s, r): TP.build_head_masks(
                 r, L, H, "variance_guided" if sig_key else "uniform", rng,
                 per_head_var=stats[s][sig_key] if sig_key else None)
             for s in sources for r in rates if r > 0}
    for s in sources:
        log(f"  {s}: selection signal '{args.placement}' -> per-head var median "
            f"{stats[s]['phv'].median():.2e}, per-sample KL median {stats[s]['kl'].median():.3f}")

    # ---- fits (decomposed): union of heads at the max rate per source ----
    decomp = {}; fit_kl = {}
    if args.repr == "decomposed":
        for s in sources:
            rmax = max(rates)
            heads = [(li, h) for li in range(L) for h in range(H) if masks[(s, rmax)][li][h]]
            if args.content == "uniform":
                zeros = torch.zeros(H, T)
                decomp[s] = [(zeros, zeros)] * L
                fit_kl[s] = {(li, h): float("nan") for li, h in heads}
                log(f"  source {s}: uniform content on {len(heads)} heads @ {rmax:.0%}")
                continue
            fck = os.path.join(args.out_dir, f"fit_{s}.pt")
            if os.path.exists(fck):
                saved = torch.load(fck, map_location="cpu", weights_only=False)
                decomp[s], fit_kl[s] = saved["decomp"], saved["fit_kl"]
                log(f"  resumed fitted alpha+rho for source {s} from {fck}")
                continue
            log(f"  fitting alpha+rho for source {s} ({len(heads)} heads @ {rmax:.0%})")
            decomp[s], fit_kl[s] = fit_source(stats[s], heads, T, device, args.fit_steps, log)
            torch.save({"decomp": decomp[s], "fit_kl": fit_kl[s]}, fck)

    # ---- switch to flash/SDPA for all evaluations ----
    set_attn_impl(model, "sdpa")
    torch.cuda.empty_cache()

    def bootstrap(lp, base_lp):
        if not args.bootstrap:
            return {}
        try:
            from bootstrap_compare import bootstrap_diff
            bs = bootstrap_diff(lp.numpy(), base_lp.numpy(), args.bootstrap, 0.05, args.seed)
            return {"ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"], "significant": bs["significant"]}
        except Exception as e:   # noqa: BLE001
            return {"bootstrap_error": str(e)}

    base = {}
    for e in names:
        nll, ppl, lp = eval_ppl(model, spans[e][1], T, device)
        base[e] = (ppl, lp)
        log(f"  baseline[{e}] ppl={ppl:.4f} (nll {nll:.4f}, {lp.numel()} tokens)")

    cells = {}
    for s in sources:
        for r in rates:
            if r == 0:
                continue
            for e in names:
                t0 = time.time()
                if args.repr == "decomposed":
                    saved = convert_targets_to_hybrid(
                        model, targets, None, masks[(s, r)], device, dtype,
                        use_triton=True, assume_causal_prefill=True, decomp=decomp[s])
                else:
                    saved = convert_targets_to_hybrid(
                        model, targets, stats[s]["mean"], masks[(s, r)], device, dtype,
                        use_triton=True, assume_causal_prefill=True)
                try:
                    nll, ppl, lp = eval_ppl(model, spans[e][1], T, device)
                finally:
                    restore_originals(saved)
                bppl, blp = base[e]
                cell = {"ppl": ppl, "delta_ppl_pct": (ppl / bppl - 1) * 100,
                        "n_frozen": int(sum(int(m.sum()) for m in masks[(s, r)])),
                        **bootstrap(lp, blp)}
                cells[f"{s}|{r}|{e}"] = cell
                log(f"  [{s:12s} @ {r:.0%} -> {e:12s}] ppl {ppl:.4f} "
                    f"(Δ {cell['delta_ppl_pct']:+.2f}%) {time.time()-t0:.0f}s")
                with open(os.path.join(args.out_dir, "matrix.json"), "w") as f:
                    json.dump({"model": args.model, "repr": args.repr, "seq_len": T,
                               "rates": rates, "corpora": names, "sources": sources,
                               "baseline_ppl": {e: base[e][0] for e in names},
                               "content": args.content, "placement": args.placement,
                               "extract_batches": args.extract_batches,
                               "fit_kl_summary": {s: {"median": float(torch.tensor(list(v.values())).median()),
                                                      "max": max(v.values()), "n": len(v)}
                                                  for s, v in fit_kl.items()},
                               "cells": cells}, f, indent=1)

    # ---- tables: rows = eval corpus, cols = prior source; cell = 4 rates ----
    rate_hdr = " / ".join(f"{int(r*100)}%" for r in rates)
    lines = [f"\n### Perplexity ({args.model}, {args.repr} priors, T={T}); "
             f"cell = PPL at freeze rates {rate_hdr} (0% = the unfrozen dense model)",
             "| eval \\ prior from | " + " | ".join(sources) + " |",
             "|---|" + "---|" * len(sources)]
    for e in names:
        row = [e]
        for s in sources:
            vals = [f"{base[e][0]:.2f}"] + [f"{cells[f'{s}|{r}|{e}']['ppl']:.2f}"
                                            for r in rates if r > 0]
            row.append(" / ".join(vals))
        lines.append("| " + " | ".join(row) + " |")
    lines += [f"\n### ΔPPL% vs each eval corpus' own baseline; cell = rates {rate_hdr}",
              "| eval \\ prior from | " + " | ".join(sources) + " |",
              "|---|" + "---|" * len(sources)]
    for e in names:
        row = [f"{e} (ppl {base[e][0]:.2f})"]
        for s in sources:
            vals = ["+0.00"] + [f"{cells[f'{s}|{r}|{e}']['delta_ppl_pct']:+.2f}"
                                for r in rates if r > 0]
            row.append(" / ".join(vals))
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    log(table)
    with open(os.path.join(args.out_dir, "matrix.md"), "w") as f:
        f.write(table + "\n")
    log(f"saved {args.out_dir}/matrix.json and matrix.md")


if __name__ == "__main__":
    main()
