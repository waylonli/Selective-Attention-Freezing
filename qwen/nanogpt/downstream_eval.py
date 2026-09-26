"""
Zero-shot downstream tasks: dense Qwen3-4B vs decomposed-prior replaced heads.

Tasks: HellaSwag, PIQA, ARC-Easy, SST-2, BoolQ with lm-eval-harness style
prompts and loglikelihood scoring (accuracy, plus byte-length-normalised
accuracy for the multiple-choice tasks).

The replaced configuration mirrors the perplexity matrix: KL-guided head
selection and FineWeb-Edu alpha/rho fits loaded from the ext256 checkpoints
(512 calibration sequences). Requests are bucketed by exact token length so
no padding exists and the fused kernel's pure-causal assumption holds.

  python nanogpt/downstream_eval.py --rate 0.1 --out results/prior_corpus/downstream/r10.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_prior_pretrained as TP                                  # noqa: E402
from hybrid_prior_attention import convert_targets_to_hybrid        # noqa: E402
from prior_corpus_matrix import set_attn_impl                       # noqa: E402

MAXLEN = 2048            # fitted alpha/rho cover this window


# --------------------------------------------------------------------------
# task definitions -> list of docs: (context, [continuations], gold)
# --------------------------------------------------------------------------

def _hs_preprocess(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_task(name, limit):
    from datasets import load_dataset
    docs = []
    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for d in ds:
            ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
            q = _hs_preprocess(d["activity_label"] + ": " + ctx)
            docs.append((q, [" " + _hs_preprocess(e) for e in d["endings"]],
                         int(d["label"])))
    elif name == "piqa":
        # ybisk/piqa still ships a loader script, unsupported by datasets>=3;
        # baber/piqa is the same data as parquet (1838 validation rows)
        ds = load_dataset("baber/piqa", split="validation")
        for d in ds:
            docs.append(("Question: " + d["goal"] + "\nAnswer:",
                         [" " + d["sol1"], " " + d["sol2"]], int(d["label"])))
    elif name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        for d in ds:
            labels = d["choices"]["label"]
            if d["answerKey"] not in labels:
                continue
            docs.append(("Question: " + d["question"] + "\nAnswer:",
                         [" " + t for t in d["choices"]["text"]],
                         labels.index(d["answerKey"])))
    elif name == "sst2":
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
        for d in ds:
            docs.append((d["sentence"].strip()
                         + "\nQuestion: Is this sentence positive or negative?\nAnswer:",
                         [" negative", " positive"], int(d["label"])))
    elif name == "boolq":
        ds = load_dataset("google/boolq", split="validation")
        for d in ds:
            docs.append((d["passage"] + "\nQuestion: " + d["question"] + "?\nAnswer:",
                         [" no", " yes"], int(bool(d["answer"]))))
    else:
        raise ValueError(name)
    return docs[:limit] if limit else docs


# --------------------------------------------------------------------------
# loglikelihood scoring with exact-length bucketing (no padding anywhere)
# --------------------------------------------------------------------------

@torch.no_grad()
def score_task(model, tok, docs, device, batch_tokens):
    reqs = []                      # (doc_idx, choice_idx, ids, n_cont, n_bytes)
    for di, (ctx, conts, _gold) in enumerate(docs):
        ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        for ci, cont in enumerate(conts):
            all_ids = tok(ctx + cont, add_special_tokens=False)["input_ids"]
            n_cont = len(all_ids) - len(ctx_ids)
            assert n_cont > 0, (ctx[-40:], cont)
            if len(all_ids) > MAXLEN:
                all_ids = all_ids[-MAXLEN:]
            reqs.append((di, ci, all_ids, n_cont, len(cont.encode())))
    reqs.sort(key=lambda r: len(r[2]))
    scores = {}
    i = 0
    while i < len(reqs):
        n = len(reqs[i][2])
        bs = max(1, batch_tokens // n)
        batch = [r for r in reqs[i:i + bs] if len(r[2]) == n]
        i += len(batch)
        x = torch.tensor([r[2] for r in batch], device=device)
        logits = model(x, use_cache=False).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(
            -1, x[:, 1:].unsqueeze(-1)).squeeze(-1)      # (B, n-1)
        for j, (di, ci, _ids, n_cont, n_bytes) in enumerate(batch):
            s = lp[j, -n_cont:].sum().item()
            scores[(di, ci)] = (s, n_bytes)
    acc = accn = 0
    for di, (_ctx, conts, gold) in enumerate(docs):
        raw = [scores[(di, ci)][0] for ci in range(len(conts))]
        nrm = [scores[(di, ci)][0] / scores[(di, ci)][1] for ci in range(len(conts))]
        acc += int(max(range(len(raw)), key=raw.__getitem__) == gold)
        accn += int(max(range(len(nrm)), key=nrm.__getitem__) == gold)
    return acc / len(docs), accn / len(docs), len(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--source", default="fineweb_edu")
    ap.add_argument("--placement", default="kl_guided",
                    choices=["kl_guided", "variance_guided"],
                    help="head-selection signal; variance_guided needs a "
                         "matching fit file from fit_for_selector.py")
    ap.add_argument("--ckpt_dir", default="results/prior_corpus/matrix_kl_ext256")
    ap.add_argument("--tasks", default="hellaswag,piqa,arc_easy,sst2,boolq")
    ap.add_argument("--batch_tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device, dtype = "cuda", torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    model.config.use_cache = False

    if args.rate > 0:
        stats = torch.load(os.path.join(args.ckpt_dir, "stats_ckpt.pt"),
                           map_location="cpu", weights_only=False)[args.source]
        sig = {"kl_guided": "kl", "variance_guided": "phv"}[args.placement]
        fit_name = (f"fit_{args.source}.pt" if args.placement == "kl_guided"
                    else f"fit_{args.source}_{sig}.pt")
        fit = torch.load(os.path.join(args.ckpt_dir, fit_name),
                         map_location="cpu", weights_only=False)
        targets = TP.discover_softmax_attention(model)
        L, H = len(targets), model.config.num_attention_heads
        rng = torch.Generator().manual_seed(args.seed)
        masks = TP.build_head_masks(args.rate, L, H, "variance_guided", rng,
                                    per_head_var=stats[sig])
        n_frozen = int(sum(int(m.sum()) for m in masks))
        fitted = torch.stack([(a.abs().sum(-1) > 0) for a, _ in fit["decomp"]])
        sel = torch.stack([m.bool() for m in masks])
        missing = int((sel & ~fitted).sum())
        assert missing == 0, (f"{missing} selected heads have no fitted prior in "
                              f"{fit_name}; run fit_for_selector.py --signal {sig}")
        convert_targets_to_hybrid(model, targets, None, masks, device, dtype,
                                  use_triton=True, assume_causal_prefill=True,
                                  decomp=fit["decomp"])
        print(f"converted: {n_frozen}/{L*H} heads frozen (rate {args.rate:.0%}, "
              f"source {args.source}, selector {args.placement})", flush=True)

    results = {"model": args.model, "rate": args.rate, "source": args.source,
               "placement": args.placement, "ckpt_dir": args.ckpt_dir, "tasks": {}}
    for t in args.tasks.split(","):
        t0 = time.time()
        docs = load_task(t, args.limit)
        acc, accn, n = score_task(model, tok, docs, device, args.batch_tokens)
        results["tasks"][t] = {"acc": acc, "acc_norm": accn, "n": n}
        print(f"[{args.placement[:3]} rate {args.rate:.0%}] {t:10s} acc {acc:.4f}  acc_norm {accn:.4f} "
              f"({n} docs, {time.time()-t0:.0f}s)", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
