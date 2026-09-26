"""
Evaluate a from-scratch nanoGPT checkpoint on its val.bin and dump per-token
log-probs for bootstrap significance testing (Shay's scheme).

The eval pass is DETERMINISTIC and SEQUENTIAL: val.bin is cut into
non-overlapping windows of length seq_len, processed in fixed order. This
guarantees that every checkpoint evaluated on the same dataset (same seq_len,
same max_tokens) sees the *same* set of tokens in the same order, so the
resulting per-token log-prob vectors are aligned and can be compared with a
paired bootstrap.

Stochastic-prior configs (prior_gaussian_stochastic) resample attention each
forward pass via the global RNG. We call torch.manual_seed(seed) once before
the pass so the result is reproducible — note this is one fixed sample of the
resampling distribution, not an expectation over it.

Output: a .pt file mapping {label: 1-D tensor of per-token log-probs}, plus a
sidecar .json with metadata (mean NLL, ppl, n_tokens, config). The .pt format
is consumed directly by bootstrap_compare.py.

Usage:
    python nanogpt/eval_nanogpt_logprobs.py \
        --ckpt results/debug/nanogpt/out-fineweb_edu-baseline/ckpt.pt \
        --out  results/nanogpt_bootstrap/fineweb_edu-baseline.pt \
        --label fineweb_edu-baseline
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import GPTConfig, GPT


def sha256_file(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config_from_ckpt(ckpt):
    """Reconstruct a GPTConfig from a checkpoint's saved args, tolerant of
    older field names and of args stored only in the full config dict."""
    ma = dict(ckpt.get("model_args", ckpt.get("model_config", {})))
    if not ma:
        raise ValueError("Checkpoint has neither model_args nor model_config")
    cfg_dict = ckpt.get("config", {}) or {}

    # Old checkpoints stored rand_mask_rate; it was renamed to rand_proj_mask_rate.
    if "rand_mask_rate" in ma:
        ma["rand_proj_mask_rate"] = ma.pop("rand_mask_rate")

    # Some prior-pattern fields were only persisted in the full config dict.
    # Pull them in if missing from model_args so the prior is reconstructed.
    valid_fields = set(GPTConfig.__dataclass_fields__.keys())
    for k in ("rand_attn_prior_path", "rand_attn_pattern_sharing",
              "rand_attn_pattern", "rand_attn_head_rate",
              "rand_attn_head_rate_per_layer"):
        if k not in ma and k in cfg_dict and cfg_dict[k] is not None:
            ma[k] = cfg_dict[k]

    # Keep only keys GPTConfig understands.
    filtered = {k: v for k, v in ma.items() if k in valid_fields}
    return GPTConfig(**filtered)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = build_config_from_ckpt(ckpt)
    model = GPT(cfg)
    layout = ckpt.get("attention_layout")
    if layout is None and ckpt.get("intervention"):
        intervention = ckpt["intervention"]
        mode = intervention.get("physical_mode", intervention["mode"])
        mode = {"prune_gate_taylor": "prune", "random_mean": "mean"}.get(mode, mode)
        layout = {"mode": mode, "heads": intervention["heads"]}
    if layout is not None:
        ckpt["attention_layout"] = layout
        if layout["mode"] in ("prune", "uniform"):
            from supplement.attention import install_packed
            install_packed(model, layout["heads"], layout["mode"])
        elif layout["mode"] not in ("ordinary", "mean"):
            raise ValueError("Unsupported checkpoint attention layout")
    state_dict = ckpt["model"]
    # Strip the compile wrapper prefix if present.
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, cfg, ckpt


def forward_all_positions(model, idx):
    """Run the GPT stack and return logits for ALL positions (B, T, V).

    Mirrors GPT.forward but always computes every position, bypassing the
    last-token-only optimisation that GPT.forward takes when is_masked() is
    False. The attention patterns used here are all causal, so per-position
    next-token log-probs are well defined.
    """
    return model.forward_all_positions(idx)


@torch.no_grad()
def eval_logprobs(model, data, seq_len, batch_size, device, dtype, max_tokens=None):
    """Deterministic sequential pass: non-overlapping windows of seq_len.
    Returns a 1-D tensor of per-token log-probs (true next token)."""
    n = len(data)
    if max_tokens is not None:
        n = min(n, max_tokens)
    # Number of full non-overlapping windows.
    n_windows = (n - 1) // seq_len  # need +1 token for the shifted target
    if n_windows < 1:
        raise ValueError(f"val data too small: {n} tokens, seq_len={seq_len}")

    ctx = (torch.amp.autocast(device_type="cuda", dtype=dtype)
           if device.startswith("cuda") else torch.no_grad())

    lp_chunks = []
    starts = [w * seq_len for w in range(n_windows)]
    for b0 in range(0, n_windows, batch_size):
        batch_starts = starts[b0:b0 + batch_size]
        # input window plus one extra token for the target shift
        xb = torch.stack([
            torch.from_numpy(
                data[s:s + seq_len + 1].astype(np.int64)
            ) for s in batch_starts
        ])  # (B, seq_len+1)
        inputs = xb[:, :-1].to(device)   # (B, T)
        targets = xb[:, 1:].to(device)   # (B, T)
        with ctx:
            logits = forward_all_positions(model, inputs)  # (B, T, V)
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        tok_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
        lp_chunks.append(tok_lp.flatten().cpu())

    return torch.cat(lp_chunks, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--label", default=None,
                    help="Key under which to store the log-prob tensor "
                         "(default: derived from ckpt dir name)")
    ap.add_argument("--data_dir", default=None,
                    help="Dir containing val.bin (default: data/<dataset> from ckpt config)")
    ap.add_argument("--val_file", default="val.bin")
    ap.add_argument("--seq_len", type=int, default=None,
                    help="Default: block_size from checkpoint")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="Cap number of val tokens used (default: all)")
    ap.add_argument("--start_fraction", type=float, default=0.0,
                    help="Inclusive fractional start of val.bin (default: 0)")
    ap.add_argument("--end_fraction", type=float, default=1.0,
                    help="Exclusive fractional end of val.bin (default: 1)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Default: rand_seed from checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    model, cfg, ckpt = load_model(args.ckpt, args.device)

    seq_len = args.seq_len or cfg.block_size
    seed = args.seed if args.seed is not None else getattr(cfg, "rand_seed", 42)
    label = args.label or pathlib.Path(args.ckpt).parent.name

    # Resolve val.bin location.
    if args.data_dir:
        data_dir = args.data_dir
    else:
        dataset = ckpt.get("config", {}).get("dataset")
        assert dataset, "could not infer dataset; pass --data_dir"
        data_dir = os.path.join("data", dataset)
    val_path = os.path.join(data_dir, args.val_file)
    assert os.path.exists(val_path), f"val data not found: {val_path}"

    print(f"ckpt:     {args.ckpt}")
    print(f"label:    {label}")
    print(f"pattern:  {cfg.rand_attn_pattern} | head_rate={cfg.rand_attn_head_rate} "
          f"| sharing={cfg.rand_attn_pattern_sharing}")
    print(f"val:      {val_path}")
    print(f"seq_len:  {seq_len} | batch_size={args.batch_size} | seed={seed} | dtype={args.dtype}")

    # Seed once for reproducibility of any stochastic-prior resampling.
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    if not (0.0 <= args.start_fraction < args.end_fraction <= 1.0):
        raise ValueError("require 0 <= start_fraction < end_fraction <= 1")
    data_start = int(args.start_fraction * len(data))
    data_end = int(args.end_fraction * len(data))
    data = data[data_start:data_end]
    tok_lp = eval_logprobs(model, data, seq_len, args.batch_size,
                           args.device, dtype, args.max_tokens)

    mean_nll = float(-tok_lp.mean())
    ppl = math.exp(mean_nll)
    n_tokens = int(tok_lp.numel())
    print(f"\nmean NLL = {mean_nll:.4f} | ppl = {ppl:.4f} | n_tokens = {n_tokens}")
    if "best_val_loss" in ckpt:
        print(f"(checkpoint best_val_loss = {float(ckpt['best_val_loss']):.4f}, "
              f"random-batch estimate during training)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({label: tok_lp}, args.out)
    meta = {
        "label": label, "ckpt": args.ckpt,
        "checkpoint_sha256": sha256_file(args.ckpt), "val_path": val_path,
        "seq_len": seq_len, "batch_size": args.batch_size, "seed": seed,
        "start_fraction": args.start_fraction, "end_fraction": args.end_fraction,
        "data_start_token": data_start, "data_end_token": data_end,
        "dtype": args.dtype, "mean_nll": mean_nll, "ppl": ppl,
        "n_tokens": n_tokens,
        "pattern": cfg.rand_attn_pattern,
        "head_rate": cfg.rand_attn_head_rate,
        "sharing": cfg.rand_attn_pattern_sharing,
    }
    with open(args.out[:-3] + ".json" if args.out.endswith(".pt")
              else args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved log-probs to {args.out}")


if __name__ == "__main__":
    main()
