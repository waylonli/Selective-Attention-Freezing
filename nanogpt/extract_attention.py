"""
Extract attention pattern statistics from a trained nanoGPT checkpoint.

Runs inference on many batches and accumulates:
  1. Mean attention matrix per (layer, head) — the "typical" attention pattern
  2. Variance per (layer, head, i, j) — how much each weight varies across inputs
  3. A small number of raw attention matrices for visualization

Usage (from repo root):
    python nanogpt/extract_attention.py \
        --ckpt_path results/debug/nanogpt/out-fineweb_edu-baseline/ckpt.pt \
        --dataset fineweb_edu \
        --debug_data \
        --num_batches 50 --batch_size 16

Output: a .pt file containing:
    - "mean":       (n_layer, n_head, T, T) — mean attention weights
    - "var":        (n_layer, n_head, T, T) — variance of attention weights
    - "samples":    (n_samples, n_layer, n_head, T, T) — raw samples for visualization
    - "config", "dataset", "num_sequences", etc.
"""

import os
import sys
import argparse
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import GPTConfig, GPT


def compute_attention(attn_module, x, rope=None):
    """Compute attention logits and weights from a CausalSelfAttention module.

    Returns:
        logits: (B, n_head, T, T) — raw QK^T/sqrt(d) with causal mask (-inf)
        weights: (B, n_head, T, T) — softmax(logits)
    """
    B, T, C = x.size()
    attn = attn_module

    # Compute Q, K
    if attn.n_random_heads > 0 and attn.n_learned_heads > 0:
        q = torch.cat([attn.q_proj_rand(x), attn.q_proj(x)], dim=-1)
        k = torch.cat([attn.k_proj_rand(x), attn.k_proj(x)], dim=-1)
    else:
        q = attn.q_proj(x)
        k = attn.k_proj(x)

    q = q.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
    k = k.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
    q, k = attn._apply_rope(q, k, rope)

    # Pre-softmax logits
    scale = 1.0 / (attn.head_dim ** 0.5)
    logits = (q @ k.transpose(-2, -1)) * scale
    causal_mask = torch.tril(torch.ones(T, T, device=logits.device, dtype=torch.bool))
    logits = logits.masked_fill(~causal_mask, float('-inf'))

    # Post-softmax weights
    weights = F.softmax(logits, dim=-1)  # (B, n_head, T, T)
    return logits, weights


def main():
    parser = argparse.ArgumentParser(description="Extract attention statistics from a trained checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="fineweb_edu")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_raw_samples", type=int, default=4,
                        help="Number of raw attention matrices to save for visualization")
    parser.add_argument("--debug_data", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=args.device, weights_only=False)
    model_args = checkpoint['model_args']

    # Handle renamed parameters from older checkpoints
    if 'rand_mask_rate' in model_args:
        model_args['rand_proj_mask_rate'] = model_args.pop('rand_mask_rate')

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    n_layer = gptconf.n_layer
    n_head = gptconf.n_head
    T = gptconf.block_size
    print(f"Model: {n_layer} layers, {n_head} heads, T={T}, n_embd={gptconf.n_embd}")

    # Get attention modules (unwrap StatefulBlock if needed)
    attn_modules = []
    for block in model.transformer.h:
        m = block.main_block
        if hasattr(m, 'block'):  # StatefulBlock wrapper
            m = m.block
        attn_modules.append(m)

    # Setup data
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    data_dir = os.path.join(repo_root, 'data', args.dataset)
    suffix = '_debug' if args.debug_data else ''
    data_path = os.path.join(data_dir, f'val{suffix}.bin')
    print(f"Loading data from {data_path}")
    data = np.memmap(data_path, dtype=np.uint16, mode='r')

    # Online mean/variance accumulators (Welford's algorithm)
    # Shape: (n_layer, n_head, T, T)
    count = 0
    # Post-softmax weights
    mean = torch.zeros(n_layer, n_head, T, T)
    m2 = torch.zeros(n_layer, n_head, T, T)
    # Pre-softmax logits (only causal entries; -inf entries are excluded)
    logit_mean = torch.zeros(n_layer, n_head, T, T)
    logit_m2 = torch.zeros(n_layer, n_head, T, T)

    # Raw samples for visualization
    raw_samples = []
    raw_logits = []
    n_raw_saved = 0

    print(f"Processing {args.num_batches} batches (batch_size={args.batch_size})...")
    with torch.no_grad():
        for batch_idx in range(args.num_batches):
            ix = torch.randint(len(data) - T, (args.batch_size,))
            x = torch.stack([
                torch.from_numpy(data[i:i+T].astype(np.int64)) for i in ix
            ]).to(args.device)

            hidden, rope = model.prepare_inputs(x)

            batch_attns = []   # post-softmax
            batch_logits = []  # pre-softmax
            for layer_idx, block in enumerate(model.transformer.h):
                attn_input = block.ln_1(hidden)
                lgt, att = compute_attention(
                    attn_modules[layer_idx], attn_input, rope=rope)
                batch_attns.append(att.cpu())
                # Replace -inf with 0 for logit statistics (only track causal entries)
                lgt_clean = lgt.cpu()
                lgt_clean = lgt_clean.masked_fill(lgt_clean == float('-inf'), 0.0)
                batch_logits.append(lgt_clean)
                hidden = block(hidden, rope=rope)

            batch_attns = torch.stack(batch_attns)    # (n_layer, B, n_head, T, T)
            batch_logits = torch.stack(batch_logits)  # (n_layer, B, n_head, T, T)

            # Update online statistics (Welford's algorithm, per-sample)
            for s in range(args.batch_size):
                count += 1
                # Post-softmax
                sample = batch_attns[:, s]
                delta = sample - mean
                mean += delta / count
                delta2 = sample - mean
                m2 += delta * delta2
                # Pre-softmax logits
                lsample = batch_logits[:, s]
                ldelta = lsample - logit_mean
                logit_mean += ldelta / count
                ldelta2 = lsample - logit_mean
                logit_m2 += ldelta * ldelta2

            # Save raw samples
            if n_raw_saved < args.num_raw_samples:
                needed = args.num_raw_samples - n_raw_saved
                take = min(needed, args.batch_size)
                raw_samples.append(batch_attns[:, :take].permute(1, 0, 2, 3, 4))
                raw_logits.append(batch_logits[:, :take].permute(1, 0, 2, 3, 4))
                n_raw_saved += take

            if (batch_idx + 1) % 10 == 0:
                print(f"  batch {batch_idx + 1}/{args.num_batches} ({count} sequences)")

    # Compute variance
    var = m2 / count
    logit_var = logit_m2 / count

    # Concatenate raw samples (store as float16 to save space)
    raw_samples = torch.cat(raw_samples, dim=0).half() if raw_samples else \
        torch.empty(0, n_layer, n_head, T, T, dtype=torch.float16)
    raw_logits = torch.cat(raw_logits, dim=0).half() if raw_logits else \
        torch.empty(0, n_layer, n_head, T, T, dtype=torch.float16)

    print(f"\nDone. Processed {count} sequences.")
    print(f"  weights — mean: {mean.shape}, var: {var.shape}, samples: {raw_samples.shape}")
    print(f"  logits  — mean: {logit_mean.shape}, var: {logit_var.shape}, samples: {raw_logits.shape}")

    # Summary stats
    print(f"\nPer-layer stats:")
    for l in range(n_layer):
        avg_var_w = var[l].mean().item()
        avg_var_l = logit_var[l].mean().item()
        print(f"  layer {l}: weight_var={avg_var_w:.6f}, logit_var={avg_var_l:.4f}")

    # Save
    if args.out_path is None:
        out_dir = os.path.join(repo_root, 'results', 'attention_matrices')
        os.makedirs(out_dir, exist_ok=True)
        args.out_path = os.path.join(out_dir, f'{args.dataset}_attn_stats.pt')

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    save_dict = {
        # Post-softmax attention weights
        "mean": mean,               # (n_layer, n_head, T, T)
        "var": var,                  # (n_layer, n_head, T, T)
        "samples": raw_samples,     # (n_raw, n_layer, n_head, T, T) float16
        # Pre-softmax logits (QK^T / sqrt(d), causal masked, -inf replaced with 0)
        "logit_mean": logit_mean,   # (n_layer, n_head, T, T)
        "logit_var": logit_var,     # (n_layer, n_head, T, T)
        "logit_samples": raw_logits, # (n_raw, n_layer, n_head, T, T) float16
        # Metadata
        "config": model_args,
        "dataset": args.dataset,
        "num_sequences": count,
        "n_layer": n_layer,
        "n_head": n_head,
        "seq_len": T,
    }
    torch.save(save_dict, args.out_path)
    size_mb = os.path.getsize(args.out_path) / 1e6
    print(f"\nSaved to {args.out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
