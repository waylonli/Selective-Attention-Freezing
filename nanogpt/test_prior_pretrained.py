"""
Test attention prior extraction + drop-in replacement on a pretrained HF model.

Direction C (lossy inference compression):
  1. Load a pretrained model (HF transformers).
  2. Discover which layers use *softmax* attention. For hybrid architectures
     such as Qwen3.5 (mix of GatedDeltaNet + standard attention) we only
     touch the softmax-attention layers; linear-attention layers are skipped.
  3. Run inference on an eval slice, capture per-(layer, head, i, j) mean
     attention weight via forward hooks on the softmax-attention modules.
  4. For each replacement rate, replace selected heads' post-softmax weights
     with the empirical mean prior and re-measure perplexity.
  5. Save JSON summary of perplexity vs replacement rate.

Engineering structure (single file, organised into clear sections):
  - Section A: eager_attention_forward override (the actual replacement mechanism)
  - Section B: attention-module discovery (detects softmax attention; ignores
               GatedDeltaNet / Mamba / RWKV / other linear / state-space variants)
  - Section C: prior extraction via forward hooks
  - Section D: evaluation utilities (perplexity)
  - Section E: main() orchestrator

Adding support for a new architecture usually requires only:
  - Confirming the new attention class follows the standard q_proj/k_proj/v_proj/
    o_proj convention (it will be auto-detected if so)
  - If the architecture lives in a new transformers sub-module that also
    uses an `eager_attention_forward + repeat_kv` pair, the post-load
    installer will patch it automatically.

Usage:
    python nanogpt/test_prior_pretrained.py \
        --model path/to/model \
        --replace_rates 0.0,0.25,0.5,0.75 \
        --num_extract_batches 32 \
        --num_eval_batches 64 \
        --offline --sanity_check
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


# ===========================================================================
# Section A — eager_attention_forward override
# ===========================================================================
#
# The HF transformers convention for standard attention is:
#
#     attn_output, attn_weights = eager_attention_forward(
#         module, query, key, value, attention_mask, scaling, dropout, **kwargs)
#
# defined as a free function in the model family's module
# (e.g. transformers.models.qwen3.modeling_qwen3).
#
# We replace that function with a wrapper that:
#   * calls the original to get the real (out, weights)
#   * if the calling module has `_prior_pattern` and `_prior_head_mask`
#     attributes attached, overwrites the selected heads' weights with the
#     prior and recomputes the output as `new_attn @ V` (with proper GQA
#     broadcasting via `repeat_kv`)
#
# Modules that don't have priors attached pass through unchanged, so it is
# safe to leave the patch installed globally.
# ===========================================================================

_ORIG_EAGER: dict = {}        # module-path -> original eager fn
_PATCH_DEBUG = {"calls": 0, "replaced_calls": 0, "first_print": False}


def _make_patched_eager(orig_eager, repeat_kv_fn):
    def patched_eager(module, query, key, value, attention_mask, scaling,
                      dropout=0.0, **kwargs):
        attn_output, attn_weights = orig_eager(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        _PATCH_DEBUG["calls"] += 1

        prior = getattr(module, "_prior_pattern", None)
        head_mask = getattr(module, "_prior_head_mask", None)
        if prior is None or head_mask is None or not head_mask.any():
            return attn_output, attn_weights

        B, H, T, _ = attn_weights.shape
        mask = head_mask.to(attn_weights.device)
        p = prior[:, :T, :T].to(device=attn_weights.device, dtype=attn_weights.dtype)
        new_attn = attn_weights.clone()
        new_attn[:, mask] = p[mask].unsqueeze(0).expand(B, -1, -1, -1)

        value_states = repeat_kv_fn(value, module.num_key_value_groups)
        new_out = torch.matmul(new_attn, value_states).transpose(1, 2).contiguous()
        _PATCH_DEBUG["replaced_calls"] += 1
        if not _PATCH_DEBUG["first_print"]:
            print(f"  [debug] patched_eager fired: replaced {int(mask.sum())} heads, "
                  f"attn={tuple(attn_weights.shape)}, v={tuple(value_states.shape)}")
            _PATCH_DEBUG["first_print"] = True
        return new_out, new_attn
    return patched_eager


# Modules we know exist in stock transformers — patched up-front
_KNOWN_MODULES = [
    "transformers.models.qwen3.modeling_qwen3",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.qwen3_moe.modeling_qwen3_moe",
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
]


def install_known_eager_overrides():
    patched = []
    for mod_path in _KNOWN_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except ModuleNotFoundError:
            continue
        if not (hasattr(mod, "eager_attention_forward") and hasattr(mod, "repeat_kv")):
            continue
        if mod_path not in _ORIG_EAGER:
            _ORIG_EAGER[mod_path] = mod.eager_attention_forward
        mod.eager_attention_forward = _make_patched_eager(
            _ORIG_EAGER[mod_path], mod.repeat_kv)
        patched.append(mod_path.split(".")[-1])
    print(f"  [setup] (pre-load)  patched eager in: {patched}")


def install_eager_overrides_for(targets: List["SoftmaxAttentionTarget"]):
    """Patch eager_attention_forward in whatever Python modules the loaded
    attention classes happen to live in (catches custom variants loaded via
    trust_remote_code that are not in our static list)."""
    mod_names = {type(t.module).__module__ for t in targets}
    patched = []
    for mod_name in mod_names:
        mod = sys.modules.get(mod_name) or _try_import(mod_name)
        if mod is None or not (hasattr(mod, "eager_attention_forward")
                               and hasattr(mod, "repeat_kv")):
            continue
        if mod_name not in _ORIG_EAGER:
            _ORIG_EAGER[mod_name] = mod.eager_attention_forward
        mod.eager_attention_forward = _make_patched_eager(
            _ORIG_EAGER[mod_name], mod.repeat_kv)
        patched.append(mod_name)
    print(f"  [setup] (post-load) patched eager in: {patched}")


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# ===========================================================================
# Section B — attention discovery
# ===========================================================================
#
# A "softmax attention module" is identified by the standard set of linear
# projections: q_proj + k_proj + v_proj + (o_proj or c_proj).
#
# This skips:
#   * Linear/state-space modules (GatedDeltaNet, Mamba, RWKV — they have
#     fused in_proj_qkv and conv layers but no separate q/k/v)
#   * Cross-attention modules without self-attention structure
#
# For hybrid models like Qwen3.5 this naturally selects only the standard
# attention layers, leaving the linear layers untouched.
# ===========================================================================

@dataclass
class SoftmaxAttentionTarget:
    layer_idx: int
    module: nn.Module
    attr_name: str  # name within the block, e.g. "self_attn"


_PROJ_INPUT_NAMES = ("q_proj", "k_proj", "v_proj")
_PROJ_OUTPUT_NAMES = ("o_proj", "c_proj")
_ATTN_ATTR_CANDIDATES = ("self_attn", "attention", "attn", "full_attn")


def _is_softmax_attention(m: nn.Module) -> bool:
    has_qkv = all(hasattr(m, n) for n in _PROJ_INPUT_NAMES)
    has_out = any(hasattr(m, n) for n in _PROJ_OUTPUT_NAMES)
    return has_qkv and has_out


def _iter_blocks(model: nn.Module):
    for path_fn in [
        lambda m: getattr(getattr(m, "model", None), "layers", None),
        lambda m: getattr(getattr(m, "transformer", None), "h", None),
        lambda m: getattr(getattr(getattr(m, "model", None), "decoder", None), "layers", None),
        lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None),
    ]:
        try:
            blocks = path_fn(model)
        except Exception:
            blocks = None
        if blocks is not None:
            return list(blocks)
    raise RuntimeError(f"Cannot find transformer blocks in {type(model).__name__}")


def discover_softmax_attention(model: nn.Module) -> List[SoftmaxAttentionTarget]:
    targets = []
    blocks = _iter_blocks(model)
    for li, block in enumerate(blocks):
        # Try common names first
        found = None
        for name in _ATTN_ATTR_CANDIDATES:
            m = getattr(block, name, None)
            if m is not None and _is_softmax_attention(m):
                found = (m, name)
                break
        # Fallback: scan all named_children
        if found is None:
            for name, m in block.named_children():
                if _is_softmax_attention(m):
                    found = (m, name)
                    break
        if found is not None:
            targets.append(SoftmaxAttentionTarget(li, found[0], found[1]))
    return targets


# ===========================================================================
# Section C — prior extraction via forward hooks
# ===========================================================================
#
# Instead of relying on model.forward(output_attentions=True) — which would
# require every layer to be a standard attention layer — we attach forward
# hooks directly to the targets we discovered. Each hook saves the
# post-softmax attention weights for one layer. We aggregate the mean across
# batches.
# ===========================================================================

class _AttnCapture:
    """Captures post-softmax attention weights from a target on each forward."""

    def __init__(self, target: SoftmaxAttentionTarget):
        self.target = target
        self.last = None
        self.handle = target.module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # HF attention modules return (attn_output, attn_weights, ...)
        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
            self.last = output[1].detach()

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def extract_priors_via_hooks(model, targets, eval_tokens, seq_len, batch_size,
                              num_batches, device, start_batch=0):
    """Run inference on `num_batches` batches starting at `start_batch`,
    capture post-softmax attention weights via forward hooks, and return:
       mean: (n_targets, n_heads, T, T)
       var:  (n_targets, n_heads, T, T)
       per_head_var_scalar: (n_targets, n_heads)  — mean variance per (layer, head)
                            useful as a head-importance heuristic.
    Uses Welford's online algorithm.
    """
    if not targets:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    captures = [_AttnCapture(t) for t in targets]
    model.eval()
    mean = None
    m2 = None
    count = 0
    try:
        for batch in iter_chunks(eval_tokens, seq_len, batch_size, num_batches,
                                 start_batch=start_batch):
            inputs = batch.to(device)
            _ = model(inputs, output_attentions=True)
            if mean is None:
                first = captures[0].last
                if first is None:
                    raise RuntimeError(
                        "Hook didn't capture attention weights. Make sure the model "
                        "was loaded with attn_implementation='eager'.")
                n_heads = first.shape[1]
                mean = torch.zeros(len(targets), n_heads, seq_len, seq_len)
                m2 = torch.zeros(len(targets), n_heads, seq_len, seq_len)
            for idx, cap in enumerate(captures):
                w = cap.last
                if w is None:
                    continue
                # batch-average → one sample per batch for the running stats
                avg = w.float().mean(dim=0).cpu()  # (n_heads, T, T)
                delta = avg - mean[idx]
                mean[idx] += delta / (count + 1)
                delta2 = avg - mean[idx]
                m2[idx] += delta * delta2
            count += 1
    finally:
        for c in captures:
            c.remove()
    var = m2 / max(count, 1)
    # Per-(layer, head) average variance — small scalar for ranking
    per_head_var = var.mean(dim=(-1, -2))  # (n_targets, n_heads)
    return mean, var, per_head_var


# ===========================================================================
# Section D — evaluation utilities
# ===========================================================================

def load_eval_tokens(tokenizer, dataset_name, dataset_config, split,
                     num_tokens, device, parquet_path=None, text_path=None):
    """Load eval text via one of three paths (in priority order):
      1) text_path:    a plain UTF-8 text file (.txt)
      2) parquet_path: a parquet file with a 'text' column (HF datasets format)
      3) datasets.load_dataset(dataset_name, dataset_config, split=split)
    Then tokenize and truncate to `num_tokens`.
    """
    if text_path is not None:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif parquet_path is not None:
        try:
            import pandas as pd  # pandas is normally available
            df = pd.read_parquet(parquet_path)
        except ImportError:
            import pyarrow.parquet as pq
            df = pq.read_table(parquet_path).to_pandas()
        text_col = "text" if "text" in df.columns else df.columns[0]
        text = "\n\n".join(df[text_col].astype(str).tolist())
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, dataset_config, split=split)
        text_col = "text" if "text" in ds.column_names else ds.column_names[0]
        text = "\n\n".join(ds[text_col])

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    if num_tokens > 0:
        ids = ids[: num_tokens]
    return ids.to(device)


def iter_chunks(ids, seq_len, batch_size, num_batches=None, start_batch=0):
    """Yield (B, T) batches starting at `start_batch * batch_size * seq_len`.

    Use `start_batch` to skip a leading region (e.g. to keep extraction tokens
    disjoint from eval tokens).
    """
    total = ids.size(0)
    n_chunks = total // seq_len
    max_batches = n_chunks // batch_size
    end_batch = max_batches if num_batches is None else min(max_batches, start_batch + num_batches)
    for b in range(start_batch, end_batch):
        starts = [(b * batch_size + k) * seq_len for k in range(batch_size)]
        yield torch.stack([ids[s: s + seq_len] for s in starts], dim=0)


@torch.no_grad()
def compute_perplexity(model, eval_tokens, seq_len, batch_size,
                       num_batches, device, start_batch=0,
                       return_token_log_probs=False):
    """Compute perplexity AND record per-step timing + peak VRAM.

    If `return_token_log_probs=True`, also returns a 1-D tensor of length
    `total_tok` containing the per-token log-probability of the true next
    token. Used downstream for bootstrap significance testing.

    Returns:
        mean_nll, ppl, perf dict [, token_log_probs (1-D tensor)]
    """
    model.eval()
    total_nll, total_tok = 0.0, 0
    step_times_ms = []
    token_lp_chunks = [] if return_token_log_probs else None

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for batch in iter_chunks(eval_tokens, seq_len, batch_size, num_batches,
                             start_batch=start_batch):
        inputs = batch.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        # Compute logits directly so we can extract per-token log-probs.
        out = model(inputs)
        logits = out.logits  # (B, T, V)
        # Standard causal-LM shift: predict token[t+1] from logits[t]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        # Per-token log-prob of the true next token
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        tok_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        # tok_lp shape: (B, T-1)
        loss_val = -tok_lp.mean().item()
        if device == "cuda":
            torch.cuda.synchronize()
        step_times_ms.append((time.time() - t0) * 1000)
        n_tok = tok_lp.numel()
        total_nll += loss_val * n_tok
        total_tok += n_tok
        if token_lp_chunks is not None:
            token_lp_chunks.append(tok_lp.flatten().cpu())

    mean_nll = total_nll / max(total_tok, 1)
    ppl = math.exp(mean_nll)

    # Drop the first step (warm-up) for stable medians
    stable = step_times_ms[1:] if len(step_times_ms) > 2 else step_times_ms
    mean_step = sum(stable) / len(stable) if stable else 0.0
    sorted_ = sorted(stable)
    median_step = sorted_[len(sorted_) // 2] if sorted_ else 0.0
    total_time_s = sum(step_times_ms) / 1000.0
    tokens_per_sec = total_tok / max(total_time_s, 1e-9)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0

    perf = {
        "mean_step_ms": mean_step,
        "median_step_ms": median_step,
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": tokens_per_sec,
        "n_tokens": total_tok,
    }
    if return_token_log_probs:
        token_log_probs = torch.cat(token_lp_chunks, dim=0)
        return mean_nll, ppl, perf, token_log_probs
    return mean_nll, ppl, perf


# ===========================================================================
# Section E — replacement helpers + main
# ===========================================================================

def attach_priors(targets, priors, head_masks, dtype, device):
    """Attach (_prior_pattern, _prior_head_mask) to each target's attention
    module. The patched eager_attention_forward picks them up automatically."""
    assert len(targets) == priors.size(0) == len(head_masks)
    for t, p, m in zip(targets, priors, head_masks):
        t.module._prior_pattern = p.to(device, dtype=dtype)
        t.module._prior_head_mask = m.to(device)


def detach_priors(targets):
    for t in targets:
        if hasattr(t.module, "_prior_pattern"):
            del t.module._prior_pattern
        if hasattr(t.module, "_prior_head_mask"):
            del t.module._prior_head_mask


def build_head_masks(rate, n_targets, n_heads, placement, rng, per_head_var=None):
    """Construct per-layer bool masks marking heads to replace with the prior.

    Placement modes:
      - uniform: random subset of `rate * n_heads` heads in every layer
      - first_layers / last_layers: replace ALL heads in the first/last
        `rate * n_targets` layers
      - variance_guided: pick the globally lowest-variance heads first, until
        the total replacement budget is filled. `per_head_var` (shape:
        (n_targets, n_heads)) is required.
    """
    masks = []
    if placement == "first_layers":
        n_full = int(rate * n_targets)
        for li in range(n_targets):
            m = torch.ones(n_heads, dtype=torch.bool) if li < n_full else torch.zeros(n_heads, dtype=torch.bool)
            masks.append(m)
        return masks
    if placement == "last_layers":
        n_full = int(rate * n_targets)
        for li in range(n_targets):
            m = torch.ones(n_heads, dtype=torch.bool) if li >= n_targets - n_full else torch.zeros(n_heads, dtype=torch.bool)
            masks.append(m)
        return masks
    if placement == "variance_guided":
        assert per_head_var is not None, "variance_guided requires per_head_var"
        budget = int(rate * n_targets * n_heads)
        # Rank all (layer, head) pairs by ascending variance
        flat = per_head_var.flatten()  # (n_targets * n_heads,)
        ranked = torch.argsort(flat)   # indices into flat
        selected = ranked[:budget]
        mask_flat = torch.zeros(n_targets * n_heads, dtype=torch.bool)
        mask_flat[selected] = True
        mask_2d = mask_flat.view(n_targets, n_heads)
        return [mask_2d[li] for li in range(n_targets)]
    # uniform (default)
    n_pattern = int(rate * n_heads)
    for li in range(n_targets):
        perm = torch.randperm(n_heads, generator=rng)
        m = torch.zeros(n_heads, dtype=torch.bool)
        m[perm[:n_pattern]] = True
        masks.append(m)
    return masks


def build_pattern_content(pattern_type, mean_attn, n_targets, n_heads, seq_len, rng):
    """Transform the extracted mean prior into one of several pattern types, for
    attribution. Returns (n_targets, n_heads, T, T) whose rows are valid causal
    attention distributions — except 'zero', which is all zeros so the replaced
    heads contribute nothing (the drop-information control).

    The attribution ladder (most→least informative):
      prior    : extracted empirical mean attention (the real, head-matched prior)
      shuffled : real prior patterns permuted across (layer,head) slots — plausible
                 content but assigned to the WRONG head (tests head specificity)
      uniform  : uniform causal average (1/(i+1) on the lower triangle) — generic
      random   : per-head random causal pattern (softmax of N(0,1) logits)
      zero     : replaced heads emit nothing

    Reading: if prior ≈ uniform ≈ shuffled the effect is regularization / removing
    low-information heads; if prior > shuffled > uniform > random > zero the
    specific matched content is doing real work.
    """
    T = seq_len
    if pattern_type == "prior":
        return mean_attn
    if pattern_type == "zero":
        return torch.zeros(n_targets, n_heads, T, T)
    if pattern_type == "uniform":
        u = torch.tril(torch.ones(T, T))
        u = u / u.sum(dim=-1, keepdim=True)
        return u.view(1, 1, T, T).expand(n_targets, n_heads, T, T).contiguous()
    if pattern_type == "random":
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        logits = torch.randn(n_targets, n_heads, T, T, generator=rng)
        logits = logits.masked_fill(~causal, float("-inf"))
        return torch.softmax(logits, dim=-1)
    if pattern_type == "shuffled":
        assert mean_attn is not None, "shuffled needs the extracted prior"
        flat = mean_attn.reshape(n_targets * n_heads, T, T)
        perm = torch.randperm(flat.shape[0], generator=rng)
        return flat[perm].reshape(n_targets, n_heads, T, T).contiguous()
    raise ValueError(f"unknown pattern_type {pattern_type}")


def parse_rates(s):
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--eval_dataset", default="wikitext")
    ap.add_argument("--eval_config", default="wikitext-103-raw-v1")
    ap.add_argument("--eval_split", default="validation")
    ap.add_argument("--eval_parquet", default=None,
                    help="Path to a local parquet file with a 'text' column "
                         "(overrides --eval_dataset/--eval_config/--eval_split)")
    ap.add_argument("--eval_text", default=None,
                    help="Path to a plain text file (overrides everything else)")
    ap.add_argument("--max_eval_tokens", type=int, default=200_000)
    ap.add_argument("--num_extract_batches", type=int, default=32)
    ap.add_argument("--num_eval_batches", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--replace_rates", type=parse_rates, default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--placement", default="uniform",
                    choices=["uniform", "first_layers", "last_layers", "variance_guided"])
    ap.add_argument("--pattern_type", default="prior",
                    choices=["prior", "uniform", "random", "shuffled", "zero"],
                    help="What content to put in the replaced heads (ATTRIBUTION "
                         "control). 'prior' = extracted empirical mean (the real "
                         "prior); 'uniform' = uniform causal average; 'random' = "
                         "per-head random causal pattern; 'shuffled' = real prior "
                         "patterns permuted across (layer,head) slots (tests head "
                         "specificity); 'zero' = replaced heads emit nothing "
                         "(drop-information control). Placement (which heads) is "
                         "unchanged; only the content differs.")
    ap.add_argument("--extract_parquet", default=None,
                    help="Optional separate corpus (parquet w/ 'text') to EXTRACT "
                         "the prior from, for cross-corpus transfer. Eval still uses "
                         "--eval_parquet. If unset, the prior is extracted from the "
                         "disjoint head of the eval corpus (within-corpus).")
    ap.add_argument("--extract_text", default=None,
                    help="Like --extract_parquet but a plain UTF-8 .txt file.")
    ap.add_argument("--out_dir", default="results/qwen_prior")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sanity_check", action="store_true")
    ap.add_argument("--save_token_log_probs", action="store_true",
                    help="Save per-token log-probs to a .pt file next to the "
                         "JSON summary. Required for downstream bootstrap "
                         "significance testing.")
    ap.add_argument("--attn_impl", default="eager",
                    choices=["eager", "sdpa", "flash_attention_2"],
                    help="Attention backend used during prior EXTRACTION. Must be "
                         "'eager' if any prior extraction happens, because Flash/SDPA "
                         "don't return attn_weights. Use --inference_attn_impl to "
                         "switch backends for the perplexity/timing runs.")
    ap.add_argument("--inference_attn_impl", default=None,
                    choices=[None, "eager", "sdpa", "flash_attention_2"],
                    help="Attention backend for the inference (ppl + timing) runs. "
                         "If set, overrides --attn_impl for those runs. The hybrid "
                         "kernel routes its standard-head attention through this "
                         "backend; unconverted layers use it directly. Defaults to "
                         "the same value as --attn_impl.")
    ap.add_argument("--use_hybrid_kernel", action="store_true",
                    help="Use the dedicated HybridPriorAttention kernel that "
                         "actually skips Q-projection and softmax(QK^T) for "
                         "pattern heads. If unset, falls back to the eager-patch "
                         "implementation (correct but no real Q/K-projection savings).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    # A.1 — pre-load patch for known-module attention families
    install_known_eager_overrides()

    # B — load
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} {'(offline)' if args.offline else ''} ...")
    t0 = time.time()
    load_kwargs = dict(trust_remote_code=True)
    if args.offline:
        load_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        attn_implementation=args.attn_impl, **load_kwargs,
    ).to(args.device)
    print(f"  attn_implementation = {args.attn_impl}")
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s; "
          f"params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    cfg = model.config
    n_blocks = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    print(f"  n_blocks={n_blocks}, n_heads={n_heads}, hidden={cfg.hidden_size}, "
          f"head_dim={head_dim}, kv_heads={getattr(cfg, 'num_key_value_heads', n_heads)}")

    # C — discover softmax attention modules (skips linear/state-space layers)
    targets = discover_softmax_attention(model)
    print(f"  [setup] softmax-attention layers found: {len(targets)} / {n_blocks} blocks")
    if targets:
        first = targets[0]
        print(f"  [setup] attention class: {type(first.module).__name__} "
              f"({type(first.module).__module__})")
    if not targets:
        raise RuntimeError("No softmax-attention layers found — nothing to do.")

    # A.2 — patch the actual modules where these classes live
    install_eager_overrides_for(targets)

    # Eval tokens
    if args.eval_text:
        print(f"Loading eval text from file: {args.eval_text}")
    elif args.eval_parquet:
        print(f"Loading eval data from parquet: {args.eval_parquet}")
    else:
        print(f"Loading eval data: {args.eval_dataset}/{args.eval_config} [{args.eval_split}] ...")
    eval_tokens = load_eval_tokens(
        tokenizer, args.eval_dataset, args.eval_config, args.eval_split,
        args.max_eval_tokens, args.device,
        parquet_path=args.eval_parquet, text_path=args.eval_text)
    print(f"  got {eval_tokens.numel()} tokens")

    # Optional: extract the prior from a DIFFERENT corpus (cross-corpus transfer).
    # Eval always uses eval_tokens; only the extraction source changes.
    extract_tokens = eval_tokens
    extract_src_tag = ""
    if args.extract_parquet or args.extract_text:
        src = args.extract_text or args.extract_parquet
        print(f"Loading SEPARATE prior-extraction corpus: {src}")
        extract_tokens = load_eval_tokens(
            tokenizer, args.eval_dataset, args.eval_config, args.eval_split,
            args.max_eval_tokens, args.device,
            parquet_path=args.extract_parquet, text_path=args.extract_text)
        print(f"  got {extract_tokens.numel()} extraction tokens")
        _src_name = os.path.splitext(os.path.basename(src))[0]
        extract_src_tag = f"_priorfrom-{_src_name}"

    # Held-out splits:
    #   extraction: batches [0, num_extract_batches)
    #   evaluation: batches [num_extract_batches, num_extract_batches + num_eval_batches)
    eval_start_batch = args.num_extract_batches
    extracted_token_range = (0, args.num_extract_batches * args.batch_size * args.seq_len)
    eval_token_range = (eval_start_batch * args.batch_size * args.seq_len,
                        (eval_start_batch + args.num_eval_batches) * args.batch_size * args.seq_len)
    print(f"\nSplits (non-overlapping):")
    print(f"  extraction: batches [0, {args.num_extract_batches}) "
          f"= tokens [{extracted_token_range[0]}, {extracted_token_range[1]})")
    print(f"  evaluation: batches [{eval_start_batch}, {eval_start_batch + args.num_eval_batches}) "
          f"= tokens [{eval_token_range[0]}, {eval_token_range[1]})")
    if eval_token_range[1] > eval_tokens.numel():
        print(f"  WARNING: evaluation needs {eval_token_range[1]} tokens but only "
              f"{eval_tokens.numel()} available — reduce --num_eval_batches or "
              f"increase --max_eval_tokens")

    # Store per-token log-probs across all runs if requested
    saved_token_lp = {}  # rate -> tensor

    # Baseline ppl on the held-out eval slice
    print(f"\nBaseline perplexity on held-out slice ({args.num_eval_batches} batches × "
          f"{args.batch_size} × {args.seq_len})...")
    t0 = time.time()
    base_result = compute_perplexity(
        model, eval_tokens, args.seq_len, args.batch_size,
        args.num_eval_batches, args.device, start_batch=eval_start_batch,
        return_token_log_probs=args.save_token_log_probs)
    if args.save_token_log_probs:
        base_loss, base_ppl, base_perf, base_token_lp = base_result
        saved_token_lp[0.0] = base_token_lp
    else:
        base_loss, base_ppl, base_perf = base_result
    print(f"  baseline: loss={base_loss:.4f}, ppl={base_ppl:.3f}  ({time.time()-t0:.1f}s)")
    print(f"  [perf] median_step={base_perf['median_step_ms']:.1f} ms, "
          f"peak_vram={base_perf['peak_vram_mb']:.0f} MB, "
          f"throughput={base_perf['tokens_per_sec']:.0f} tok/s")
    print(f"  [debug] eager calls during baseline: {_PATCH_DEBUG['calls']}, "
          f"replaced: {_PATCH_DEBUG['replaced_calls']}")

    # Sanity check: 100% uniform causal attention on ALL softmax layers
    if args.sanity_check:
        print(f"\nSanity check: forcing uniform causal attention on all "
              f"{len(targets)} softmax-attention layers...")
        uniform = torch.tril(torch.ones(args.seq_len, args.seq_len))
        uniform = uniform / uniform.sum(dim=-1, keepdim=True)
        uniform = uniform.unsqueeze(0).expand(n_heads, -1, -1).contiguous()
        priors = uniform.unsqueeze(0).expand(len(targets), -1, -1, -1).contiguous()
        masks = [torch.ones(n_heads, dtype=torch.bool) for _ in targets]
        attach_priors(targets, priors, masks, dtype, args.device)
        _PATCH_DEBUG["first_print"] = False
        try:
            loss_s, ppl_s, _ = compute_perplexity(
                model, eval_tokens, args.seq_len, args.batch_size,
                args.num_eval_batches, args.device, start_batch=eval_start_batch)
        finally:
            detach_priors(targets)
        print(f"  sanity: loss={loss_s:.4f}, ppl={ppl_s:.3f}  "
              f"(should be MUCH worse than {base_ppl:.3f})")

    # Extract priors from the disjoint extraction slice (must use eager so
    # we can read attn_weights). Skip entirely if no non-zero rates are
    # requested (e.g. when this is a pure baseline run).
    need_extraction = any(r > 0 for r in args.replace_rates)
    if need_extraction:
        if model.config._attn_implementation != "eager":
            raise RuntimeError(
                f"Prior extraction requires attn_impl=eager but model is loaded "
                f"with {model.config._attn_implementation}. Run with --attn_impl eager "
                f"(and optionally --inference_attn_impl sdpa to switch backends post-extraction).")
        print(f"\nExtracting attention priors from extraction slice "
              f"({args.num_extract_batches} batches, attn_impl={model.config._attn_implementation})...")
        t0 = time.time()
        mean_attn, var_attn, per_head_var = extract_priors_via_hooks(
            model, targets, extract_tokens, args.seq_len, args.batch_size,
            args.num_extract_batches, args.device, start_batch=0)
        print(f"  extracted mean {tuple(mean_attn.shape)} ({time.time()-t0:.1f}s)")
        print(f"  per-head mean variance: min={per_head_var.min():.6f}, "
              f"max={per_head_var.max():.6f}, mean={per_head_var.mean():.6f}")
        # Attribution control: replace the prior content with the chosen pattern
        # type. Placement (which heads, via per_head_var) is unchanged; only the
        # content the replaced heads use differs.
        pat_rng = torch.Generator().manual_seed(args.seed)
        pattern_attn = build_pattern_content(
            args.pattern_type, mean_attn, len(targets), n_heads,
            args.seq_len, pat_rng)
        if args.pattern_type != "prior":
            print(f"  pattern_type={args.pattern_type}: replaced prior content "
                  f"with {args.pattern_type} pattern {tuple(pattern_attn.shape)}")
    else:
        print(f"\nSkipping prior extraction — no non-zero replacement rates requested.")
        mean_attn = None
        pattern_attn = None
        per_head_var = None

    # Sweep replacement rates
    results = [{"rate": 0.0, "loss": base_loss, "ppl": base_ppl,
                "delta_ppl_pct": 0.0, "n_replaced_total": 0,
                "perf": base_perf,
                "perf_vs_baseline": {"time_pct": 0.0, "vram_pct": 0.0}}]
    rng = torch.Generator().manual_seed(args.seed)

    # Switch the model's attention backend for the inference runs, if requested.
    # The hybrid kernel inside HybridQwen3_5Attention reads config._attn_implementation
    # at every forward call, so this affects the standard-head dispatch too.
    if args.inference_attn_impl and args.inference_attn_impl != args.attn_impl:
        old_impl = model.config._attn_implementation
        model.config._attn_implementation = args.inference_attn_impl
        # Walk the model and update each sub-config + each module's recorded
        # attn implementation if they cache it.
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = args.inference_attn_impl
        print(f"\nSwitched attention backend for inference: "
              f"{old_impl} -> {args.inference_attn_impl}")
        # Re-measure baseline under the new backend
        print(f"  re-measuring baseline ppl/perf...")
        base_result = compute_perplexity(
            model, eval_tokens, args.seq_len, args.batch_size,
            args.num_eval_batches, args.device, start_batch=eval_start_batch,
            return_token_log_probs=args.save_token_log_probs)
        if args.save_token_log_probs:
            base_loss, base_ppl, base_perf, base_token_lp = base_result
            saved_token_lp[0.0] = base_token_lp
        else:
            base_loss, base_ppl, base_perf = base_result
        print(f"  new baseline: loss={base_loss:.4f}, ppl={base_ppl:.3f}")
        print(f"  [perf] median_step={base_perf['median_step_ms']:.1f} ms, "
              f"peak_vram={base_perf['peak_vram_mb']:.0f} MB, "
              f"throughput={base_perf['tokens_per_sec']:.0f} tok/s")
        # Update results[0] which holds the baseline entry
        if results := locals().get("results"):
            pass
        # results doesn't exist yet at this point — it's built below — so no
        # need to update; the new base_* will flow into the loop naturally.

    # Import hybrid kernel lazily — only when needed
    if args.use_hybrid_kernel:
        from hybrid_prior_attention import (
            convert_targets_to_hybrid, restore_originals)
        print(f"  [setup] hybrid kernel: ENABLED (will skip Q-proj + softmax for pattern heads)")
    else:
        print(f"  [setup] hybrid kernel: disabled (using eager-patch — pattern heads "
              f"still pay Q-proj/softmax cost)")

    for rate in args.replace_rates:
        if rate == 0.0:
            continue
        print(f"\nReplacement rate {rate*100:.0f}% ({args.placement})...")
        masks = build_head_masks(rate, len(targets), n_heads, args.placement, rng,
                                 per_head_var=per_head_var)
        n_replaced = int(sum(m.sum().item() for m in masks))

        saved_state = None
        if args.use_hybrid_kernel:
            # Real Q-proj + softmax skip via module swap
            saved_state = convert_targets_to_hybrid(
                model, targets, pattern_attn, masks, args.device, dtype)
        else:
            # Eager-patch path (correct but no Q-proj savings)
            attach_priors(targets, pattern_attn, masks, dtype, args.device)
            _PATCH_DEBUG["first_print"] = False
            _PATCH_DEBUG["calls"] = 0; _PATCH_DEBUG["replaced_calls"] = 0

        try:
            t0 = time.time()
            ppl_result = compute_perplexity(
                model, eval_tokens, args.seq_len, args.batch_size,
                args.num_eval_batches, args.device, start_batch=eval_start_batch,
                return_token_log_probs=args.save_token_log_probs)
            if args.save_token_log_probs:
                loss, ppl, perf, token_lp_rate = ppl_result
                saved_token_lp[rate] = token_lp_rate
            else:
                loss, ppl, perf = ppl_result
        finally:
            if args.use_hybrid_kernel:
                restore_originals(saved_state)
            else:
                detach_priors(targets)
        delta = (ppl - base_ppl) / base_ppl * 100
        time_pct = (perf["median_step_ms"] / base_perf["median_step_ms"] - 1) * 100
        vram_pct = (perf["peak_vram_mb"] / base_perf["peak_vram_mb"] - 1) * 100
        print(f"  rate={rate}: loss={loss:.4f}, ppl={ppl:.3f}, "
              f"ΔPPL={delta:+.2f}%  ({time.time()-t0:.1f}s)  "
              f"[{n_replaced}/{len(targets)*n_heads} heads replaced]")
        print(f"  [perf] median_step={perf['median_step_ms']:.1f} ms "
              f"({time_pct:+.1f}% vs baseline), "
              f"peak_vram={perf['peak_vram_mb']:.0f} MB ({vram_pct:+.1f}% vs baseline), "
              f"throughput={perf['tokens_per_sec']:.0f} tok/s")
        print(f"  [debug] eager calls={_PATCH_DEBUG['calls']}, "
              f"replaced={_PATCH_DEBUG['replaced_calls']}")
        results.append({
            "rate": rate, "loss": loss, "ppl": ppl,
            "delta_ppl_pct": delta, "n_replaced_total": n_replaced,
            "perf": perf,
            "perf_vs_baseline": {"time_pct": time_pct, "vram_pct": vram_pct},
        })

    # Save summary
    summary = {
        "model": args.model,
        "attn_impl": args.attn_impl,
        "inference_attn_impl": args.inference_attn_impl or args.attn_impl,
        "use_hybrid_kernel": args.use_hybrid_kernel,
        "n_blocks": n_blocks,
        "n_softmax_layers": len(targets),
        "softmax_layer_indices": [t.layer_idx for t in targets],
        "n_heads": n_heads,
        "eval": {
            "dataset": args.eval_dataset, "config": args.eval_config,
            "split": args.eval_split, "seq_len": args.seq_len,
            "num_eval_batches": args.num_eval_batches, "batch_size": args.batch_size,
            "eval_start_batch": eval_start_batch,
        },
        "extraction": {
            "num_extract_batches": args.num_extract_batches,
            "per_head_var": per_head_var.tolist() if per_head_var is not None else None,
        },
        "placement": args.placement,
        "pattern_type": args.pattern_type,
        "extract_source": (args.extract_text or args.extract_parquet or "eval_corpus"),
        "results": results,
    }
    kernel_tag = "hybrid" if args.use_hybrid_kernel else "eagerpatch"
    inf_impl = args.inference_attn_impl or args.attn_impl
    out_path = os.path.join(
        args.out_dir,
        f"{args.model.replace('/', '_').replace(os.sep, '_')}_"
        f"{args.placement}_pat-{args.pattern_type}"
        f"_extract-{args.attn_impl}_infer-{inf_impl}_{kernel_tag}{extract_src_tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")

    # Save per-token log-probs for bootstrap analysis (Shay's request).
    # One .pt file per run, keyed by replacement rate.
    if args.save_token_log_probs and saved_token_lp:
        lp_path = out_path[:-5] + "_token_lp.pt"
        # Store as a plain dict: {rate (float): 1-D tensor of token log-probs}
        torch.save({float(k): v.cpu() for k, v in saved_token_lp.items()}, lp_path)
        n_toks = next(iter(saved_token_lp.values())).numel()
        print(f"Saved per-token log-probs ({len(saved_token_lp)} runs × "
              f"{n_toks} tokens) to {lp_path}")


if __name__ == "__main__":
    main()
