"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Triton frozen-head kernel lives in the repo-root `kernel/` package; scripts run
# with nanogpt/ (not the repo root) on sys.path, so add the root explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from kernel import (frozen_pv, frozen_pv_usable, linear_rows, linear_rows2,
                    linear_permuted_cols, fused_mixed_attn, fused_attn_usable,
                    decomposed_logz, decomp_logits, make_rho_band, BAND_ROWS)

# One fused Triton kernel for frozen+unfrozen heads (no head split, no cat, no
# c_proj weight permutation). Set FROZEN_ATTN_FUSED=0 to fall back to the
# split path (SDPA for unfrozen heads + the frozen-only P@V kernel).
_FUSED = os.environ.get("FROZEN_ATTN_FUSED", "1") != "0"
_FUSED_DISPATCH_COUNT = 0


def fused_dispatch_count():
    """Return this process's mixed frozen-attention kernel dispatch count."""
    return _FUSED_DISPATCH_COUNT


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def rotate_half(x):
    """Llama-style half rotation used by RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply the same rotary position basis to ``(B,H,T,D)`` Q and K."""
    if q.size(-1) != cos.size(-1) or k.size(-1) != cos.size(-1):
        raise ValueError("RoPE cache dimension does not match Q/K head dimension")
    return ((q * cos) + (rotate_half(q) * sin),
            (k * cos) + (rotate_half(k) * sin))


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cache shared by every layer in a GPT forward pass."""

    def __init__(self, dim, max_seq_len, base=10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        # BF16 is the production training path; retaining both caches avoids a
        # dtype conversion/allocation in each of the 12 attention layers.
        self.register_buffer("cos_f32", cos, persistent=False)
        self.register_buffer("sin_f32", sin, persistent=False)
        self.register_buffer("cos_bf16", cos.to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_bf16", sin.to(torch.bfloat16), persistent=False)

    def forward(self, seq_len, dtype):
        if seq_len > self.cos_f32.size(0):
            raise ValueError("sequence length exceeds the configured RoPE cache")
        if dtype == torch.bfloat16:
            cos, sin = self.cos_bf16[:seq_len], self.sin_bf16[:seq_len]
        else:
            cos = self.cos_f32[:seq_len].to(dtype=dtype)
            sin = self.sin_f32[:seq_len].to(dtype=dtype)
        return cos.view(1, 1, seq_len, -1), sin.view(1, 1, seq_len, -1)


# ---------------------------------------------------------------------------
# Randomization helpers
# ---------------------------------------------------------------------------

def parse_components(s):
    """Parse a component string into a set of component names."""
    s = s.strip().lower()
    if s in ("none", ""):
        return set()
    if s == "all":
        return {"q", "k", "v", "o"}
    if s == "qk":
        return {"q", "k"}
    # allow comma-separated like "q,v"
    return set(s.split(","))


def get_random_layer_ids(config):
    """Return set of layer indices that should be fully randomized."""
    n = config.n_layer
    rate = config.rand_layer_rate
    if rate <= 0:
        return set()
    n_rand = max(1, round(rate * n))
    mode = config.rand_layer_mode

    if mode == "explicit":
        if config.rand_layer_ids:
            return {int(x) for x in config.rand_layer_ids.split(",")}
        return set()
    elif mode == "first":
        return set(range(n_rand))
    elif mode == "last":
        return set(range(n - n_rand, n))
    elif mode == "random":
        rng = torch.Generator().manual_seed(config.rand_seed)
        perm = torch.randperm(n, generator=rng).tolist()
        return set(perm[:n_rand])
    else:  # evenly_spaced (default)
        if n_rand >= n:
            return set(range(n))
        step = n / n_rand
        return {int(i * step) for i in range(n_rand)}


# ---------------------------------------------------------------------------
# RandomizationWrapper — wraps nn.Linear for freeze / mask
# ---------------------------------------------------------------------------

class RandomizationWrapper(nn.Module):
    """Wraps an nn.Linear to apply freeze / mask strategies."""

    def __init__(self, linear, strategy, rate=0.0, rng=None):
        super().__init__()
        self.strategy = strategy
        self.rate = rate

        if strategy == "freeze":
            linear.weight.requires_grad_(False)
            if linear.bias is not None:
                linear.bias.requires_grad_(False)
            self.linear = linear

        elif strategy == "mask":
            self.linear = linear
            # Create binary mask: 1 = frozen at init, 0 = learnable
            mask = torch.bernoulli(torch.full(linear.weight.shape, rate,
                                              dtype=torch.float32),
                                   generator=rng).bool()
            self.register_buffer("frozen_mask", mask)
            self.register_buffer("frozen_weight", linear.weight.data.clone())
            if linear.bias is not None:
                bias_mask = torch.bernoulli(torch.full(linear.bias.shape, rate,
                                                       dtype=torch.float32),
                                            generator=rng).bool()
                self.register_buffer("frozen_bias_mask", bias_mask)
                self.register_buffer("frozen_bias", linear.bias.data.clone())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, x):
        if self.strategy == "freeze":
            return self.linear(x)

        elif self.strategy == "mask":
            w = torch.where(self.frozen_mask, self.frozen_weight, self.linear.weight)
            b = None
            if self.linear.bias is not None:
                b = torch.where(self.frozen_bias_mask, self.frozen_bias, self.linear.bias)
            return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# AttentionPatternOverride — replaces softmax(QK^T) with alternatives
# ---------------------------------------------------------------------------

class AttentionPatternOverride(nn.Module):
    """Replaces the standard softmax(QK^T) attention with a fixed or
    structured alternative pattern.

    Supports memory-sharing of the pattern across heads/layers via the
    `rand_attn_pattern_sharing` config:
      - "none"      : each (layer, head) has its own (T, T) pattern (default)
      - "per_layer" : all pattern heads in a layer share one (1, T, T)
      - "global"    : all pattern heads across all layers share one (1, T, T)
    """

    _shared_cache = {}  # class-level cache for tensors shared across instances

    @classmethod
    def reset_shared_cache(cls):
        cls._shared_cache.clear()

    def __init__(self, pattern_type, config, rng=None, layer_idx=0, n_head=None):
        super().__init__()
        self.pattern_type = pattern_type
        self.n_head = n_head if n_head is not None else config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # always use full config for head_dim
        self.dropout = config.dropout
        T = config.block_size
        self.sharing = getattr(config, 'rand_attn_pattern_sharing', 'none')

        Nh = self.n_head

        # When sharing is enabled, the pattern shape collapses the head dim to 1.
        # Broadcasting in `pattern @ V` handles (1,T,T) @ (B,Nh,T,Hd) → (B,Nh,T,Hd).
        sharing = self.sharing
        if sharing in ("per_layer", "global"):
            n_pattern_heads_in_buf = 1
        else:
            n_pattern_heads_in_buf = Nh

        # Cache key: same key → same tensor object → real memory sharing
        if sharing == "global":
            shared_key = ("global", pattern_type, T, config.rand_seed)
        elif sharing == "per_layer":
            shared_key = ("per_layer", pattern_type, T, layer_idx, config.rand_seed)
        else:
            shared_key = None  # no sharing

        # Helper to load prior data (avoid re-reading file on each layer)
        def _load_prior():
            assert config.rand_attn_prior_path, "rand_attn_prior_path required for prior patterns"
            cache_key_prior = ("__prior_file__", config.rand_attn_prior_path)
            if cache_key_prior in AttentionPatternOverride._shared_cache:
                return AttentionPatternOverride._shared_cache[cache_key_prior]
            data = torch.load(config.rand_attn_prior_path, map_location='cpu', weights_only=False)
            AttentionPatternOverride._shared_cache[cache_key_prior] = data
            return data

        # Helper to reduce a per-(layer, head) tensor according to sharing mode.
        # input: tensor of shape (n_layer, n_head, T, T) from the prior file
        # returns: tensor of shape (1, T, T) for shared modes, or (Nh, T, T) for none
        def _select_and_share(prior_tensor):
            if sharing == "global":
                # Average across all layers and the first Nh heads
                return prior_tensor[:, :Nh].mean(dim=(0, 1), keepdim=False).unsqueeze(0)
            elif sharing == "per_layer":
                # Average across heads in this layer
                return prior_tensor[layer_idx, :Nh].mean(dim=0, keepdim=True)
            else:
                return prior_tensor[layer_idx, :Nh]

        def _crop_or_pad(x, target_T, uniform_fill=False):
            cur_T = x.shape[-1]
            if cur_T >= target_T:
                return x[..., :target_T, :target_T]
            padded = torch.zeros(*x.shape[:-2], target_T, target_T)
            padded[..., :cur_T, :cur_T] = x
            if uniform_fill:
                for i in range(cur_T, target_T):
                    padded[..., i, :i+1] = 1.0 / (i + 1)
            return padded

        if pattern_type == "prior":
            # Deterministic mean prior, optionally shared
            if shared_key is not None and shared_key in AttentionPatternOverride._shared_cache:
                pattern = AttentionPatternOverride._shared_cache[shared_key]
            else:
                prior_data = _load_prior()
                prior_mean = _select_and_share(prior_data['mean'].float())
                pattern = _crop_or_pad(prior_mean, T, uniform_fill=True)
                if shared_key is not None:
                    AttentionPatternOverride._shared_cache[shared_key] = pattern
            self.register_buffer("pattern", pattern)

        elif pattern_type in ("prior_gaussian", "prior_gaussian_stochastic"):
            # Gaussian-over-logits prior, optionally shared
            mean_key = (shared_key, "logit_mean") if shared_key is not None else None
            std_key = (shared_key, "logit_std") if shared_key is not None else None

            if mean_key is not None and mean_key in AttentionPatternOverride._shared_cache:
                lm = AttentionPatternOverride._shared_cache[mean_key]
                logit_std = AttentionPatternOverride._shared_cache[std_key]
            else:
                prior_data = _load_prior()
                assert 'logit_mean' in prior_data and 'logit_var' in prior_data, \
                    "Prior file missing logit_mean/logit_var (re-extract with updated script)"
                lm = _crop_or_pad(_select_and_share(prior_data['logit_mean'].float()), T)
                lv = _crop_or_pad(_select_and_share(prior_data['logit_var'].float()), T)
                logit_std = lv.sqrt()
                if mean_key is not None:
                    AttentionPatternOverride._shared_cache[mean_key] = lm
                    AttentionPatternOverride._shared_cache[std_key] = logit_std

            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            self.register_buffer("logit_mean", lm)
            self.register_buffer("logit_std", logit_std)
            self.register_buffer("causal_mask_bool", causal_mask)

            if pattern_type == "prior_gaussian":
                if shared_key is not None and shared_key in AttentionPatternOverride._shared_cache:
                    pattern = AttentionPatternOverride._shared_cache[shared_key]
                else:
                    noise_shape = (n_pattern_heads_in_buf, T, T)
                    noise = torch.randn(noise_shape, generator=rng)
                    logits = lm + logit_std * noise
                    logits = logits.masked_fill(~causal_mask, float('-inf'))
                    pattern = F.softmax(logits, dim=-1)
                    if shared_key is not None:
                        AttentionPatternOverride._shared_cache[shared_key] = pattern
                self.register_buffer("pattern", pattern)

        elif pattern_type == "random":
            if shared_key is not None and shared_key in AttentionPatternOverride._shared_cache:
                pattern = AttentionPatternOverride._shared_cache[shared_key]
            else:
                raw = torch.randn(n_pattern_heads_in_buf, T, T, generator=rng)
                causal = torch.tril(raw)
                causal = causal.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float('-inf'))
                pattern = F.softmax(causal, dim=-1)
                if shared_key is not None:
                    AttentionPatternOverride._shared_cache[shared_key] = pattern
            self.register_buffer("pattern", pattern)

        elif pattern_type == "uniform":
            counts = torch.arange(1, T + 1, dtype=torch.float32).view(-1, 1)
            pattern = torch.tril(torch.ones(T, T)) / counts
            self.register_buffer("pattern", pattern.unsqueeze(0).expand(Nh, -1, -1))

        elif pattern_type == "identity":
            pattern = torch.eye(T).unsqueeze(0).expand(Nh, -1, -1)
            self.register_buffer("pattern", pattern)

        elif pattern_type == "local_window":
            W = config.rand_attn_window
            pattern = torch.zeros(T, T)
            for i in range(T):
                start = max(0, i - W + 1)
                pattern[i, start:i+1] = 1.0 / (i - start + 1)
            self.register_buffer("pattern", pattern.unsqueeze(0).expand(Nh, -1, -1))

        elif pattern_type == "learned":
            self.time_mixer = nn.Parameter(torch.empty(Nh, T, T))
            nn.init.normal_(self.time_mixer, std=0.1 / math.sqrt(T))
            self.register_buffer("causal_mask",
                                 torch.tril(torch.ones(T, T)).view(1, T, T))

        elif pattern_type == "cumulative":
            self.time_mixer = nn.Parameter(torch.empty(1, Nh, self.head_dim, T))
            nn.init.normal_(self.time_mixer, std=0.1 / math.sqrt(T))
            self.register_buffer("causal_mask",
                                 torch.tril(torch.ones(T, T)).view(1, 1, T, T))

        self.attn_dropout = nn.Dropout(config.dropout)

    @property
    def skips_qk(self):
        """Whether this pattern skips Q/K computation entirely."""
        return self.pattern_type not in ("cumulative",)

    def forward(self, v, q=None, k=None):
        """
        v: (B, n_head, T, head_dim)
        For fixed patterns and "learned": only v is used.
        For "cumulative": q,k are ignored; uses cumsum on v.
        """
        B, Nh, T, Hd = v.shape

        if self.pattern_type in ("prior", "prior_gaussian", "random", "uniform", "identity", "local_window"):
            attn = self.pattern[:, :T, :T]  # (n_head, T, T)
            attn = self.attn_dropout(attn)
            y = attn @ v  # (B, n_head, T, head_dim)
            return y

        elif self.pattern_type == "prior_gaussian_stochastic":
            # Resample logits each forward pass, then softmax
            lm = self.logit_mean[:, :T, :T]          # (Nh, T, T)
            ls = self.logit_std[:, :T, :T]
            mask = self.causal_mask_bool[:T, :T]     # (T, T)
            noise = torch.randn_like(lm)
            logits = lm + ls * noise
            logits = logits.masked_fill(~mask, float('-inf'))
            attn = F.softmax(logits, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v
            return y

        elif self.pattern_type == "learned":
            attn = self.time_mixer[:, :T, :T] * self.causal_mask[:, :T, :T]
            attn = self.attn_dropout(attn)
            y = attn @ v
            return y

        elif self.pattern_type == "cumulative":
            # O(n) running average
            counts = torch.arange(1, T + 1, dtype=v.dtype, device=v.device).view(1, 1, -1, 1)
            cum_v = torch.cumsum(v, dim=2) / counts  # (B, Nh, T, Hd)
            # Mix via learned parameter → produces (B, Nh, T, T) style attention
            mixer = cum_v @ self.time_mixer[:, :, :, :T]  # (B, Nh, T, T)
            mixer = mixer[:, :, :T, :T] * self.causal_mask[:, :, :T, :T]
            y = mixer @ v  # (B, Nh, T, Hd)
            return y


# ---------------------------------------------------------------------------
# apply_randomization — factory function called in Block.__init__
# ---------------------------------------------------------------------------

def _wrap_projection(proj, components, comp_name, strategy, rate, rng):
    """Conditionally wrap a single projection with RandomizationWrapper."""
    if comp_name in components:
        return RandomizationWrapper(proj, strategy, rate=rate, rng=rng)
    return proj


def apply_randomization(attn, config, layer_idx, rng):
    """Apply randomization strategies to a CausalSelfAttention module."""
    random_layer_ids = get_random_layer_ids(config)
    components = parse_components(config.rand_components)

    # Method B: if this is a "random layer", freeze everything
    if layer_idx in random_layer_ids:
        for name in ("q_proj", "k_proj", "v_proj", "c_proj"):
            proj = getattr(attn, name)
            setattr(attn, name, RandomizationWrapper(proj, "freeze"))
        return attn

    # Determine strategy: full freeze (rate >= 1.0) or partial mask
    rate = config.rand_proj_mask_rate
    if rate >= 1.0:
        strategy = "freeze"
    else:
        strategy = "mask"

    # Component-selective wrapping
    if components:
        comp_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "c_proj"}
        for comp_key, attr_name in comp_map.items():
            proj = getattr(attn, attr_name)
            wrapped = _wrap_projection(proj, components, comp_key, strategy, rate, rng)
            setattr(attn, attr_name, wrapped)

    # Method F: attention pattern override
    if config.rand_attn_pattern not in ("none", ""):
        attn.pattern_override = AttentionPatternOverride(
            config.rand_attn_pattern, config, rng=rng, layer_idx=layer_idx)

    return attn


# ---------------------------------------------------------------------------
# StatefulBlock (kept for recurrent state experiments)
# ---------------------------------------------------------------------------

class StatefulBlock(nn.Module):

    def __init__(self, block, config):
        super().__init__()
        self.block = block
        self.state_size = config.state_size
        self.n_embd = config.n_embd
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.linear = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.current_step = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.prev_state = None
        self.reset_state()

    def is_masked(self):
        if hasattr(self.block, "is_masked"):
            return self.block.is_masked()
        return True

    def reset_state(self):
        device = self.current_step.device
        self.current_step.copy_(torch.tensor(0, device=device))
        size = (1, self.state_size, self.n_embd)
        self.prev_state = torch.zeros(*size, device=device)

    def forward(self, x, rope=None):
        B, T, C = x.size()
        self.current_step += 1

        if self.prev_state is not None:
            prev_state = self.prev_state.to(x.device)
            prev_state = self.ln(prev_state)
            if prev_state.size(0) == 1:
                prev_state = prev_state.expand(x.size(0), -1, -1)
            x = torch.cat([prev_state, x], 1)

        x = self.block(x, rope=rope)
        if self.prev_state is not None:
            x = x[:, self.state_size:]

        state_update_size = min(self.prev_state.size(1), x.size(1))
        prev_state = self.prev_state.roll(-state_update_size, 1)
        if prev_state.size(0) == 1:
            prev_state = prev_state.expand(x.size(0), -1, -1).clone()
        prev_state[:,-state_update_size:] = x[:,-state_update_size:].detach()
        self.prev_state = prev_state
        return x


# ---------------------------------------------------------------------------
# CausalSelfAttention — with split projections and per-head randomization
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.position_encoding = config.position_encoding

        # --- Hybrid attention heads (Method H) ---
        # Some heads use a fixed attention pattern (no Q/K), rest use standard attention.
        # Optionally a per-layer head rate string overrides the uniform value.
        per_layer = getattr(config, 'rand_attn_head_rate_per_layer', '')
        if per_layer:
            # Accept both string ("0.75,1.0,...") and tuple/list (from configurator parsing)
            if isinstance(per_layer, str):
                rates = [float(x) for x in per_layer.split(',')]
            else:
                rates = [float(x) for x in per_layer]
            assert len(rates) == config.n_layer, \
                f"rand_attn_head_rate_per_layer has {len(rates)} entries, expected {config.n_layer}"
            head_rate_for_this_layer = rates[self.layer_idx]
        else:
            head_rate_for_this_layer = config.rand_attn_head_rate
        n_pattern = int(head_rate_for_this_layer * config.n_head)
        n_standard = config.n_head - n_pattern
        self.n_pattern_heads = n_pattern
        self.n_standard_heads = n_standard

        # --- Per-head projection randomization (Method A) ---
        # Only applies to the standard (non-pattern) heads.
        if n_pattern > 0:
            # Hybrid mode: pattern heads + standard heads
            n_random = 0  # Method A doesn't apply in hybrid mode
            n_learned = n_standard
        else:
            n_random = int(config.rand_head_rate * config.n_head)
            n_learned = config.n_head - n_random
        self.n_random_heads = n_random
        self.n_learned_heads = n_learned

        # Scaled random projection dimension (Method G)
        self.rand_head_dim = int(self.head_dim * config.rand_proj_scale)
        self.has_scaled_random = (n_random > 0 and n_learned > 0
                                  and self.rand_head_dim != self.head_dim)

        # --- Build projections ---
        if n_pattern > 0:
            # Hybrid: pattern heads only need V; standard heads (if any) need Q/K/V
            rng = torch.Generator().manual_seed(config.rand_seed)
            self.v_proj_pattern = nn.Linear(config.n_embd, n_pattern * self.head_dim, bias=config.bias)
            if n_standard > 0:
                self.q_proj = nn.Linear(config.n_embd, n_standard * self.head_dim, bias=config.bias)
                self.k_proj = nn.Linear(config.n_embd, n_standard * self.head_dim, bias=config.bias)
                self.v_proj = nn.Linear(config.n_embd, n_standard * self.head_dim, bias=config.bias)
            # Fixed attention pattern for the pattern heads
            pattern_type = config.rand_attn_pattern if config.rand_attn_pattern not in ("none", "") else "random"
            self.head_pattern_override = AttentionPatternOverride(
                pattern_type, config, rng=rng, layer_idx=self.layer_idx, n_head=n_pattern)
        elif n_random > 0 and n_learned > 0:
            # Method A: frozen random projections + learned projections
            rng = torch.Generator().manual_seed(config.rand_seed)
            self.q_proj_rand = nn.Linear(config.n_embd, n_random * self.rand_head_dim, bias=config.bias)
            self.k_proj_rand = nn.Linear(config.n_embd, n_random * self.rand_head_dim, bias=config.bias)
            self.v_proj_rand = nn.Linear(config.n_embd, n_random * self.rand_head_dim, bias=config.bias)
            for p in [self.q_proj_rand, self.k_proj_rand, self.v_proj_rand]:
                p.weight.requires_grad_(False)
                if p.bias is not None:
                    p.bias.requires_grad_(False)
            self.q_proj = nn.Linear(config.n_embd, n_learned * self.head_dim, bias=config.bias)
            self.k_proj = nn.Linear(config.n_embd, n_learned * self.head_dim, bias=config.bias)
            self.v_proj = nn.Linear(config.n_embd, n_learned * self.head_dim, bias=config.bias)
        else:
            # Standard: all heads same type
            self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection
        if n_random > 0 and n_learned > 0:
            c_proj_in = n_random * self.rand_head_dim + n_learned * self.head_dim
        else:
            c_proj_in = config.n_embd  # hybrid and standard both output n_embd
        self.c_proj = nn.Linear(c_proj_in, config.n_embd, bias=config.bias)
        self.c_proj_in = c_proj_in

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        # Will be set by apply_randomization if needed (global pattern override)
        self.pattern_override = None

        # Attention weight randomization (Method E)
        self.attn_mask_rate = config.rand_attn_mask_rate
        if self.attn_mask_rate > 0:
            rng = torch.Generator().manual_seed(config.rand_seed + 9999)
            causal = torch.tril(torch.ones(config.n_head, config.block_size, config.block_size))
            rand_mask = torch.bernoulli(torch.full_like(causal, self.attn_mask_rate), generator=rng)
            diag = torch.eye(config.block_size).unsqueeze(0).expand_as(rand_mask)
            rand_mask = rand_mask * (1 - diag)
            rand_mask = rand_mask * torch.tril(torch.ones_like(rand_mask))
            self.register_buffer("attn_rand_mask", rand_mask)

        # Dynamic per-head freezing (Version A: gradual attention training).
        # When enabled, heads can be frozen to a fixed pattern at runtime (during
        # training); frozen heads use the stored pattern instead of softmax(QK^T),
        # so their Q/K receive no gradient (V stays trainable). All heads keep full
        # projections so the architecture is constant — only the mask changes.
        self.dynamic = bool(getattr(config, "rand_attn_dynamic", False))
        self.prior_repr = getattr(config, "rand_attn_prior_repr", "dense")
        if self.dynamic:
            pattern_dtype = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }[config.freeze_pattern_dtype]
            self._pattern_dtype = pattern_dtype
            self.register_buffer("dyn_frozen", torch.zeros(config.n_head, dtype=torch.bool))
            if self.prior_repr == "decomposed":
                # O(T) per head: the frozen prior is softmax(alpha[j] + rho[i-j])
                # rebuilt in-register by the kernel; z holds the precomputed row
                # normalizers. The dense (n_head, S, S) buffer is NOT allocated —
                # that is the memory win — and appears lazily only if a dense
                # freeze_heads() call ever asks for it.
                assert config.block_size % BAND_ROWS == 0, (
                    f"rand_attn_prior_repr='decomposed' needs block_size divisible "
                    f"by {BAND_ROWS} (got {config.block_size}) — fail here, not at "
                    f"the first freeze hundreds of iterations in")
                S = config.block_size
                self.register_buffer("dyn_decomp", torch.zeros(config.n_head, dtype=torch.bool))
                self.register_buffer("dyn_alpha", torch.zeros(config.n_head, S))
                self.register_buffer("dyn_rho", torch.zeros(config.n_head, S))
                self.register_buffer("dyn_z", torch.zeros(config.n_head, S))
                # aligned rho band (derived; the kernel cannot vectorize a raw
                # Toeplitz tile load — see make_rho_band). (n_head, 64, S),
                # allocated lazily at the first decomposed freeze.
                self.register_buffer("dyn_rho_band", None)
                self.register_buffer("dyn_pattern", None)
            else:
                self.register_buffer("dyn_pattern",
                                     torch.zeros(config.n_head, config.block_size,
                                                 config.block_size, dtype=pattern_dtype))
            self.register_buffer("_dyn_causal",
                                 torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
                                 persistent=False)
            self._capture = False     # set by the freeze controller to grab attn for variance
            self._capture_logits = True
            self._retain_attention = True
            self._last_attn = None
            self._last_logits = None  # raw pre-mask scores, for the gaussian (sampled) prior
            self._fz_dirty = True     # recompute cached freeze indices on next cheap forward
            self._all_patt = None     # lazy materialized-pattern cache (set in refresh too)

    def _apply_rope(self, q, k, rope):
        if self.position_encoding != "rope":
            return q, k
        if rope is None:
            raise ValueError("RoPE Q/K requested without a rotary cache")
        return apply_rotary_pos_emb(q, k, *rope)

    def set_capture(self, flag, capture_logits=True, retain_attention=True):
        """Toggle eager attention and optional retention of its dense matrix.

        Long-context measurement enables the eager path in every layer for
        numerical parity, but retains one layer at a time so dense attention
        matrices cannot accumulate across the full model.
        """
        if getattr(self, "dynamic", False):
            self._capture = flag
            self._capture_logits = bool(capture_logits)
            self._retain_attention = bool(retain_attention)
            if not flag:
                self._last_attn = None
                self._last_logits = None

    def freeze_heads(self, head_idx, patterns):
        """Freeze the given heads to fixed attention patterns (T x T each).
        head_idx: iterable of head indices; patterns: (len(head_idx), T, T)."""
        head_idx = list(head_idx)
        if patterns.ndim != 3 or patterns.size(-2) != patterns.size(-1):
            raise ValueError("frozen patterns must have shape (n_heads, T, T)")
        if patterns.size(0) != len(head_idx):
            raise ValueError("one frozen pattern is required per head index")
        T = patterns.size(-1)
        if T > self._dyn_causal.size(-1):
            raise ValueError("frozen pattern exceeds the configured block size")
        # Both the eager and Triton paths assume causal, row-stochastic patterns.
        # Enforce that boundary once at freeze time rather than inside each step.
        patterns = patterns.tril()
        patterns = patterns / patterns.sum(-1, keepdim=True).clamp_min(1e-12)
        if self.dyn_pattern is None:
            # decomposed mode: the dense buffer exists only if someone actually
            # dense-freezes (keeps the O(T)-storage promise otherwise)
            S = self._dyn_causal.size(0)
            self.dyn_pattern = torch.zeros(self.n_head, S, S,
                                           device=self.dyn_frozen.device,
                                           dtype=self._pattern_dtype)
        for j, h in enumerate(head_idx):
            self.dyn_frozen[h] = True
            self.dyn_pattern[h, :T, :T] = patterns[j].to(self.dyn_pattern.device,
                                                         self.dyn_pattern.dtype)
            if self.prior_repr == "decomposed":
                self.dyn_decomp[h] = False
        self._fz_dirty = True

    def freeze_heads_decomposed(self, head_idx, alpha, rho, z=None):
        """Freeze heads to the decomposed prior softmax(alpha[j] + rho[i-j]).
        head_idx: iterable of head indices; alpha, rho: (len(head_idx), T) logit
        vectors; z: optional matching normalizers if the caller already computed
        them (e.g. the controller's fit), else derived here. The normalizers are
        always computed in fp32 — even on a bf16-cast model — so only the final
        storage rounds (a bf16 logsumexp over up to S terms would systematically
        mis-normalize every pattern row)."""
        assert self.prior_repr == "decomposed", \
            "freeze_heads_decomposed requires rand_attn_prior_repr='decomposed'"
        T = alpha.size(-1)
        buf = self.dyn_alpha
        idx = torch.as_tensor(list(head_idx), dtype=torch.long, device=buf.device)
        a32 = alpha.to(buf.device, torch.float32)
        r32 = rho.to(buf.device, torch.float32)
        z32 = decomposed_logz(a32, r32) if z is None \
            else z.to(buf.device, torch.float32)
        self.dyn_frozen[idx] = True
        self.dyn_decomp[idx] = True
        self.dyn_alpha[idx, :T] = a32.to(buf.dtype)
        self.dyn_rho[idx, :T] = r32.to(buf.dtype)
        self.dyn_z[idx, :T] = z32.to(buf.dtype)
        if self.dyn_rho_band is None:
            self.dyn_rho_band = torch.zeros(
                self.n_head, BAND_ROWS, self.dyn_rho.size(1),
                device=buf.device, dtype=buf.dtype)
        # one batched band build for the whole call (from the STORED, rounded
        # rho so the kernel and the eager oracle see identical values)
        self.dyn_rho_band[idx] = make_rho_band(self.dyn_rho[idx])
        self._fz_dirty = True

    def unfreeze_heads(self, head_idx):
        """Un-freeze heads (used to roll back a freeze that hurt val loss)."""
        for h in head_idx:
            self.dyn_frozen[h] = False
            if self.prior_repr == "decomposed":
                self.dyn_decomp[h] = False
        self._fz_dirty = True

    @torch.no_grad()
    def refresh_dynamic_derived_state(self, *, rebuild_rho_band=False):
        """Refresh caches after externally synchronizing dynamic buffers.

        DDP broadcasts the compact alpha/rho/z representation. The aligned rho
        band is deterministic but much larger, so destination ranks rebuild it
        locally rather than transferring it from rank 0.
        """
        if not self.dynamic:
            return
        if self.prior_repr == "decomposed" and rebuild_rho_band:
            idx = (self.dyn_decomp & self.dyn_frozen).nonzero(as_tuple=True)[0]
            if idx.numel():
                if self.dyn_rho_band is None:
                    self.dyn_rho_band = torch.zeros(
                        self.n_head, BAND_ROWS, self.dyn_rho.size(1),
                        device=self.dyn_rho.device, dtype=self.dyn_rho.dtype)
                self.dyn_rho_band[idx] = make_rho_band(self.dyn_rho[idx])
        self._fz_dirty = True
        self._all_patt = None
        self._refresh_freeze_cache()

    def _decomp_patterns(self, idx, T):
        """Materialize (len(idx), T, T) decomposed patterns
        exp(alpha[j] + rho[i-j] - z[i]) — only the eager oracle / capture path
        and the non-fused fallback need a dense matrix; the fused kernel
        rebuilds rows in registers."""
        lg = decomp_logits(self.dyn_alpha[idx, :T].float(),
                           self.dyn_rho[idx, :T].float())
        return torch.exp(lg - self.dyn_z[idx, :T].float()[:, :, None])

    def _frozen_patterns(self, T):
        """(n_head, T, T) fixed patterns of ALL heads (rows of live heads are
        meaningless — callers mask on dyn_frozen). Cached until the freeze set
        changes: the capture path calls this on EVERY measurement forward, and
        the patterns only move when heads (un)freeze."""
        if self._all_patt is not None and self._all_patt[0] == T:
            return self._all_patt[1]
        if self.dyn_pattern is not None:
            fp = self.dyn_pattern[:, :T, :T]
            if self._dec_any:
                fp = fp.clone()
        else:                           # decomposed mode, nothing dense-frozen
            fp = torch.zeros(self.n_head, T, T, device=self.dyn_frozen.device)
        if self._dec_any:
            fp[self.dyn_decomp] = self._decomp_patterns(self.dyn_decomp, T) \
                .to(fp.dtype)
        self._all_patt = (T, fp)
        return fp

    def _attend_dynamic(self, q, k, v):
        """Eager attention that overrides frozen heads with their stored pattern.
        Unfrozen heads use normal softmax(QK^T); frozen heads use dyn_pattern @ V
        (no gradient to their Q/K). Optionally captures attn for variance stats."""
        B, nh, T, hd = q.shape
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))  # raw logits (pre-mask)
        if self._capture and self._capture_logits:
            self._last_logits = scores.detach()   # for the gaussian (sampled) prior
        causal = self._dyn_causal[:T, :T]
        att = scores.masked_fill(~causal, float('-inf'))
        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        if self._fz_dirty:
            self._refresh_freeze_cache()
        if self._n_frozen:
            fp = self._frozen_patterns(T).to(att.dtype).unsqueeze(0)  # (1, nh, T, T)
            m = self.dyn_frozen.view(1, nh, 1, 1)
            att = torch.where(m, fp.expand_as(att), att)
        if self._capture and self._retain_attention:
            self._last_attn = att.detach()
        att = self.attn_dropout(att)
        return att @ v

    def _refresh_freeze_cache(self):
        """Recompute freeze indices/columns/pattern once per freeze-set change, so the
        hot path has NO GPU->CPU syncs (no .sum()/.nonzero() per forward)."""
        frozen = self.dyn_frozen
        hd = self.head_dim
        self._fz_idx = frozen.nonzero(as_tuple=True)[0]
        self._uf_idx = (~frozen).nonzero(as_tuple=True)[0]
        self._n_frozen = int(self._fz_idx.numel())       # python int (one sync, on change)
        ar = torch.arange(hd, device=frozen.device)
        if self._uf_idx.numel() > 0:
            self._uf_cols = (self._uf_idx[:, None] * hd + ar).reshape(-1)
        else:
            self._uf_cols = None
        if self._n_frozen > 0:
            self._fz_cols = (self._fz_idx[:, None] * hd + ar).reshape(-1)
            # cheap-path feature order is [unfrozen heads..., frozen heads...];
            # c_proj input columns are permuted to match (weights are ~1000x smaller
            # than the activations, so we permute weights, never activations)
            self._cperm_cols = self._fz_cols if self._uf_cols is None \
                else torch.cat([self._uf_cols, self._fz_cols])
        else:
            self._fz_cols = None
            self._cperm_cols = None
        # int32 head indices / per-head state for the Triton kernels:
        # 0 = live, 1 = frozen on the dense pattern, 2 = frozen decomposed
        self._fz_idx_i32 = self._fz_idx.to(torch.int32)
        if self.prior_repr == "decomposed":
            dec = self.dyn_decomp & frozen
            self._fz_mask_i32 = frozen.to(torch.int32) + dec.to(torch.int32)
            self._dec_any = bool(dec.any())          # one sync, on change only
            self._dec_args = ((self.dyn_alpha, self.dyn_rho, self.dyn_z,
                               self.dyn_rho_band)
                              if self._dec_any else (None, None, None, None))
        else:
            self._fz_mask_i32 = frozen.to(torch.int32)
            self._dec_any = False
            self._dec_args = (None, None, None, None)
        # head -> slot in the compact (unfrozen-only) Q/K of the fused kernel
        hidx = torch.zeros(frozen.numel(), dtype=torch.int32, device=frozen.device)
        if self._uf_idx.numel() > 0:
            hidx[self._uf_idx] = torch.arange(self._uf_idx.numel(),
                                              dtype=torch.int32, device=frozen.device)
        self._hidx_i32 = hidx
        # identity map, for the full-Q/K variant used on tiny (launch-bound) workloads
        self._hidx_id_i32 = torch.arange(frozen.numel(), dtype=torch.int32,
                                         device=frozen.device)
        # gathered-pattern cache: only the dense torch fallback needs it (lazy)
        self._fz_patt = None
        # all-heads materialized-pattern cache (eager capture / fallback paths)
        self._all_patt = None
        # inference-only gathered-weight cache (invalidated by weight version bumps)
        self._wc_key = None
        self._fz_dirty = False

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # dyn_pattern (decomposed mode) and dyn_rho_band are LAZY buffers
        # (registered as None until first use). A checkpoint saved after heads
        # were frozen contains them; materialize before the parent load so
        # load_state_dict does not report them as unexpected keys.
        if getattr(self, "dynamic", False) and self.prior_repr == "decomposed":
            for name in ("dyn_pattern", "dyn_rho_band"):
                key = prefix + name
                if key in state_dict and getattr(self, name) is None:
                    setattr(self, name, torch.zeros_like(
                        state_dict[key], device=self.dyn_frozen.device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def train(self, mode=True):
        # Fused AdamW can update tensors without bumping their version counters.
        # Each explicit train/eval boundary starts a new inference cache lifetime;
        # repeated batches within an evaluation still share the gathered weights.
        self._wc_key = None
        return super().train(mode)

    def _infer_weight_cache(self):
        """Pre-gathered projection weights for the no-grad cheap path. Weights are
        static during inference, so the per-step row/column gathers of the training
        path are wasted there; rebuild only when a weight is updated in place
        (where version counters detect it), at a train/eval boundary, or
        when the freeze set changes. Fused optimisers need the boundary reset."""
        qw, kw, vw, cw = (self.q_proj.weight, self.k_proj.weight,
                          self.v_proj.weight, self.c_proj.weight)
        key = (qw._version, kw._version, vw._version, cw._version)
        if self._wc_key != key:
            uf, perm = self._uf_cols, self._cperm_cols
            self._wc_v = vw.index_select(0, perm).detach()
            self._wc_vb = None if self.v_proj.bias is None \
                else self.v_proj.bias.index_select(0, perm).detach()
            if uf is not None:
                self._wc_qk = torch.cat([qw.index_select(0, uf),
                                         kw.index_select(0, uf)]).detach()
                self._wc_qkb = None if self.q_proj.bias is None else \
                    torch.cat([self.q_proj.bias.index_select(0, uf),
                               self.k_proj.bias.index_select(0, uf)]).detach()
            else:
                self._wc_qk = self._wc_qkb = None
            self._wc_c = cw.index_select(1, perm).detach()
            self._wc_key = key
        return self._wc_v, self._wc_vb, self._wc_qk, self._wc_qkb, self._wc_c

    def _attend_cheap(self, x, rope=None):
        """Cheap dynamic path: frozen heads SKIP Q/K (projections + QK^T) and use
        their stored pattern @ V; unfrozen heads use SDPA. This is where the actual
        train/infer FLOP saving comes from. Returns the ATTENTION OUTPUT AFTER
        c_proj, (B, T, n_embd): the per-head work is laid out [unfrozen..., frozen...]
        so every slice is a free view, and c_proj's input columns are permuted to
        match. No activation-sized gather/scatter runs on this path (weights are
        permuted instead; they are ~1000x smaller).
        Numerically matches the eager override path (up to SDPA vs eager softmax)."""
        if self._fz_dirty:
            self._refresh_freeze_cache()
        B, T, C = x.size()
        nh, hd = self.n_head, self.head_dim
        if self._n_frozen == 0:
            # nothing frozen yet -> plain full attention (no slicing overhead)
            q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
            k = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
            v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
            q, k = self._apply_rope(q, k, rope)
            y = self._attend(q, k, v).transpose(1, 2).contiguous().view(B, T, nh * hd)
            return self.c_proj(y)

        nf = self._n_frozen
        nuf = nh - nf

        # The fused kernel implements no attention dropout: with dropout>0 in
        # training, fall back to the split path (its SDPA half applies dropout).
        if _FUSED and self.attn_mask_rate == 0 and fused_attn_usable(x) \
                and (self.dropout == 0.0 or not self.training):
            global _FUSED_DISPATCH_COUNT
            _FUSED_DISPATCH_COUNT += 1
            # ONE kernel for all heads, in NATURAL head order: no head split, no
            # cat, no c_proj weight permutation. Q/K are projected only for the
            # unfrozen heads (one fused row-sliced GEMM), so frozen heads have no
            # Q/K slot at all and cannot receive Q/K gradient.
            v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
            hidx = self._hidx_i32
            if nuf == 0:                # every head frozen: no Q/K at all
                qu = ku = v[:, :0]
            elif not torch.is_grad_enabled():
                # inference: weights are static -> use the cached pre-gathered QK
                # weight (no per-step index_select/cat launches)
                _, _, wqk, wqkb, _ = self._infer_weight_cache()
                qk = F.linear(x, wqk, wqkb).view(B, T, 2, nuf, hd)
                qu, ku = qk[:, :, 0].transpose(1, 2), qk[:, :, 1].transpose(1, 2)
            elif B * T <= 1024 and C <= 1024:
                # truly tiny, launch-bound workloads (small batch*seq AND narrow
                # model): two plain full-head GEMMs beat a row-sliced GEMM plus
                # gather/cat launches. On wider models the wasted frozen-head QK
                # FLOPs (~C^2) outgrow the saved launches, so they stay compact.
                # Frozen heads' Q/K are computed but never read (identity head
                # map), and their projection rows still get exactly zero gradient
                # (dq/dk slots for frozen heads are never written by the kernel).
                qu = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
                ku = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
                hidx = self._hidx_id_i32
            else:
                qk = linear_rows2(x, self.q_proj.weight, self.k_proj.weight,
                                  self.q_proj.bias, self.k_proj.bias, self._uf_cols) \
                    .view(B, T, 2, nuf, hd)
                qu, ku = qk[:, :, 0].transpose(1, 2), qk[:, :, 1].transpose(1, 2)
            if nuf > 0:
                qu, ku = self._apply_rope(qu, ku, rope)
            al, rh, zz, rb = self._dec_args  # (None,)*4 unless decomposed heads
            y = fused_mixed_attn(qu, ku, v,
                                 self.dyn_pattern, self._fz_mask_i32, hidx,
                                 alpha=al, rho=rh, z=zz, rho_band=rb,
                                 validate=False)
            y = y.transpose(1, 2).reshape(B, T, nh * hd)
            return self.c_proj(y)

        infer = not torch.is_grad_enabled()
        if infer:
            wv, wvb, wqk, wqkb, wc = self._infer_weight_cache()
        # V for ALL heads in one standard-shape GEMM, rows permuted to
        # [unfrozen..., frozen...] so both groups are free views
        if infer:
            v_all = F.linear(x, wv, wvb).view(B, T, nh, hd).transpose(1, 2)
        else:
            v_all = linear_rows(x, self.v_proj.weight, self.v_proj.bias, self._cperm_cols) \
                .view(B, T, nh, hd).transpose(1, 2)
        vf = v_all[:, nuf:]
        if self._dec_any:
            # decomposed heads on the (rare) non-fused fallback: use the cached
            # materialization (rebuilt only when the freeze set changes)
            patt = self._frozen_patterns(T)[self._fz_idx]
            yf = torch.matmul(patt.to(vf.dtype).unsqueeze(0), vf)
        elif frozen_pv_usable(vf):
            # causal-tiled Triton GEMM reading dyn_pattern in place (no per-step
            # cast/gather copies; backward is dV = P^T @ dY)
            yf = frozen_pv(self.dyn_pattern, vf, self._fz_idx_i32, T)
        else:
            if self._fz_patt is None:
                self._fz_patt = self.dyn_pattern[self._fz_idx]
            patt = self._fz_patt[:, :T, :T].to(vf.dtype).unsqueeze(0)  # (1, nf, T, T)
            yf = torch.matmul(patt, vf)
        if nuf > 0:
            # Q and K ONLY for unfrozen heads, fused into one row-sliced GEMM
            if infer:
                qk = F.linear(x, wqk, wqkb).view(B, T, 2, nuf, hd)
            else:
                qk = linear_rows2(x, self.q_proj.weight, self.k_proj.weight,
                                  self.q_proj.bias, self.k_proj.bias, self._uf_cols) \
                    .view(B, T, 2, nuf, hd)
            qu = qk[:, :, 0].transpose(1, 2)
            ku = qk[:, :, 1].transpose(1, 2)
            qu, ku = self._apply_rope(qu, ku, rope)
            yu = self._attend(qu, ku, v_all[:, :nuf])
            y = torch.cat([yu.transpose(1, 2), yf.transpose(1, 2)], dim=2)
        else:
            y = yf.transpose(1, 2)
        y = y.reshape(B, T, nh * hd)
        # heads are ordered [unfrozen..., frozen...] -> permute c_proj's input columns
        if infer:
            return F.linear(y, wc, self.c_proj.bias)
        return linear_permuted_cols(y, self.c_proj.weight, self.c_proj.bias, self._cperm_cols)

    def _forward_dynamic(self, x, rope=None):
        """Dispatch the dynamic path: eager+capture during variance measurement
        (controller sets _capture), cheap skip-QK path during normal steps.
        Returns the attention output INCLUDING c_proj (the cheap path folds the
        frozen/unfrozen head permutation into c_proj's columns)."""
        B, T, C = x.size()
        nh, hd = self.n_head, self.head_dim
        if self._capture:
            q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
            k = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
            v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
            q, k = self._apply_rope(q, k, rope)
            y = self._attend_dynamic(q, k, v).transpose(1, 2).contiguous().view(B, T, nh * hd)
            return self.c_proj(y)
        return self._attend_cheap(x, rope=rope)

    def _attend(self, q, k, v):
        """Run attention (flash or manual) on a single group of heads."""
        if self.attn_mask_rate > 0:
            # Must compute attention manually to apply randomization
            T = q.size(2)
            Nh = q.size(1)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Causal mask
            causal = torch.tril(torch.ones(T, T, device=att.device, dtype=torch.bool))
            att = att.masked_fill(~causal, float('-inf'))
            att = F.softmax(att, dim=-1)
            # Replace selected positions with random values, then renormalize
            mask = self.attn_rand_mask[:Nh, :T, :T]  # (n_head, T, T)
            rand_weights = torch.rand_like(att) * mask  # random values where mask=1
            att = att * (1 - mask) + rand_weights       # keep learned where mask=0, random where mask=1
            att = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # renormalize rows
            att = self.attn_dropout(att)
            return att @ v
        elif self.flash:
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:q.size(2),:q.size(2)] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            return att @ v

    def forward(self, x, rope=None):
        B, T, C = x.size()

        if self.n_pattern_heads > 0:
            # Hybrid mode (Method H): pattern heads (+ optional standard heads)
            # Pattern heads: V only → fixed attention pattern
            vp = self.v_proj_pattern(x).view(B, T, self.n_pattern_heads, self.head_dim).transpose(1, 2)
            yp = self.head_pattern_override(vp)
            yp = yp.transpose(1, 2).contiguous().view(B, T, self.n_pattern_heads * self.head_dim)

            if self.n_standard_heads > 0:
                # Standard heads: Q/K/V → learned attention
                qs = self.q_proj(x).view(B, T, self.n_standard_heads, self.head_dim).transpose(1, 2)
                ks = self.k_proj(x).view(B, T, self.n_standard_heads, self.head_dim).transpose(1, 2)
                vs = self.v_proj(x).view(B, T, self.n_standard_heads, self.head_dim).transpose(1, 2)
                qs, ks = self._apply_rope(qs, ks, rope)
                ys = self._attend(qs, ks, vs)
                ys = ys.transpose(1, 2).contiguous().view(B, T, self.n_standard_heads * self.head_dim)
                y = torch.cat([yp, ys], dim=-1)  # (B, T, n_embd)
            else:
                y = yp  # 100% pattern heads

        elif self.n_random_heads > 0 and self.n_learned_heads > 0 and self.has_scaled_random:
            # Method A with scaled projection: different head_dim, compute separately
            qr = self.q_proj_rand(x).view(B, T, self.n_random_heads, self.rand_head_dim).transpose(1, 2)
            kr = self.k_proj_rand(x).view(B, T, self.n_random_heads, self.rand_head_dim).transpose(1, 2)
            vr = self.v_proj_rand(x).view(B, T, self.n_random_heads, self.rand_head_dim).transpose(1, 2)
            qr, kr = self._apply_rope(qr, kr, rope)
            yr = self._attend(qr, kr, vr)
            yr = yr.transpose(1, 2).contiguous().view(B, T, self.n_random_heads * self.rand_head_dim)

            ql = self.q_proj(x).view(B, T, self.n_learned_heads, self.head_dim).transpose(1, 2)
            kl = self.k_proj(x).view(B, T, self.n_learned_heads, self.head_dim).transpose(1, 2)
            vl = self.v_proj(x).view(B, T, self.n_learned_heads, self.head_dim).transpose(1, 2)
            ql, kl = self._apply_rope(ql, kl, rope)
            yl = self._attend(ql, kl, vl)
            yl = yl.transpose(1, 2).contiguous().view(B, T, self.n_learned_heads * self.head_dim)

            y = torch.cat([yr, yl], dim=-1)

        elif self.n_random_heads > 0 and self.n_learned_heads > 0:
            # Method A: same head_dim, compute together
            q = torch.cat([self.q_proj_rand(x), self.q_proj(x)], dim=-1)
            k = torch.cat([self.k_proj_rand(x), self.k_proj(x)], dim=-1)
            v = torch.cat([self.v_proj_rand(x), self.v_proj(x)], dim=-1)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, k, rope)

            if self.pattern_override is not None:
                y = self.pattern_override(v) if self.pattern_override.skips_qk else self.pattern_override(v, q=q, k=k)
            else:
                y = self._attend(q, k, v)
            y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)

        elif getattr(self, "dynamic", False):
            # Dynamic freezing: cheap skip-QK path (eager+capture only when measuring).
            # Projections INCLUDING c_proj are done inside (the cheap path folds the
            # frozen/unfrozen head permutation into c_proj's input columns).
            return self.resid_dropout(self._forward_dynamic(x, rope=rope))

        else:
            # Standard: all heads same type
            q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, k, rope)

            if self.pattern_override is not None:
                y = self.pattern_override(v) if self.pattern_override.skips_qk else self.pattern_override(v, q=q, k=k)
            else:
                y = self._attend(q, k, v)
            y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, config, layer_idx=0, rng=None):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.main_block = CausalSelfAttention(config, layer_idx=layer_idx)
        self.main_block = apply_randomization(self.main_block, config, layer_idx, rng)

        if config.state_size:
            self.main_block = StatefulBlock(self.main_block, config)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def reset_state(self):
        if hasattr(self.main_block, "reset_state"):
            self.main_block.reset_state()

    def is_masked(self):
        if hasattr(self.main_block, "is_masked"):
            return self.main_block.is_masked()
        return True

    def forward(self, x, rope=None):
        x = x + self.main_block(self.ln_1(x), rope=rope)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# GPTConfig
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    state_size: int = 0
    shared_weights: bool = False
    position_encoding: str = "learned_absolute"  # "learned_absolute" | "rope"
    rope_base: float = 10000.0

    # Randomization controls
    rand_head_rate: float = 0.0       # A: fraction of random heads
    rand_proj_scale: float = 1.0      # A: scale factor for random head projection dim
    rand_layer_rate: float = 0.0      # B: fraction of random layers
    rand_layer_mode: str = "evenly_spaced"  # B: "evenly_spaced"|"first"|"last"|"random"|"explicit"
    rand_layer_ids: str = ""          # B: comma-separated layer indices for "explicit" mode
    rand_components: str = "none"     # C/D: which projections to randomize
    rand_proj_mask_rate: float = 1.0       # C/D: fraction of weight entries frozen (1.0 = full freeze)
    rand_attn_mask_rate: float = 0.0   # E: fraction of attention weights randomly masked (zeroed + renorm)
    rand_attn_head_rate: float = 0.0  # H: fraction of heads using fixed attention pattern (hybrid)
    rand_attn_head_rate_per_layer: str = ""  # H: optional per-layer rates as comma-separated string, overrides uniform rate
    rand_attn_pattern: str = "none"   # F/H: attention pattern type (for global override or hybrid heads)
    rand_attn_pattern_sharing: str = "none"  # H sharing: "none"|"per_layer"|"global" — share pattern memory
    rand_attn_window: int = 32        # F: window size for "local_window"
    rand_attn_prior_path: str = ""    # F: path to .pt file for "prior" pattern
    rand_attn_dynamic: bool = False   # Version A: per-head freezing decided dynamically
    rand_attn_prior_repr: str = "dense"  # A: frozen-prior storage — "dense" (T x T matrix)
                                      # | "decomposed" (alpha[j] + rho[i-j] logit vectors, O(T)/head)
                                      # during training (variance-threshold driven)
    freeze_pattern_dtype: str = "float32"  # dense dynamic-prior storage; bf16 saves 2x memory
    rand_seed: int = 42              # deterministic seed for all randomization

    # Backward compatibility — mapped internally
    random_proj: bool = False
    block_type: str = "selfattention"

    def __post_init__(self):
        # Backward compatibility: map old flags to new framework
        if self.random_proj and self.rand_components == "none":
            self.rand_components = "all"
        if self.position_encoding not in ("learned_absolute", "rope"):
            raise ValueError(
                "position_encoding must be 'learned_absolute' or 'rope'")
        if self.freeze_pattern_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                "freeze_pattern_dtype must be 'float32' or 'bfloat16'")
        if self.position_encoding == "rope":
            if self.state_size:
                raise ValueError("RoPE is not defined for the stateful path yet")
            if (self.n_embd // self.n_head) % 2:
                raise ValueError("RoPE requires an even attention head dimension")
            if self.rand_head_rate > 0 and self.rand_proj_scale != 1.0:
                raise ValueError(
                    "RoPE does not support scaled random-projection head dimensions")
        # block_type is ignored (only selfattention supported now)


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        rng = torch.Generator().manual_seed(config.rand_seed)

        # Reset the shared-pattern cache so this model gets fresh shared tensors
        # (otherwise patterns would leak between models constructed in the same process)
        AttentionPatternOverride.reset_shared_cache()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx=i, rng=rng) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.rotary = (RotaryEmbedding(
            config.n_embd // config.n_head, config.block_size, config.rope_base)
            if config.position_encoding == "rope" else None)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        if config.position_encoding == "rope":
            # Keep the unused table so RoPE/absolute models consume the same RNG
            # during initialization and old checkpoints retain strict state shape.
            self.transformer.wpe.weight.requires_grad_(False)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_state(self):
        for block in self.transformer.h:
            block.reset_state()

    def is_masked(self):
        return all([b.is_masked() for b in self.transformer.h])

    def prepare_inputs(self, idx):
        """Embed tokens once and build the shared per-forward RoPE cache."""
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        tok_emb = self.transformer.wte(idx)
        if self.config.position_encoding == "learned_absolute":
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = tok_emb + self.transformer.wpe(pos)
            rope = None
        else:
            rope_dtype = tok_emb.dtype
            if device.type == "cuda" and torch.is_autocast_enabled():
                rope_dtype = torch.get_autocast_gpu_dtype()
            rope = self.rotary(t, rope_dtype)
        return self.transformer.drop(tok_emb), rope

    def forward_hidden(self, idx):
        x, rope = self.prepare_inputs(idx)
        for block in self.transformer.h:
            x = block(x, rope=rope)
        return self.transformer.ln_f(x)

    def forward_all_positions(self, idx):
        """Return next-token logits at every input position."""
        return self.lm_head(self.forward_hidden(idx))

    def forward(self, idx, targets=None):
        x = self.forward_hidden(idx)

        if targets is not None:
            if self.is_masked():
                logits = self.lm_head(x)
            else:
                logits = self.lm_head(x[:, -1:, :])
                targets = targets[:, -1:]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Keep the sequence dimension without advanced indexing. The
            # equivalent slice is compatible with CUDA Graph capture.
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        wpe_requires_grad = self.transformer.wpe.weight.requires_grad
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size],
            requires_grad=wpe_requires_grad)
        for block in self.transformer.h:
            if hasattr(block.main_block, 'bias'):
                block.main_block.bias = block.main_block.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # HuggingFace uses fused c_attn — we must split into q_proj/k_proj/v_proj
        # Also transpose Conv1D weights
        for k_hf in sd_keys_hf:
            # Handle the fused QKV projection
            if 'attn.c_attn.weight' in k_hf:
                # Conv1D: shape (n_embd, 3*n_embd) -> need transpose then split
                w = sd_hf[k_hf].t()  # (3*n_embd, n_embd)
                n_embd = w.shape[1]
                wq, wk, wv = w.split(n_embd, dim=0)
                prefix = k_hf.replace('attn.c_attn.weight', '')
                # Map transformer.h.X. -> transformer.h.X.main_block.
                prefix = prefix.replace('.attn.c_attn.weight', '')
                layer_prefix = k_hf.split('.attn.')[0] + '.main_block.'
                with torch.no_grad():
                    sd[layer_prefix + 'q_proj.weight'].copy_(wq)
                    sd[layer_prefix + 'k_proj.weight'].copy_(wk)
                    sd[layer_prefix + 'v_proj.weight'].copy_(wv)
            elif 'attn.c_attn.bias' in k_hf:
                b = sd_hf[k_hf]
                n_embd = b.shape[0] // 3
                bq, bk, bv = b.split(n_embd, dim=0)
                layer_prefix = k_hf.split('.attn.')[0] + '.main_block.'
                with torch.no_grad():
                    sd[layer_prefix + 'q_proj.bias'].copy_(bq)
                    sd[layer_prefix + 'k_proj.bias'].copy_(bk)
                    sd[layer_prefix + 'v_proj.bias'].copy_(bv)
            elif 'attn.c_proj' in k_hf:
                our_key = k_hf.replace('.attn.', '.main_block.')
                if 'weight' in k_hf:
                    with torch.no_grad():
                        sd[our_key].copy_(sd_hf[k_hf].t())
                else:
                    with torch.no_grad():
                        sd[our_key].copy_(sd_hf[k_hf])
            elif 'mlp.c_fc.weight' in k_hf or 'mlp.c_proj.weight' in k_hf:
                our_key = k_hf
                with torch.no_grad():
                    sd[our_key].copy_(sd_hf[k_hf].t())
            else:
                our_key = k_hf
                if our_key in sd:
                    with torch.no_grad():
                        sd[our_key].copy_(sd_hf[k_hf])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def flops_per_iter(self, fwdbwd_per_iter):
        """Return (actual_flops, baseline_flops) per training iteration.

        Pattern heads skip Q proj + K proj + softmax(QK^T), saving ~3/4 of the per-head
        attention compute. The 12*L*H*Q*T term is the standard transformer attention cost;
        we replace H with (H_standard + H_pattern * pattern_factor) where pattern_factor
        accounts for the V-only + pattern@V matmul (about 3/12 of full attention).
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size

        # Per-layer rate may differ; sum the pattern-head count across all layers
        per_layer = getattr(cfg, 'rand_attn_head_rate_per_layer', '')
        if per_layer:
            if isinstance(per_layer, str):
                rates = [float(x) for x in per_layer.split(',')]
            else:
                rates = [float(x) for x in per_layer]
            total_pattern_heads = sum(int(r * H) for r in rates)
        else:
            uniform_rate = getattr(cfg, 'rand_attn_head_rate', 0.0)
            total_pattern_heads = L * int(uniform_rate * H)

        total_standard_heads = L * H - total_pattern_heads
        pattern_factor = 3.0 / 12.0  # pattern head: just V proj + pattern@V

        attn_flops_token_actual = 12 * Q * T * (total_standard_heads + total_pattern_heads * pattern_factor)
        attn_flops_token_baseline = 12 * L * H * Q * T

        flops_per_token_actual = 6 * N + attn_flops_token_actual
        flops_per_token_baseline = 6 * N + attn_flops_token_baseline
        return (flops_per_token_actual * T * fwdbwd_per_iter,
                flops_per_token_baseline * T * fwdbwd_per_iter)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        flops, _ = self.flops_per_iter(fwdbwd_per_iter)
        flops_achieved = flops * (1.0 / dt)
        flops_promised = 312e12  # A100 bf16 peak
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
