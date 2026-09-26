"""
Hybrid prior-attention kernel for pretrained HuggingFace models.

For pattern heads we skip Q-projection, QK^T, and softmax entirely — only
V-projection and a single `pattern @ V` matmul remain. For standard heads,
the original attention forward is used unchanged.

Design notes:
- Per-query-head granularity (the K and V projections are kept full so any
  subset of query heads can be marked as pattern heads, even with GQA where
  multiple query heads share a KV head).
- Subclass the original attention class dynamically — this preserves all the
  model-specific glue (RoPE, q/k norms, KV cache, attention mask handling,
  etc.) without re-implementing it.
- Designed to coexist with the existing eager-override path (which is the
  default). Activate via `convert_targets_to_hybrid(...)` after extraction.

Architecture support layered as:
  1. `HybridAttentionBase` — shared logic (shrunken q_proj, prior buffer,
     output assembly). Subclass for each model family.
  2. `HYBRID_REGISTRY` — maps original attention class name -> hybrid factory.
  3. `convert_targets_to_hybrid(...)` — top-level entry point that walks the
     model, swaps each target's attention module for its hybrid counterpart,
     and returns a handle that can restore the originals.

Extending to a new model family (e.g. LLaMA):
  - Implement `make_<family>_hybrid_class()` returning a class that
    inherits from `(HybridAttentionBase, OriginalAttentionClass)` and
    overrides `forward()`. The forward only needs to:
        1. Compute K, V for all heads (using the original projections)
        2. Compute Q only for standard heads (using self.q_proj_std)
        3. Call the original eager_attention_forward for standard heads
        4. Compute pattern @ V for pattern heads
        5. Concat in the original head ordering, apply o_proj
  - Register it in `_REGISTRY_BUILDERS`.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Optional, Type

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from kernel import fused_attn_usable, fused_mixed_attn
from kernel.fused_attn import decomposed_logz, make_rho_band, decomp_logits


# ---------------------------------------------------------------------------
# Base mixin — shared shrunken-q_proj + pattern-buffer setup
# ---------------------------------------------------------------------------

class HybridAttentionBase:
    """Shared infrastructure for hybrid attention modules.

    Concrete model-family subclasses inherit from
    `(HybridAttentionBase, OriginalAttentionClass)` and implement `forward`.
    This mixin provides:
        - shrunken q_proj that projects only to standard query heads
        - prior pattern buffer + per-(layer,head) selection indices
        - debug counter

    It does NOT implement forward — that's family-specific.
    """

    ORIGINAL_CLASS: Optional[type] = None  # set by subclasses

    def _hybrid_init(self, original: nn.Module,
                     pattern: torch.Tensor,
                     head_mask: torch.Tensor,
                     n_q: int, head_dim: int,
                     device, dtype,
                     q_per_head_dim: int = None,
                     decomp=None):
        """Shared shrunken-q_proj + prior-buffer setup.

        Args:
            n_q: number of query heads in the original module.
            head_dim: per-head dimension.
            q_per_head_dim: how many output features per head in the original
                q_proj. For most models this is just `head_dim`. For Qwen3.5
                it's `head_dim * 2` because q_proj also produces a gate.
            decomp: optional (alpha, rho) pair of (n_q, S) fp32 tensors — the
                DECOMPOSED prior softmax(alpha[j] + rho[i-j]) for the pattern
                heads (rows of live heads are ignored). When given, no dense
                (T, T) pattern is stored at all: the kernel rebuilds tiles in
                registers (frozen state 2) and `pattern` may be None.
        """
        if q_per_head_dim is None:
            q_per_head_dim = head_dim
        n_pattern = int(head_mask.sum().item())
        n_standard = n_q - n_pattern
        self.decomposed = decomp is not None

        self.n_q_heads = n_q
        self.n_pattern_heads = n_pattern
        self.n_standard_heads = n_standard
        self.head_dim_hybrid = head_dim

        # ---- shrunken q_proj for the standard heads only --------------------
        if n_standard > 0:
            std_mask = ~head_mask
            row_keep = std_mask.repeat_interleave(q_per_head_dim)
            q_orig = original.q_proj
            q_small = nn.Linear(q_orig.in_features,
                                n_standard * q_per_head_dim,
                                bias=q_orig.bias is not None,
                                device=device, dtype=dtype)
            with torch.no_grad():
                q_small.weight.copy_(q_orig.weight[row_keep])
                if q_orig.bias is not None:
                    q_small.bias.copy_(q_orig.bias[row_keep])
            self.q_proj_std = q_small
        else:
            self.q_proj_std = None

        # ---- pattern selection indices + frozen pattern buffer --------------
        pat_idx = torch.nonzero(head_mask, as_tuple=True)[0]
        std_idx = torch.nonzero(~head_mask, as_tuple=True)[0]
        self.register_buffer("pattern_head_idx", pat_idx.to(device))
        self.register_buffer("standard_head_idx", std_idx.to(device))
        if self.decomposed:
            # O(S) per head: alpha / rho / z (row normalizers) + the aligned
            # rho band the kernel reads. Kept in fp32 regardless of the model
            # dtype (the normalizer must not accumulate in bf16). Rows of live
            # heads are never read by the kernel.
            alpha, rho = decomp
            alpha = alpha.detach().to(device=device, dtype=torch.float32)
            rho = rho.detach().to(device=device, dtype=torch.float32)
            z = decomposed_logz(alpha, rho)
            self.register_buffer("dec_alpha", alpha.contiguous())
            self.register_buffer("dec_rho", rho.contiguous())
            self.register_buffer("dec_z", z.contiguous())
            self.register_buffer("dec_rho_band", make_rho_band(rho))
            self.register_buffer("prior_pattern",
                                 torch.empty(0, device=device, dtype=dtype))
        elif n_pattern > 0:
            pattern_subset = pattern[pat_idx].to(device=device, dtype=dtype)
            # The Triton frozen branch skips masking inside a diagonal tile, so
            # enforce the API invariant here and restore row normalization.
            pattern_subset = pattern_subset.tril()
            pattern_subset = pattern_subset / pattern_subset.sum(
                dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
            self.register_buffer("prior_pattern", pattern_subset.contiguous())
        else:
            self.register_buffer("prior_pattern",
                                 torch.empty(0, device=device, dtype=dtype))

        # Natural-head maps for the fused mixed-head Triton kernel. Q/K and P
        # are compact, while V/output retain all query heads in natural order.
        # Head state: 0 live, 1 frozen on the dense pattern, 2 frozen decomposed.
        frozen_i32 = head_mask.to(device=device, dtype=torch.int32)
        if self.decomposed:
            frozen_i32 = frozen_i32 * 2
        hidx_i32 = torch.zeros(n_q, device=device, dtype=torch.int32)
        pidx_i32 = torch.zeros(n_q, device=device, dtype=torch.int32)
        if n_standard:
            hidx_i32[std_idx.to(device)] = torch.arange(
                n_standard, device=device, dtype=torch.int32)
        if n_pattern:
            pidx_i32[pat_idx.to(device)] = torch.arange(
                n_pattern, device=device, dtype=torch.int32)
        self.register_buffer("frozen_mask_i32", frozen_i32)
        self.register_buffer("hidx_i32", hidx_i32)
        self.register_buffer("pidx_i32", pidx_i32)

        self.hybrid_calls = 0
        self.triton_calls = 0

    # ---- representation-agnostic helpers used by every family's forward ----

    def _fused_prior_kwargs(self):
        """Keyword args that hand the frozen prior to `fused_mixed_attn`."""
        if self.decomposed:
            return dict(patt=None, alpha=self.dec_alpha, rho=self.dec_rho,
                        z=self.dec_z, rho_band=self.dec_rho_band)
        return dict(patt=self.prior_pattern)

    def _fused_prior_ready(self, T):
        """Can the fused kernel serve this call's sequence length?"""
        if self.decomposed:
            return self.dec_alpha.size(-1) >= T
        return self.prior_pattern.ndim == 3 and self.prior_pattern.size(-1) >= T

    def _split_patterns(self, T, dtype):
        """(n_pattern, T, T) dense patterns for the torch fallback / oracle —
        materialized transiently from alpha/rho/z in the decomposed case."""
        if not self.decomposed:
            return self.prior_pattern[:, :T, :T].to(dtype)
        idx = self.pattern_head_idx
        lg = decomp_logits(self.dec_alpha[idx, :T], self.dec_rho[idx, :T])
        return torch.exp(lg - self.dec_z[idx, :T][:, :, None]).to(dtype)


# ---------------------------------------------------------------------------
# Qwen3.5 implementation
# ---------------------------------------------------------------------------

def make_qwen35_hybrid_class():
    """Build the Qwen3.5 hybrid class lazily.

    Qwen3.5-specific quirks the implementation has to handle:
      * `num_attention_heads` / `num_key_value_heads` live on `config`, not on
        the module directly.
      * `q_proj` projects to `2 * num_attention_heads * head_dim` because it
        also produces a per-token gate that is sigmoid-applied to the
        attention output (Qwen3.5's gated attention). The shrunken q_proj
        must also include that gate slice.
      * Forward signature uses `past_key_values` (plural).
    """
    qwen35 = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    OriginalAttention = qwen35.Qwen3_5Attention
    eager_attention_forward = qwen35.eager_attention_forward
    repeat_kv = qwen35.repeat_kv
    apply_rotary_pos_emb = qwen35.apply_rotary_pos_emb

    class HybridQwen3_5Attention(HybridAttentionBase, OriginalAttention):
        ORIGINAL_CLASS = OriginalAttention

        def __init__(self, original: OriginalAttention,
                     pattern: torch.Tensor, head_mask: torch.Tensor,
                     device, dtype, use_triton=True,
                     assume_causal_prefill=False, decomp=None):
            # Skip OriginalAttention.__init__ (would re-allocate weights);
            # adopt the original's submodules instead.
            nn.Module.__init__(self)
            cfg = original.config
            self.config = cfg
            self.layer_idx = original.layer_idx
            self.attention_dropout = original.attention_dropout
            # Pull head counts from config (NOT from the module — Qwen3.5
            # doesn't expose them as direct attrs)
            self.num_attention_heads = cfg.num_attention_heads
            self.num_key_value_heads = cfg.num_key_value_heads
            self._real_num_kv_groups = original.num_key_value_groups
            # IMPORTANT: we manually call repeat_kv on K, V before slicing them
            # by query-head index. eager_attention_forward will call repeat_kv
            # again using `module.num_key_value_groups`. If we leave that at
            # the real value (4 for Qwen3.5) it would broadcast a second time
            # and the shapes wouldn't match the (already-broadcast-and-sliced)
            # query tensor. Setting this to 1 makes the inner repeat_kv a no-op.
            self.num_key_value_groups = 1
            self.head_dim = original.head_dim
            self.scaling = original.scaling
            self.is_causal = original.is_causal
            self.use_triton_kernel = use_triton
            # Some HF models materialize a standard square causal mask even for
            # unpadded, no-cache prefill. The fused kernel already implements
            # that mask internally. This opt-in must never be used for padding,
            # packed sequences, custom masks, or cache-offset decoding.
            self.assume_causal_prefill = assume_causal_prefill

            # Adopt full k_proj / v_proj / o_proj / q_norm / k_norm
            self.k_proj = original.k_proj
            self.v_proj = original.v_proj
            self.o_proj = original.o_proj
            self.q_norm = original.q_norm
            self.k_norm = original.k_norm

            # Build shrunken q_proj. Note q_per_head_dim = 2 * head_dim because
            # the q_proj output is (query, gate) per head.
            self._hybrid_init(
                original, pattern, head_mask,
                n_q=self.num_attention_heads,
                head_dim=self.head_dim,
                device=device, dtype=dtype,
                q_per_head_dim=self.head_dim * 2,
                decomp=decomp,
            )

            # Qwen3.5-specific: pattern heads still need their *gate* even though
            # they skip the query (sigmoid(gate) is applied to every head's
            # attention output in the original forward — skipping it makes the
            # pattern heads' contribution ~2x too large and blows up perplexity).
            # Build a gate-only projection by copying the gate-slice rows of the
            # original q_proj for pattern heads.
            n_pat = int(head_mask.sum().item())
            if n_pat > 0:
                pat_idx_cpu = torch.nonzero(head_mask, as_tuple=True)[0]
                # Gate rows in original q_proj.weight (shape [n_q * 2 * head_dim, hidden]):
                # for head i, the rows [i*2*head_dim + head_dim, (i+1)*2*head_dim) hold the gate
                row_indices = []
                for i in pat_idx_cpu.tolist():
                    start = i * 2 * self.head_dim + self.head_dim
                    row_indices.extend(range(start, start + self.head_dim))
                row_indices = torch.tensor(row_indices, dtype=torch.long)
                q_orig = original.q_proj
                gate_proj = nn.Linear(q_orig.in_features,
                                      n_pat * self.head_dim,
                                      bias=q_orig.bias is not None,
                                      device=device, dtype=dtype)
                with torch.no_grad():
                    gate_proj.weight.copy_(q_orig.weight[row_indices])
                    if q_orig.bias is not None:
                        gate_proj.bias.copy_(q_orig.bias[row_indices])
                self.gate_proj_pat = gate_proj
            else:
                self.gate_proj_pat = None

            # Fuse the compact standard-head (query, gate) rows and the
            # pattern-head gate-only rows into ONE projection. Two separate
            # GEMMs can erase the small Q-row saving at realistic freeze rates.
            std_width = self.n_standard_heads * 2 * self.head_dim
            pat_width = self.n_pattern_heads * self.head_dim
            q_orig = original.q_proj
            qg_proj = nn.Linear(
                q_orig.in_features, std_width + pat_width,
                bias=q_orig.bias is not None, device=device, dtype=dtype,
            )
            with torch.no_grad():
                if std_width:
                    qg_proj.weight[:std_width].copy_(self.q_proj_std.weight)
                    if q_orig.bias is not None:
                        qg_proj.bias[:std_width].copy_(self.q_proj_std.bias)
                if pat_width:
                    qg_proj.weight[std_width:].copy_(self.gate_proj_pat.weight)
                    if q_orig.bias is not None:
                        qg_proj.bias[std_width:].copy_(self.gate_proj_pat.bias)
            self.qg_proj = qg_proj
            self.q_proj_std = None
            self.gate_proj_pat = None

        def forward(self, hidden_states, position_embeddings,
                    attention_mask=None, past_key_values=None,
                    cache_position=None, **kwargs):
            self.hybrid_calls += 1
            B, T, _ = hidden_states.shape
            n_std = self.n_standard_heads
            n_pat = self.n_pattern_heads
            head_dim = self.head_dim
            n_q = self.num_attention_heads
            cos, sin = position_embeddings

            # K/V remain full because Qwen uses GQA: several query heads share
            # one KV head, so freezing an arbitrary query head cannot remove a
            # corresponding slice of the K/V projection.
            k = self.k_proj(hidden_states).view(
                B, T, self.num_key_value_heads, head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(
                B, T, self.num_key_value_heads, head_dim).transpose(1, 2)
            k = self.k_norm(k)
            k_full = repeat_kv(k, self._real_num_kv_groups)
            v_full = repeat_kv(v, self._real_num_kv_groups)

            # One compact projection contains (query, gate) for standard heads,
            # followed by gate-only rows for pattern heads.
            qg = self.qg_proj(hidden_states)
            std_width = n_std * 2 * head_dim
            if n_std > 0:
                q_std_proj = qg[..., :std_width]
                q_std_proj = q_std_proj.view(B, T, n_std, head_dim * 2)
                q_std_raw, gate_std = torch.chunk(q_std_proj, 2, dim=-1)
                q_std = self.q_norm(q_std_raw).transpose(1, 2)
                k_std = k_full[:, self.standard_head_idx]
                q_std, k_std = apply_rotary_pos_emb(q_std, k_std, cos, sin)
            else:
                q_std = v_full[:, :0]
                k_std = v_full[:, :0]
                gate_std = hidden_states.new_empty(B, T, 0, head_dim)

            # Pattern heads still require Qwen's query gate even though their
            # query vector and QK/softmax are removed.
            if n_pat > 0:
                gate_pat = qg[..., std_width:].view(
                    B, T, n_pat, head_dim)
            else:
                gate_pat = hidden_states.new_empty(B, T, 0, head_dim)

            # Triton path: one mixed-head launch. This is deliberately limited
            # to square, unpadded, no-cache causal prefill. The kernel has no
            # cache offset or arbitrary-mask interface yet.
            mask_is_supported = attention_mask is None
            if self.assume_causal_prefill and isinstance(attention_mask, torch.Tensor):
                mask_is_supported = (
                    attention_mask.ndim == 4
                    and attention_mask.shape[-2:] == (T, T)
                    and attention_mask.shape[0] in (1, B)
                )
            can_fuse = (
                self.use_triton_kernel
                and n_pat > 0
                and not self.training
                and fused_attn_usable(hidden_states)
                and mask_is_supported
                and past_key_values is None
                and self._fused_prior_ready(T)
            )
            if can_fuse:
                out_heads = fused_mixed_attn(
                    q_std, k_std, v_full,
                    frozen=self.frozen_mask_i32, hidx=self.hidx_i32,
                    pidx=self.pidx_i32, sm_scale=self.scaling, validate=False,
                    **self._fused_prior_kwargs(),
                )
                out = out_heads.transpose(1, 2)
                gate = hidden_states.new_empty(B, T, n_q, head_dim)
                if n_std > 0:
                    gate[:, :, self.standard_head_idx] = gate_std
                if n_pat > 0:
                    gate[:, :, self.pattern_head_idx] = gate_pat
                out = out * torch.sigmoid(gate)
                self.triton_calls += 1
                return self.o_proj(out.reshape(B, T, n_q * head_dim)), None

            # Portable split fallback: PyTorch SDPA for standard heads and dense
            # P@V for pattern heads. It also serves as the correctness oracle for
            # the fused Qwen adapter.
            if n_std > 0:
                v_std = v_full[:, self.standard_head_idx]
                impl_name = getattr(self.config, "_attn_implementation", "eager")
                if impl_name in ("sdpa", "flash_attention_2") and T > 1:
                    import torch.nn.functional as F
                    sdpa_out = F.scaled_dot_product_attention(
                        q_std, k_std, v_std,
                        attn_mask=attention_mask,
                        is_causal=attention_mask is None,
                        scale=self.scaling,
                    )
                    out_std = sdpa_out.transpose(1, 2).contiguous()
                else:
                    try:
                        from transformers import ALL_ATTENTION_FUNCTIONS
                        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                            impl_name, eager_attention_forward)
                    except Exception:
                        attention_interface = eager_attention_forward
                    out_std, _ = attention_interface(
                        self, q_std, k_std, v_std, attention_mask,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        scaling=self.scaling,
                    )
                out_std = out_std.view(B, T, n_std, head_dim)
                out_std = out_std * torch.sigmoid(gate_std)
            else:
                out_std = hidden_states.new_zeros(B, T, 0, head_dim)

            if n_pat > 0:
                v_pat = v_full[:, self.pattern_head_idx]
                pat = self._split_patterns(T, v_pat.dtype)
                out_pat = torch.matmul(pat.unsqueeze(0), v_pat)
                out_pat = out_pat.transpose(1, 2).contiguous()
                out_pat = out_pat * torch.sigmoid(gate_pat)
            else:
                out_pat = hidden_states.new_zeros(B, T, 0, head_dim)

            out = hidden_states.new_empty(B, T, n_q, head_dim)
            if n_std > 0:
                out[:, :, self.standard_head_idx] = out_std
            if n_pat > 0:
                out[:, :, self.pattern_head_idx] = out_pat
            out = out.reshape(B, T, n_q * head_dim)

            out = self.o_proj(out)
            return out, None

    return HybridQwen3_5Attention


# ---------------------------------------------------------------------------
# Qwen3 implementation (Qwen3-4B etc.: plain GQA + QK-norm + RoPE, no gate)
# ---------------------------------------------------------------------------

def make_qwen3_hybrid_class():
    """Build the Qwen3 hybrid class lazily.

    Qwen3 differs from Qwen3.5 in being the simple case: q_proj emits exactly
    head_dim features per head (no output gate), every layer is a standard
    softmax GQA attention, and RMSNorm is applied per head to q and k before
    RoPE. The structure below mirrors HF's `Qwen3Attention.forward`.
    """
    qwen3 = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    OriginalAttention = qwen3.Qwen3Attention
    eager_attention_forward = qwen3.eager_attention_forward
    repeat_kv = qwen3.repeat_kv
    apply_rotary_pos_emb = qwen3.apply_rotary_pos_emb

    class HybridQwen3Attention(HybridAttentionBase, OriginalAttention):
        ORIGINAL_CLASS = OriginalAttention

        def __init__(self, original: OriginalAttention,
                     pattern: torch.Tensor, head_mask: torch.Tensor,
                     device, dtype, use_triton=True,
                     assume_causal_prefill=False, decomp=None):
            nn.Module.__init__(self)
            cfg = original.config
            self.config = cfg
            self.layer_idx = original.layer_idx
            self.layer_type = getattr(original, "layer_type", None)
            self.attention_dropout = original.attention_dropout
            self.num_attention_heads = cfg.num_attention_heads
            self.num_key_value_heads = cfg.num_key_value_heads
            self._real_num_kv_groups = original.num_key_value_groups
            # K/V are broadcast to query heads by hand below; make the inner
            # repeat_kv of eager_attention_forward a no-op (see Qwen3.5 notes).
            self.num_key_value_groups = 1
            self.head_dim = original.head_dim
            self.scaling = original.scaling
            self.is_causal = original.is_causal
            self.sliding_window = getattr(original, "sliding_window", None)
            self.use_triton_kernel = use_triton
            self.assume_causal_prefill = assume_causal_prefill

            self.k_proj = original.k_proj
            self.v_proj = original.v_proj
            self.o_proj = original.o_proj
            self.q_norm = original.q_norm
            self.k_norm = original.k_norm

            self._hybrid_init(
                original, pattern, head_mask,
                n_q=self.num_attention_heads,
                head_dim=self.head_dim,
                device=device, dtype=dtype,
                q_per_head_dim=self.head_dim,
                decomp=decomp,
            )
            # GQA map for the fused kernel: query head h reads KV head
            # h // groups directly from the raw (B, HKV, T, hd) K/V — no
            # repeat_kv expansion and no per-head gather on the fused path.
            self.register_buffer("kvidx_i32", (
                torch.arange(self.num_attention_heads, device=device)
                // self._real_num_kv_groups).to(torch.int32))

        def forward(self, hidden_states, position_embeddings,
                    attention_mask=None, past_key_values=None,
                    cache_position=None, **kwargs):
            self.hybrid_calls += 1
            B, T, _ = hidden_states.shape
            n_std = self.n_standard_heads
            n_pat = self.n_pattern_heads
            head_dim = self.head_dim
            n_q = self.num_attention_heads
            cos, sin = position_embeddings

            # Raw K/V heads (GQA: several query heads share one KV head, so no
            # K/V rows can be dropped for frozen query heads).
            k = self.k_norm(self.k_proj(hidden_states).view(
                B, T, self.num_key_value_heads, head_dim)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(
                B, T, self.num_key_value_heads, head_dim).transpose(1, 2)

            mask_is_supported = attention_mask is None
            if self.assume_causal_prefill and isinstance(attention_mask, torch.Tensor):
                mask_is_supported = (
                    attention_mask.ndim == 4
                    and attention_mask.shape[-2:] == (T, T)
                    and attention_mask.shape[0] in (1, B)
                )
            can_fuse = (
                self.use_triton_kernel
                and n_pat > 0
                and not self.training
                and fused_attn_usable(hidden_states)
                and mask_is_supported
                and past_key_values is None
                and self.sliding_window is None
                and self._fused_prior_ready(T)
            )
            if can_fuse:
                # Q only for the standard heads (compact projection); RoPE is
                # per position, so q (n_std heads) and raw k (HKV heads) can
                # be rotated independently of the GQA broadcast.
                if n_std > 0:
                    q_std = self.q_norm(self.q_proj_std(hidden_states).view(
                        B, T, n_std, head_dim)).transpose(1, 2)
                    q_std, k = apply_rotary_pos_emb(q_std, k, cos, sin)
                else:
                    q_std = v[:, :0]
                out_heads = fused_mixed_attn(
                    q_std, k, v,
                    frozen=self.frozen_mask_i32, hidx=self.hidx_i32,
                    pidx=self.pidx_i32, kvidx=self.kvidx_i32,
                    sm_scale=self.scaling, validate=False,
                    **self._fused_prior_kwargs(),
                )
                self.triton_calls += 1
                out = out_heads.transpose(1, 2).reshape(B, T, n_q * head_dim)
                return self.o_proj(out), None

            # Portable split fallback (also the correctness oracle): SDPA /
            # eager for standard heads, dense P@V for pattern heads.
            k_full = repeat_kv(k, self._real_num_kv_groups)
            v_full = repeat_kv(v, self._real_num_kv_groups)
            if n_std > 0:
                q_std = self.q_norm(self.q_proj_std(hidden_states).view(
                    B, T, n_std, head_dim)).transpose(1, 2)
                k_std = k_full[:, self.standard_head_idx]
                q_std, k_std = apply_rotary_pos_emb(q_std, k_std, cos, sin)
            else:
                q_std = v_full[:, :0]
                k_std = v_full[:, :0]
            if n_std > 0:
                v_std = v_full[:, self.standard_head_idx]
                impl_name = getattr(self.config, "_attn_implementation", "eager")
                if impl_name in ("sdpa", "flash_attention_2") and T > 1:
                    import torch.nn.functional as F
                    sdpa_out = F.scaled_dot_product_attention(
                        q_std, k_std, v_std,
                        attn_mask=attention_mask,
                        is_causal=attention_mask is None,
                        scale=self.scaling,
                    )
                    out_std = sdpa_out.transpose(1, 2).contiguous()
                else:
                    try:
                        from transformers import ALL_ATTENTION_FUNCTIONS
                        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                            impl_name, eager_attention_forward)
                    except Exception:
                        attention_interface = eager_attention_forward
                    out_std, _ = attention_interface(
                        self, q_std, k_std, v_std, attention_mask,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        scaling=self.scaling,
                    )
                out_std = out_std.view(B, T, n_std, head_dim)
            else:
                out_std = hidden_states.new_zeros(B, T, 0, head_dim)

            if n_pat > 0:
                v_pat = v_full[:, self.pattern_head_idx]
                pat = self._split_patterns(T, v_pat.dtype)
                out_pat = torch.matmul(pat.unsqueeze(0), v_pat)
                out_pat = out_pat.transpose(1, 2).contiguous()
            else:
                out_pat = hidden_states.new_zeros(B, T, 0, head_dim)

            out = hidden_states.new_empty(B, T, n_q, head_dim)
            if n_std > 0:
                out[:, :, self.standard_head_idx] = out_std
            if n_pat > 0:
                out[:, :, self.pattern_head_idx] = out_pat
            out = self.o_proj(out.reshape(B, T, n_q * head_dim))
            return out, None

    return HybridQwen3Attention


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Use class names so we don't have to import every model family at import time.
_REGISTRY_BUILDERS = {
    "Qwen3_5Attention": make_qwen35_hybrid_class,
    "Qwen3Attention": make_qwen3_hybrid_class,
    # To add a new family:
    #   "LlamaAttention":   make_llama_hybrid_class,
    #   "MistralAttention": make_mistral_hybrid_class,
}

_BUILT_CACHE: dict = {}


def get_hybrid_class_for(original_module: nn.Module) -> Optional[Type]:
    cls_name = type(original_module).__name__
    if cls_name not in _REGISTRY_BUILDERS:
        return None
    if cls_name not in _BUILT_CACHE:
        _BUILT_CACHE[cls_name] = _REGISTRY_BUILDERS[cls_name]()
    return _BUILT_CACHE[cls_name]


# ---------------------------------------------------------------------------
# Conversion entry point
# ---------------------------------------------------------------------------

def convert_targets_to_hybrid(model, targets, mean_attn, head_masks,
                              device, dtype, use_triton=True,
                              assume_causal_prefill=False, decomp=None):
    """Swap each target's attention module for its hybrid counterpart.

    mean_attn: (n_targets, n_heads, T, T) dense priors, or None when `decomp`
    is given. decomp: optional list (one per target) of (alpha, rho) pairs of
    (n_heads, S) fp32 tensors — the decomposed prior; entries may be None for
    targets with no frozen head.
    Returns a `saved` list that can be passed to `restore_originals` to undo.
    """
    if mean_attn is None:
        assert decomp is not None, "need mean_attn or decomp"
        priors = [None] * len(targets)
    else:
        priors = list(mean_attn)
    assert len(targets) == len(priors) == len(head_masks)
    if decomp is not None:
        assert len(decomp) == len(targets)
    from test_prior_pretrained import _iter_blocks
    blocks = _iter_blocks(model)
    saved = []
    converted = 0
    for ti, (t, prior, mask) in enumerate(zip(targets, priors, head_masks)):
        if not mask.any():
            saved.append((blocks[t.layer_idx], t.attr_name, None))
            continue
        original = getattr(blocks[t.layer_idx], t.attr_name)
        HybridCls = get_hybrid_class_for(original)
        if HybridCls is None:
            raise RuntimeError(
                f"No hybrid class registered for {type(original).__name__}. "
                f"Add a factory to _REGISTRY_BUILDERS in hybrid_prior_attention.py.")
        hybrid = HybridCls(original, prior, mask, device, dtype,
                           use_triton=use_triton,
                           assume_causal_prefill=assume_causal_prefill,
                           decomp=None if decomp is None else decomp[ti])
        # The parent model may already be in eval mode. Newly constructed
        # nn.Modules default to training=True, which would silently disable the
        # inference-only fused path (and incorrectly enable attention dropout).
        hybrid.train(original.training)
        setattr(blocks[t.layer_idx], t.attr_name, hybrid)
        saved.append((blocks[t.layer_idx], t.attr_name, original))
        converted += 1
    backend = "triton-fused" if use_triton else "torch-split"
    repr_ = "decomposed" if decomp is not None else "dense"
    print(f"  [hybrid] converted {converted}/{len(targets)} attention modules "
          f"(backend={backend}, prior={repr_}; Triton requires no-cache unpadded prefill)")
    return saved


def restore_originals(saved):
    """Undo `convert_targets_to_hybrid`."""
    for block, attr_name, original in saved:
        if original is not None:
            setattr(block, attr_name, original)
