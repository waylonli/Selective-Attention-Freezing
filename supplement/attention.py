"""Compute-saving uniform attention and physical head pruning."""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from nanogpt.model import apply_rotary_pos_emb


class CausalUniform(torch.autograd.Function):
    """Linear-time causal mean with FP32 accumulation and reverse-sum backward."""

    @staticmethod
    def forward(ctx, value):
        ctx.length = value.shape[-2]
        denominator = torch.arange(1, ctx.length + 1, device=value.device,
                                   dtype=torch.float32).view(-1, 1)
        return (value.float().cumsum(-2) / denominator).to(value.dtype)

    @staticmethod
    def backward(ctx, grad):
        denominator = torch.arange(1, ctx.length + 1, device=grad.device,
                                   dtype=torch.float32).view(-1, 1)
        scaled = grad.float() / denominator
        return scaled.flip(-2).cumsum(-2).flip(-2).to(grad.dtype)


def replace_parameter(old, new, optimizer, transform):
    """Preserve the Adam step and slice its moments exactly like the weights."""
    if optimizer is None:
        return
    found = 0
    for group in optimizer.param_groups:
        for i, parameter in enumerate(group['params']):
            if parameter is old:
                group['params'][i] = new
                found += 1
    if found != 1:
        raise RuntimeError(f'expected one optimiser reference, found {found}')
    state = optimizer.state.pop(old, {})
    optimizer.state[new] = {
        key: (transform(value).clone() if isinstance(value, torch.Tensor)
              and value.shape == old.shape else copy.deepcopy(value))
        for key, value in state.items()
    }


def sliced_linear(old, indices, dimension, optimizer):
    indices = indices.to(old.weight.device)
    weight = old.weight.detach().index_select(dimension, indices).clone()
    # Avoid random initialisation: creating the replacement must not consume RNG.
    new = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(new)
    new.in_features, new.out_features = weight.shape[1], weight.shape[0]
    new.weight = nn.Parameter(weight)
    replace_parameter(old.weight, new.weight, optimizer,
                      lambda value: value.index_select(dimension, indices))
    if old.bias is None:
        new.register_parameter('bias', None)
    elif dimension == 0:
        new.bias = nn.Parameter(old.bias.detach().index_select(0, indices).clone())
        replace_parameter(old.bias, new.bias, optimizer,
                          lambda value: value.index_select(0, indices))
    else:
        new.bias = old.bias
    return new


class PackedAttention(nn.Module):
    """Ordinary heads plus prefix-sum heads, or physically removed heads.

    Projections and optimiser states are compacted once at intervention. Uniform
    heads retain V and output weights; pruning removes Q/K/V rows and output
    columns. There are no per-update parameter gathers or zeroed dense heads.
    """

    def __init__(self, original, selected, mode, optimizer=None):
        super().__init__()
        if mode not in ('uniform', 'prune'):
            raise ValueError(mode)
        if original.dropout != 0 or original.attn_mask_rate != 0:
            raise ValueError('this control requires zero attention dropout/masking')
        selected = sorted(selected)
        if len(set(selected)) != len(selected) or any(
                h < 0 or h >= original.n_head for h in selected):
            raise ValueError('invalid selected head list')
        live = [h for h in range(original.n_head) if h not in selected]
        self.mode = mode
        self.n_live, self.n_fixed = len(live), len(selected)
        self.head_dim = original.head_dim
        self.position_encoding = original.position_encoding
        self.resid_dropout = original.resid_dropout
        def columns(heads):
            return torch.tensor([h * self.head_dim + j for h in heads
                                 for j in range(self.head_dim)], dtype=torch.long)
        live_cols = columns(live)
        value_cols = columns(live + selected if mode == 'uniform' else live)
        self.q_proj = sliced_linear(original.q_proj, live_cols, 0, optimizer)
        self.k_proj = sliced_linear(original.k_proj, live_cols, 0, optimizer)
        self.v_proj = sliced_linear(original.v_proj, value_cols, 0, optimizer)
        self.c_proj = sliced_linear(original.c_proj, value_cols, 1, optimizer)

    def is_masked(self):
        return True

    def forward(self, x, rope=None):
        batch, length, _ = x.shape
        n_value = self.n_live + (self.n_fixed if self.mode == 'uniform' else 0)
        value = self.v_proj(x).view(batch, length, n_value, self.head_dim).transpose(1, 2)
        pieces = []
        if self.n_live:
            query = self.q_proj(x).view(batch, length, self.n_live, self.head_dim).transpose(1, 2)
            key = self.k_proj(x).view(batch, length, self.n_live, self.head_dim).transpose(1, 2)
            if self.position_encoding == 'rope':
                if rope is None:
                    raise ValueError('RoPE cache is required')
                query, key = apply_rotary_pos_emb(query, key, *rope)
            pieces.append(F.scaled_dot_product_attention(
                query, key, value[:, :self.n_live], is_causal=True))
        if self.mode == 'uniform' and self.n_fixed:
            pieces.append(CausalUniform.apply(value[:, self.n_live:]))
        mixed = (torch.cat(pieces, dim=1) if len(pieces) > 1
                 else pieces[0] if pieces else value)
        output = mixed.transpose(1, 2).reshape(batch, length, n_value * self.head_dim)
        return self.resid_dropout(self.c_proj(output))


def install_packed(model, heads_by_layer, mode, optimizer=None):
    for layer, selected in enumerate(heads_by_layer):
        if selected:
            block = model.transformer.h[layer]
            block.main_block = PackedAttention(block.main_block, selected, mode, optimizer)


def matched_random_heads(reference, n_head, seed):
    generator = torch.Generator().manual_seed(seed)
    return [sorted(torch.randperm(n_head, generator=generator)[:len(heads)].tolist())
            for heads in reference]


def random_eligible_heads(reference, n_head, seed, install):
    """Randomise candidate order within each layer and retain the common fit gate."""
    generator = torch.Generator().manual_seed(seed)
    result=[]
    for layer, target in enumerate(reference):
        order=torch.randperm(n_head,generator=generator).tolist()
        chosen=[];cursor=0
        while len(chosen)<len(target) and cursor<n_head:
            count=len(target)-len(chosen)
            candidates=order[cursor:cursor+count];cursor+=len(candidates)
            accepted=install([(layer,h) for h in candidates])
            if any(li!=layer or h not in candidates for li,h in accepted):
                raise RuntimeError('installer returned a head outside the candidate set')
            chosen.extend(h for li,h in accepted)
        if len(chosen)!=len(target) or len(set(chosen))!=len(chosen):
            raise RuntimeError('not enough fit-eligible heads for the matched layer count')
        result.append(sorted(chosen))
    return result
