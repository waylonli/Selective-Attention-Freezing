"""Opt-in learnable compact patterns; the published frozen path stays unchanged."""

import torch
from torch import nn


def dense_reference(value, alpha, rho):
    length = value.shape[-2]
    i = torch.arange(length, device=value.device)
    distance = i[:, None] - i[None, :]
    logits = alpha[:, None, :length] + rho[:, distance.clamp_min(0)]
    p = logits.masked_fill(distance < 0, -float('inf')).softmax(-1)
    return p.to(value.dtype) @ value


def enable(model, optimizer=None):
    parameters = []
    for block in model.transformer.h:
        attn = block.main_block
        if not hasattr(attn, 'dyn_decomp') or not attn.dyn_decomp.any().item():
            continue
        for name in ('dyn_alpha', 'dyn_rho'):
            old = getattr(attn, name)
            if isinstance(old, nn.Parameter):
                raise RuntimeError('Pattern parameters were already enabled')
            delattr(attn, name)
            parameter = nn.Parameter(old.detach().clone())
            attn.register_parameter(name, parameter)
            parameters.append(parameter)
        with torch.no_grad():
            attn.refresh_dynamic_derived_state(rebuild_rho_band=True)
        attn.register_state_dict_pre_hook(_export_current_buffers)
    if not parameters:
        raise RuntimeError('No fitted patterns to train')
    if optimizer is not None:
        optimizer.add_param_group(dict(params=parameters, weight_decay=0.))
    patch_fused_dispatch()
    if optimizer is not None:
        attach_optimizer(model, optimizer)
    return parameters


def attach_optimizer(model, optimizer):
    from nanogpt import model as module
    patch_fused_dispatch()
    for name, parameter in model.named_parameters():
        if name.endswith(('dyn_alpha', 'dyn_rho')):
            parameter._pattern_cache_enabled = True
    # Fused Adam does not reliably increment Tensor._version.
    optimizer.register_step_post_hook(lambda *args: module.fused_mixed_attn.clear_pattern_cache())


@torch.no_grad()
def _export_current_buffers(attn, prefix, keep_vars):
    from supplement.trainable_pattern_kernels import normalisers
    from kernel.fused_attn import make_rho_band
    attn.dyn_z.copy_(normalisers(attn.dyn_alpha, attn.dyn_rho, attn.dyn_alpha.shape[1]))
    attn.dyn_rho_band.copy_(make_rho_band(attn.dyn_rho))


def patch_fused_dispatch():
    from nanogpt import model as module
    if getattr(module.fused_mixed_attn, '_trainable_patterns', False):
        return
    original = module.fused_mixed_attn
    from kernel.fused_attn import make_rho_band
    from supplement.trainable_pattern_kernels import normalisers, PatternGradient
    cache = {}

    def dispatch(q, k, v, patt, frozen, hidx=None, pidx=None, **kwargs):
        alpha, rho = kwargs.get('alpha'), kwargs.get('rho')
        if alpha is None or not alpha.requires_grad:
            return original(q, k, v, patt, frozen, hidx, pidx, **kwargs)
        key = id(alpha)
        signature = (alpha._version, rho._version, v.shape[-2], alpha.device)
        old = cache.get(key)
        if (not getattr(alpha, '_pattern_cache_enabled', False)
                or old is None or old[0] is not alpha or old[1] != signature):
            with torch.no_grad():
                z = normalisers(alpha, rho, v.shape[-2])
                band = make_rho_band(rho)
            cache[key] = (alpha, signature, z, band)
        else:
            z, band = old[2:]
        kwargs.update(alpha=alpha.detach(), rho=rho.detach(), z=z, rho_band=band)
        out = original(q, k, v, patt, frozen, hidx, pidx, **kwargs)
        if torch.is_grad_enabled():
            out = PatternGradient.apply(out, v, alpha, rho, z, frozen)
        return out

    dispatch._trainable_patterns = True
    dispatch.clear_pattern_cache = cache.clear
    module.fused_mixed_attn = dispatch
