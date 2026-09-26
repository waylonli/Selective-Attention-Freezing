"""One-shot head pruning by per-sequence loss sensitivity of output gates."""

from contextlib import contextmanager
import hashlib
import math
import time

import torch

from supplement.maturity_protocol import require
from supplement.attention import install_packed


@contextmanager
def output_gates(model, gates):
    """Gate each concatenated head before its output projection, in original order."""
    handles = []
    try:
        require(len(gates) == len(model.transformer.h), 'Wrong number of gate layers')
        for block, gate in zip(model.transformer.h, gates):
            attn = block.main_block
            require(hasattr(attn, 'dyn_frozen') and not attn.dyn_frozen.any().item(),
                    'Gate selection requires an ordinary, unpacked parent')
            require(gate.ndim == 2 and gate.shape[1] == attn.n_head, 'Wrong gate shape')
            def apply_gate(module, args, gate=gate, heads=attn.n_head):
                value, = args
                batch, length, width = value.shape
                require(batch == gate.shape[0] and width % heads == 0, 'Invalid head layout')
                shaped = value.reshape(batch, length, heads, width // heads)
                # Keep gate derivatives in FP32 even when activations are BF16.
                dtype = torch.promote_types(shaped.dtype, gate.dtype)
                gated = shaped.to(dtype) * gate.to(dtype)[:, None, :, None]
                return (gated.to(value.dtype).reshape_as(value),)
            handles.append(attn.c_proj.register_forward_pre_hook(apply_gate))
        yield
    finally:
        for handle in handles:
            handle.remove()


def gate_derivatives(model, x, y, context):
    """Return signed d(mean-token NLL of sequence)/dz, shaped [B, L, H]."""
    require(y.shape == x.shape and (y >= 0).all().item(), 'Expected equal-length unpadded targets')
    gates = [torch.ones(x.shape[0], b.main_block.n_head, device=x.device,
                        dtype=torch.float32, requires_grad=True) for b in model.transformer.h]
    with torch.enable_grad(), output_gates(model, gates), context():
        _, loss = model(x, y)
        # Loss averages across B sequences; undo that factor before abs/averaging.
        derivatives = torch.autograd.grad(loss * x.shape[0], gates)
    return torch.stack(derivatives, dim=1).detach()


def rank_heads(raw_scores, rate=.25):
    require(raw_scores.ndim == 2 and torch.isfinite(raw_scores).all().item()
            and (raw_scores >= 0).all().item(), 'Invalid gate importance scores')
    require(0 < rate < 1, 'Invalid pruning rate')
    norms = torch.linalg.vector_norm(raw_scores, dim=1, keepdim=True)
    scores = raw_scores / norms.clamp_min(torch.finfo(raw_scores.dtype).tiny)
    count = round(rate * scores.numel())
    order = sorted(range(scores.numel()), key=lambda i: (float(scores.flatten()[i]), i))
    heads = [sorted(i % scores.shape[1] for i in order[:count] if i // scores.shape[1] == layer)
             for layer in range(scores.shape[0])]
    return scores, heads


def intervene(model, optimiser, windows, context, cfg):
    def sync():
        if windows.device == 'cuda':
            torch.cuda.synchronize()
    require(all(p.grad is None for p in model.parameters()), 'Unexpected pending gradients')
    sync()
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    rows, digest = [], hashlib.sha256()
    try:
        for _ in range(cfg['calibration_batches']):
            x, y = windows.calibration('train')
            for tensor in (x, y):
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            derivative = gate_derivatives(model, x, y, context).cpu().double()
            require(torch.isfinite(derivative).all().item(), 'Non-finite gate derivative')
            rows.append(derivative)
    finally:
        model.train(was_training)
    require(all(p.grad is None for p in model.parameters()), 'Scoring changed parameter gradients')
    signed = torch.cat(rows)
    raw = signed.abs().mean(dim=0)
    scores, heads = rank_heads(raw)
    sync()
    measured = time.perf_counter()
    install_packed(model, heads, 'prune', optimiser)
    sync()
    installed = time.perf_counter()
    require(math.isfinite(installed-started), 'Invalid intervention time')
    return dict(mode='prune_gate_taylor', physical_mode='prune',
                selector='head_gate_taylor_layer_l2', rate=.25, heads=heads,
                per_layer_counts=list(map(len, heads)),
                score_definition='mean_sequence(abs(d(mean_token_NLL)/d(head_output_gate)))',
                normalisation='L2 within each layer; zero-norm rows remain zero',
                ranking='global ascending normalised score; flattened head index breaks ties',
                selection_schedule='one-shot at the shared trained parent, not iterative pruning',
                raw_scores=raw.tolist(), normalised_scores=scores.tolist(),
                signed_sequence_derivatives=signed.tolist(),
                calibration_sequences=len(signed), calibration_tokens=len(signed)*cfg['context'],
                calibration_inputs_sha256=digest.hexdigest(),
                scoring_s=measured-started, installation_s=installed-measured,
                total_s=installed-started)
