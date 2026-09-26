"""Distributed synchronization for dynamic frozen-attention state."""

from __future__ import annotations

import hashlib
from datetime import timedelta

import torch
import torch.distributed as dist


_SYNC_BUFFER_NAMES = ("dyn_frozen", "dyn_decomp", "dyn_alpha", "dyn_rho", "dyn_z")


def create_freeze_control_group(*, timeout_seconds: float = 7200):
    """Create a CPU-only group for the long rank-zero calibration phase."""
    if timeout_seconds <= 0:
        raise ValueError("freeze control timeout must be positive")
    if not dist.is_available() or not dist.is_initialized():
        return None
    return dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))


def wait_for_freeze_fit(group, *, timeout_seconds: float = 7200) -> None:
    """Do not enqueue NCCL broadcasts while rank zero is still fitting priors."""
    if timeout_seconds <= 0:
        raise ValueError("freeze control timeout must be positive")
    if group is None:
        if dist.is_available() and dist.is_initialized():
            raise ValueError("distributed freeze requires a CPU control group")
        return
    dist.monitored_barrier(
        group=group, timeout=timedelta(seconds=timeout_seconds), wait_all_ranks=True)


def _state_digest(modules) -> bytes:
    digest = hashlib.sha256()
    for layer, module in enumerate(modules):
        digest.update(layer.to_bytes(4, "little"))
        for name in _SYNC_BUFFER_NAMES:
            tensor = getattr(module, name)
            digest.update(name.encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(raw.tobytes())
    return digest.digest()


@torch.no_grad()
def synchronize_decomposed_freeze(modules, *, source_rank: int = 0) -> str:
    """Broadcast one rank's compact freeze state and assert byte identity.

    ``dyn_rho_band`` is a large deterministic cache, so each destination rank
    rebuilds it locally from the broadcast ``dyn_rho`` instead of transferring
    it. Dense T x T priors are deliberately unsupported for the long-context
    DDP path.
    """
    if not dist.is_available() or not dist.is_initialized():
        return _state_digest(modules).hex()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if not modules:
        raise ValueError("cannot synchronize an empty attention-module list")
    for module in modules:
        if getattr(module, "prior_repr", None) != "decomposed":
            raise ValueError(
                "distributed dynamic freezing supports only compact decomposed priors")
        if getattr(module, "dyn_pattern", None) is not None:
            raise ValueError(
                "distributed compact freezing refuses a lazy dense-pattern fallback")
        for name in _SYNC_BUFFER_NAMES:
            dist.broadcast(getattr(module, name), src=source_rank)
        module.refresh_dynamic_derived_state(rebuild_rho_band=rank != source_rank)

    digest = _state_digest(modules)
    device = modules[0].dyn_frozen.device
    local = torch.tensor(list(digest), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    if any(not torch.equal(gathered[0], candidate) for candidate in gathered[1:]):
        raise RuntimeError("dynamic freeze state differs across DDP ranks after broadcast")
    dist.barrier()
    return digest.hex()
