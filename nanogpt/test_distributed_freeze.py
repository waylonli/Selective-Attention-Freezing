import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_freeze import (
    create_freeze_control_group, synchronize_decomposed_freeze, wait_for_freeze_fit,
)


class _FakeAttention:
    prior_repr = "decomposed"
    dyn_pattern = None

    def __init__(self):
        self.dyn_frozen = torch.zeros(3, dtype=torch.bool)
        self.dyn_decomp = torch.zeros(3, dtype=torch.bool)
        self.dyn_alpha = torch.zeros(3, 8)
        self.dyn_rho = torch.zeros(3, 8)
        self.dyn_z = torch.zeros(3, 8)
        self.rebuilt = False

    def refresh_dynamic_derived_state(self, *, rebuild_rho_band=False):
        self.rebuilt = bool(rebuild_rho_band)


def _distributed_worker(rank, init_file, delay=0.0):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        control = create_freeze_control_group(timeout_seconds=15)
        module = _FakeAttention()
        if rank == 0:
            time.sleep(delay)
            module.dyn_frozen[:] = torch.tensor([True, False, True])
            module.dyn_decomp.copy_(module.dyn_frozen)
            module.dyn_alpha.copy_(torch.arange(24).reshape(3, 8))
            module.dyn_rho.copy_(module.dyn_alpha + 100)
            module.dyn_z.copy_(module.dyn_alpha + 200)
        wait_for_freeze_fit(control, timeout_seconds=15)
        digest = synchronize_decomposed_freeze([module])
        assert len(digest) == 64
        assert module.dyn_frozen.tolist() == [True, False, True]
        assert float(module.dyn_alpha[-1, -1]) == 23.0
        assert module.rebuilt == (rank != 0)
        dist.destroy_process_group(control)
    finally:
        dist.destroy_process_group()


class DistributedFreezeTests(unittest.TestCase):
    def test_rank_zero_state_is_broadcast_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "process_group")
            mp.spawn(_distributed_worker, args=(init_file,), nprocs=2, join=True)

    def test_slow_source_waits_on_cpu_before_broadcast(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "process_group")
            mp.spawn(_distributed_worker, args=(init_file, 0.5), nprocs=2, join=True)

    def test_control_group_is_gloo_with_bounded_timeout(self):
        with patch("distributed_freeze.dist.is_initialized", return_value=True), \
             patch("distributed_freeze.dist.new_group") as new_group:
            create_freeze_control_group(timeout_seconds=17)
            new_group.assert_called_once_with(backend="gloo", timeout=timedelta(seconds=17))

    def test_wait_uses_host_barrier(self):
        group = object()
        with patch("distributed_freeze.dist.monitored_barrier") as barrier:
            wait_for_freeze_fit(group, timeout_seconds=19)
            barrier.assert_called_once_with(
                group=group, timeout=timedelta(seconds=19), wait_all_ranks=True)

    def test_missing_control_group_fails_closed(self):
        with patch("distributed_freeze.dist.is_initialized", return_value=True):
            with self.assertRaises(ValueError):
                wait_for_freeze_fit(None)

    def test_invalid_timeouts(self):
        with self.assertRaises(ValueError):
            create_freeze_control_group(timeout_seconds=0)
        with self.assertRaises(ValueError):
            wait_for_freeze_fit(None, timeout_seconds=-1)


if __name__ == "__main__":
    unittest.main()
