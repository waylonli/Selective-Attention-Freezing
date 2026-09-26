"""Deterministic training-window samplers used by formal experiments."""

from __future__ import annotations

import torch


class ShuffledNonoverlapSampler:
    """One shuffled pass over disjoint next-token target windows.

    Starts are ``lo + permutation[k] * window_size``. Consecutive windows may
    share the boundary token as input/target context, but their *target token*
    ranges are disjoint: ``[start+1, start+window_size+1)``. The permutation is
    reconstructed from its seed; checkpoints only need the cursor and metadata.
    """

    schema_version = 2

    def __init__(self, *, lo: int, end: int, window_size: int, seed: int,
                 rank: int = 0, world_size: int = 1):
        if not 0 <= lo < end:
            raise ValueError(f"invalid token interval [{lo}, {end})")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(
                f"invalid distributed partition rank={rank}, world_size={world_size}")
        # A window needs T inputs plus the following target token.
        self.n_windows = (end - lo - 1) // window_size
        if self.n_windows <= 0:
            raise ValueError("token interval is too short for one training window")
        self.lo = int(lo)
        self.end = int(end)
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.rank_windows = (
            0 if self.rank >= self.n_windows
            else 1 + (self.n_windows - 1 - self.rank) // self.world_size
        )
        generator = torch.Generator().manual_seed(self.seed)
        self.permutation = torch.randperm(self.n_windows, generator=generator)
        self.cursor = 0

    def take(self, count: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive")
        stop = self.cursor + count
        if stop > self.rank_windows:
            raise RuntimeError(
                "non-repeating training data exhausted: "
                f"rank {self.rank} requested local windows [{self.cursor}, {stop}), "
                f"rank capacity={self.rank_windows}, global capacity={self.n_windows}")
        # All ranks reconstruct the same global permutation. Rank r consumes
        # positions r, r+world_size, ... so target windows are globally disjoint
        # without communication and the shared local cursor is exactly resumable.
        positions = (
            torch.arange(self.cursor, stop, dtype=torch.long) * self.world_size
            + self.rank
        )
        ids = self.permutation[positions]
        self.cursor = stop
        return self.lo + ids * self.window_size

    def state_dict(self, *, cursor_rewind: int = 0) -> dict[str, int]:
        cursor = self.cursor - int(cursor_rewind)
        if not 0 <= cursor <= self.rank_windows:
            raise ValueError(
                f"cannot rewind sampler cursor {self.cursor} by {cursor_rewind}")
        return {
            "schema_version": self.schema_version,
            "lo": self.lo,
            "end": self.end,
            "window_size": self.window_size,
            "seed": self.seed,
            "n_windows": self.n_windows,
            "world_size": self.world_size,
            "cursor": cursor,
            "global_windows_reserved": cursor * self.world_size,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        state_schema = int(state["schema_version"])
        if state_schema == 1:
            if self.world_size != 1:
                raise ValueError(
                    "schema-v1 sampler checkpoints cannot resume a distributed run")
            state = dict(state, schema_version=2, world_size=1)
        expected = self.state_dict()
        for key in ("schema_version", "lo", "end", "window_size", "seed",
                    "n_windows", "world_size"):
            if int(state[key]) != int(expected[key]):
                raise ValueError(
                    f"training sampler mismatch for {key}: {state[key]} != {expected[key]}")
        cursor = int(state["cursor"])
        if not 0 <= cursor <= self.rank_windows:
            raise ValueError(f"invalid sampler cursor {cursor}")
        self.cursor = cursor
