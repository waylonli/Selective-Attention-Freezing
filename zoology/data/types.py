"""Minimal return container for the vendored MQAR generator."""

from dataclasses import dataclass
import torch


@dataclass
class DataSegment:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: dict | None = None

    def __len__(self):
        if len(self.inputs) != len(self.labels):
            raise ValueError("Input and label lengths differ")
        return len(self.inputs)
