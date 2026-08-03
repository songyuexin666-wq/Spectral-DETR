"""Checkpoint loading helpers shared by training and inference entry points."""

from typing import Dict, List, Tuple

import torch


ShapeMismatch = Tuple[str, Tuple[int, ...], Tuple[int, ...]]


def filter_state_dict_by_shape(
    model_state: Dict[str, torch.Tensor],
    checkpoint_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[ShapeMismatch]]:
    """Drop checkpoint tensors whose shapes do not match the current model."""
    filtered: Dict[str, torch.Tensor] = {}
    dropped: List[ShapeMismatch] = []

    for name, value in checkpoint_state.items():
        reference = model_state.get(name)
        if reference is not None and tuple(reference.shape) != tuple(value.shape):
            dropped.append((name, tuple(value.shape), tuple(reference.shape)))
            continue
        filtered[name] = value

    return filtered, dropped
