"""Shared utilities for Mosaic runner and context building."""
from __future__ import annotations

import time
from typing import Dict, List

from mosaic.common import LossTerm, LinearCombination
from mosaic.proteinmpnn.mpnn import ProteinMPNN

Loss = LossTerm | LinearCombination


def log(message: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[mosaic_runner][{now}]", message)


def sum_losses(*losses: Loss) -> Loss:
    result = losses[0]
    for loss in losses[1:]:
        result = result + loss
    return result


def schedule_value(value, stage_idx: int = 0):
    if isinstance(value, list):
        return value[stage_idx]
    return value


def is_scheduled(value) -> bool:
    return isinstance(value, list) and not all(v == value[0] for v in value)


_mpnn_model_cache: Dict[str, ProteinMPNN] = {}
def load_mpnn(variant: str = "default", backbone_noise: float = 0.0) -> ProteinMPNN:
    cache_key = f"{variant}_noise{backbone_noise}"
    if cache_key not in _mpnn_model_cache:
        from mosaic.proteinmpnn.mpnn import load_mpnn, load_mpnn_sol, load_abmpnn
        loaders = {"default": load_mpnn, "soluble": load_mpnn_sol, "abmpnn": load_abmpnn}
        if variant not in loaders:
            raise ValueError(f"Unknown MPNN variant: {variant}. Available: {sorted(loaders)}")
        log(f"Loading ProteinMPNN variant: {variant} (backbone_noise={backbone_noise})")
        _mpnn_model_cache[cache_key] = loaders[variant](backbone_noise=backbone_noise)
    return _mpnn_model_cache[cache_key]
