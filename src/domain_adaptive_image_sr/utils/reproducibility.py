"""Utilities for reproducible experiment execution."""

from __future__ import annotations

from typing import Any

from lightning import seed_everything
from omegaconf import DictConfig


def seed_experiment(seed: int) -> int:
    """Seed all supported random number generators for an experiment.

    Args:
        seed: Global random seed for Python, NumPy, PyTorch, and dataloader workers.

    Returns:
        The seed value returned by Lightning after initialization.
    """

    return seed_everything(seed, workers=True)


def prepare_trainer_determinism_flags(trainer_config: DictConfig) -> dict[str, Any]:
    """Build deterministic trainer flags from Hydra trainer config.

    Args:
        trainer_config: Hydra config section available as ``config.trainer``.

    Returns:
        Dictionary with Trainer keyword arguments related to deterministic
        execution.
    """

    deterministic = bool(trainer_config.deterministic)
    benchmark = bool(trainer_config.benchmark)

    if deterministic:
        benchmark = False

    return {
        "deterministic": deterministic,
        "benchmark": benchmark,
    }
