"""Hydra entry point for training super-resolution experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import hydra
from omegaconf import DictConfig
from rich.console import Console

from domain_adaptive_image_sr.core.evaluator import SuperResolutionEvaluator
from domain_adaptive_image_sr.core.events import build_callbacks
from domain_adaptive_image_sr.core.trainer import (
    SuperResolutionLightningModule,
    build_lightning_trainer,
)
from domain_adaptive_image_sr.data.datamodule import SuperResolutionDataModule
from domain_adaptive_image_sr.models.factory import get_model
from domain_adaptive_image_sr.utils.reproducibility import (
    prepare_trainer_determinism_flags,
    seed_experiment,
)

console = Console()


def _patch_hydra_argparse_for_python314() -> None:
    """Patch argparse help handling for Hydra on Python 3.14."""

    original_expand_help = argparse.HelpFormatter._expand_help

    def expand_help(self: argparse.HelpFormatter, action: argparse.Action) -> str:
        """Convert Hydra lazy help objects to strings before argparse checks."""

        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return original_expand_help(self, action)

    argparse.HelpFormatter._expand_help = expand_help


_patch_hydra_argparse_for_python314()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Run model training from the project Hydra config.

    Args:
        config: Root Hydra config loaded from ``configs/config.yaml``.
    """

    seed = seed_experiment(int(config.seed))
    determinism_flags = prepare_trainer_determinism_flags(config.trainer)
    config.trainer.deterministic = determinism_flags["deterministic"]
    config.trainer.benchmark = determinism_flags["benchmark"]

    console.print(f"Seeded experiment with seed={seed}.")
    datamodule = SuperResolutionDataModule(config)
    model = get_model(config)
    evaluator = SuperResolutionEvaluator(config.metrics)
    lightning_module = SuperResolutionLightningModule(config, model, evaluator)
    callbacks = build_callbacks(config)
    trainer = build_lightning_trainer(config, callbacks)

    checkpoint_path = _get_optional_string(config, ("checkpoint_path",), None)
    console.print("Starting training.")
    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=checkpoint_path,
    )

    if _get_optional_bool(config, ("run", "validate_after_fit"), False):
        console.print("Running validation after training.")
        trainer.validate(lightning_module, datamodule=datamodule)

    if _get_optional_bool(config, ("run", "test_after_fit"), False):
        console.print("Running test after training.")
        best_checkpoint = getattr(trainer.checkpoint_callback, "best_model_path", "")
        test_checkpoint = best_checkpoint or None
        trainer.test(
            lightning_module,
            datamodule=datamodule,
            ckpt_path=test_checkpoint,
        )


def _get_optional_bool(
    config: DictConfig,
    path: Sequence[str],
    default: bool,
) -> bool:
    """Read an optional boolean from a nested Hydra config path."""

    value = _get_optional_value(config, path, default)
    return bool(value)


def _get_optional_string(
    config: DictConfig,
    path: Sequence[str],
    default: str | None,
) -> str | None:
    """Read an optional non-empty string from a nested Hydra config path."""

    value = _get_optional_value(config, path, default)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _get_optional_value(
    config: DictConfig,
    path: Sequence[str],
    default: object,
) -> object:
    """Read an optional value from a nested Hydra config path."""

    node: object = config
    for key in path:
        if not isinstance(node, DictConfig) or key not in node:
            return default
        node = node[key]
    return node


if __name__ == "__main__":
    main()
