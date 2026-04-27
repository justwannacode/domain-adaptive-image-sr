"""Hydra entry point for evaluating a trained SR checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from omegaconf import DictConfig
from rich.console import Console
import torch

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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Evaluate a trained checkpoint with the project datamodule and evaluator.

    Args:
        config: Root Hydra config loaded from ``configs/config.yaml``.
    """

    seed = seed_experiment(int(config.seed))
    determinism_flags = prepare_trainer_determinism_flags(config.trainer)
    config.trainer.deterministic = determinism_flags["deterministic"]
    config.trainer.benchmark = determinism_flags["benchmark"]

    checkpoint_path = _resolve_checkpoint_path(config)
    stage = _get_optional_string(config, ("evaluation", "stage"), "test")
    stage = _get_optional_string(config, ("stage",), stage)
    if stage not in {"validate", "val", "test", "predict"}:
        raise ValueError(
            "Evaluation stage must be one of: validate, val, test, predict."
        )

    console.print(f"Seeded evaluation with seed={seed}.")

    datamodule = SuperResolutionDataModule(config)
    model = get_model(config)
    evaluator = SuperResolutionEvaluator(config.metrics)

    if checkpoint_path == "null" or checkpoint_path == "base":
        console.print("Evaluating base model without checkpoint.")
        lightning_module = SuperResolutionLightningModule(
            config=config,
            model=model,
            evaluator=evaluator
        )
    else:
        console.print(f"Loading checkpoint: {checkpoint_path}")
        lightning_module = SuperResolutionLightningModule.load_from_checkpoint(
            checkpoint_path,
            config=config,
            model=model,
            evaluator=evaluator,
        )

    callbacks = build_callbacks(config)
    trainer = build_lightning_trainer(config, callbacks)

    if stage in {"validate", "val"}:
        console.print("Running validation.")
        trainer.validate(lightning_module, datamodule=datamodule)
    elif stage == "predict":
        console.print("Running prediction.")
        _run_prediction(config, trainer, datamodule, lightning_module)
    else:
        console.print("Running test.")
        trainer.test(lightning_module, datamodule=datamodule)


class _PredictionModule(L.LightningModule):
    """Small Lightning wrapper for checkpoint inference batches."""

    def __init__(self, module: SuperResolutionLightningModule) -> None:
        super().__init__()
        self.module = module

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, object]:
        """Run one prediction batch.

        Args:
            batch: Datamodule batch containing ``lr`` and metadata.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index supplied by Lightning.

        Returns:
            Prediction payload with SR tensor and image ids.
        """

        lr = batch["lr"]
        if not isinstance(lr, torch.Tensor):
            raise TypeError("Prediction batch must contain tensor key `lr`.")
        sr = self.module(lr)  # [B, C, H * scale, W * scale]
        return {
            "batch_idx": batch_idx,
            "dataloader_idx": dataloader_idx,
            "image_id": batch.get("image_id", []),
            "sr": sr.detach(),
        }


def _run_prediction(
    config: DictConfig,
    trainer: L.Trainer,
    datamodule: SuperResolutionDataModule,
    lightning_module: SuperResolutionLightningModule,
) -> None:
    """Run checkpoint inference and save prediction tensors."""

    output_dir = _resolve_prediction_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_module = _PredictionModule(lightning_module)
    predictions = trainer.predict(
        prediction_module,
        datamodule=datamodule,
        return_predictions=True,
    )
    if predictions is None:
        console.print("No predictions were returned by Lightning.")
        return

    for index, prediction in enumerate(predictions):
        output_path = output_dir / f"batch_{index:05d}.pt"
        torch.save(prediction, output_path)
    console.print(f"Wrote {len(predictions)} prediction batches to {output_dir}.")


def _resolve_prediction_output_dir(config: DictConfig) -> Path:
    """Resolve prediction output directory from Hydra overrides."""

    output_dir = _get_optional_string(config, ("evaluation", "output_dir"), None)
    if output_dir is None:
        output_dir = str(Path(str(config.output_dir)) / "predictions")
    return Path(output_dir)


def _resolve_checkpoint_path(config: DictConfig) -> str:
    """Resolve checkpoint path from Hydra overrides."""

    checkpoint_path = _get_optional_string(
        config,
        ("evaluation", "checkpoint_path"),
        None,
    )
    checkpoint_path = _get_optional_string(
        config,
        ("checkpoint_path",),
        checkpoint_path,
    )
    if checkpoint_path is None or checkpoint_path.lower() == "base":
        return "base"
    return checkpoint_path


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
