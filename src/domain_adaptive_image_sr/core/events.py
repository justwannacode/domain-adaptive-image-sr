"""Lightning callbacks for experiment events."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig

from domain_adaptive_image_sr.core.trainer import VisualQASample
from domain_adaptive_image_sr.metrics.visual_qa import save_visual_qa_panel


class VisualQACallback(Callback):
    """Save intermediate SR visual QA panels from LightningModule samples.

    Args:
        config: Root Hydra config.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.callback_config = config.callbacks.visualization
        self.output_dir = Path(str(self.callback_config.output_dir))
        self.every_n_epochs = int(self.callback_config.every_n_epochs)
        self.filename_prefix = str(self.callback_config.filename_prefix)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Save validation visual QA panels at configured epoch intervals."""

        if trainer.sanity_checking:
            return
        if not bool(self.callback_config.enabled):
            return
        if not _should_run_for_epoch(trainer.current_epoch, self.every_n_epochs):
            return
        self._save_stage_samples(trainer, pl_module, stage="val")

    def on_test_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Save test visual QA panels when enabled."""

        if not bool(self.callback_config.enabled):
            return
        if not bool(self.callback_config.save_test):
            return
        self._save_stage_samples(trainer, pl_module, stage="test")

    def _save_stage_samples(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *,
        stage: str,
    ) -> None:
        """Pop visual samples from a module and persist them as PNG panels."""

        pop_samples = getattr(pl_module, "pop_visual_qa_samples", None)
        if pop_samples is None:
            return

        samples = pop_samples(stage)
        if not isinstance(samples, Sequence):
            return

        stage_dir = self.output_dir / stage / f"epoch_{trainer.current_epoch:04d}"
        for index, sample in enumerate(samples):
            if not isinstance(sample, VisualQASample):
                continue
            sample_id = (
                f"{self.filename_prefix}_{stage}_{trainer.current_epoch:04d}_"
                f"{index:03d}_{sample.sample_id}"
            )
            save_visual_qa_panel(
                lr=sample.lr,
                baseline=sample.baseline,
                adaptation=sample.adaptation,
                hr=sample.hr,
                config=self.config.metrics,
                output_dir=stage_dir,
                sample_id=sample_id,
            )


def build_callbacks(config: DictConfig) -> list[Callback]:
    """Build Lightning callbacks from ``config.callbacks``.

    Args:
        config: Root Hydra config.

    Returns:
        List of configured Lightning callbacks.
    """

    callbacks: list[Callback] = []
    callback_config = config.callbacks

    if bool(callback_config.checkpoint.enabled) and _checkpointing_enabled(config):
        callbacks.append(_build_checkpoint_callback(config))

    if bool(callback_config.lr_monitor.enabled):
        callbacks.append(
            LearningRateMonitor(
                logging_interval=str(callback_config.lr_monitor.logging_interval),
            )
        )

    if bool(callback_config.visualization.enabled):
        callbacks.append(VisualQACallback(config))

    return callbacks


def _build_checkpoint_callback(config: DictConfig) -> ModelCheckpoint:
    """Create a ModelCheckpoint callback from Hydra config."""

    checkpoint_config = config.callbacks.checkpoint
    return ModelCheckpoint(
        dirpath=str(checkpoint_config.dirpath),
        filename=str(checkpoint_config.filename),
        monitor=str(checkpoint_config.monitor),
        mode=str(checkpoint_config.mode),
        save_top_k=int(checkpoint_config.save_top_k),
        save_last=bool(checkpoint_config.save_last),
        every_n_epochs=int(checkpoint_config.every_n_epochs),
        save_weights_only=bool(checkpoint_config.save_weights_only),
        auto_insert_metric_name=bool(checkpoint_config.auto_insert_metric_name),
    )


def _checkpointing_enabled(config: DictConfig) -> bool:
    """Return whether Lightning checkpointing is enabled in trainer config."""

    trainer_config = config.get("trainer")
    if trainer_config is None:
        return True
    return bool(trainer_config.enable_checkpointing)


def _should_run_for_epoch(current_epoch: int, every_n_epochs: int) -> bool:
    """Return whether a callback should run for the current epoch."""

    if every_n_epochs <= 0:
        raise ValueError("`every_n_epochs` must be >= 1.")
    return (current_epoch + 1) % every_n_epochs == 0


__all__ = ["VisualQACallback", "build_callbacks"]
