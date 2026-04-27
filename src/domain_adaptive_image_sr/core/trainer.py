"""PyTorch Lightning training loop for super-resolution experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from domain_adaptive_image_sr.core.evaluator import SuperResolutionEvaluator

Batch = Mapping[str, Any]


@dataclass(frozen=True)
class VisualQASample:
    """Detached tensors needed to build one visual QA panel.

    Attributes:
        sample_id: Stable sample identifier.
        lr: Low-resolution tensor with shape ``[C, H, W]``.
        baseline: Baseline SR tensor with shape ``[C, H, W]``.
        adaptation: Adapted SR tensor with shape ``[C, H, W]``.
        hr: High-resolution tensor with shape ``[C, H, W]``.
    """

    sample_id: str
    lr: torch.Tensor
    baseline: torch.Tensor
    adaptation: torch.Tensor
    hr: torch.Tensor


class SuperResolutionLightningModule(L.LightningModule):
    """LightningModule that owns training, validation, and test logic.

    Args:
        config: Root Hydra config.
        model: Model created by ``models.factory.get_model(config)``.
        evaluator: Evaluator component created from ``config.metrics``.
    """

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        evaluator: SuperResolutionEvaluator,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.evaluator = evaluator
        self.loss_config = config.loss
        self.visual_qa_samples: dict[str, list[VisualQASample]] = {
            "val": [],
            "test": [],
        }
        self.save_hyperparameters(
            {"config": OmegaConf.to_container(config, resolve=True)},
            ignore=["model", "evaluator"],
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """Run SR inference.

        Args:
            lr: Low-resolution batch with shape ``[B, C, H, W]``.

        Returns:
            Super-resolved batch with shape ``[B, C, H * scale, W * scale]``.
        """

        return self.model(lr)  # [B, C, H * scale, W * scale]

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Run one supervised training step."""

        lr, hr = _extract_supervised_tensors(batch)
        sr = self(lr)  # [B, C, H_hr, W_hr]
        loss = self._compute_loss(sr, hr)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=lr.shape[0],
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor | None:
        """Run one validation step and update evaluator metrics."""

        lr = _extract_tensor(batch, "lr")
        sr = self(lr)  # [B, C, H_sr, W_sr]
        if not _batch_has_hr(batch):
            return None

        hr = _extract_tensor(batch, "hr")
        loss = self._compute_loss(sr, hr)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=lr.shape[0],
        )
        self.evaluator.update("val", sr.detach(), hr.detach())
        self._collect_visual_qa_samples("val", batch, sr)
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor | None:
        """Run one test step and update evaluator metrics."""

        lr = _extract_tensor(batch, "lr")
        sr = self(lr)  # [B, C, H_sr, W_sr]
        if not _batch_has_hr(batch):
            return None

        hr = _extract_tensor(batch, "hr")
        loss = self._compute_loss(sr, hr)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=lr.shape[0],
        )
        self.evaluator.update("test", sr.detach(), hr.detach())
        self._collect_visual_qa_samples("test", batch, sr)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset validation metric and visual QA state."""

        self.evaluator.reset("val")
        self.visual_qa_samples["val"].clear()

    def on_test_epoch_start(self) -> None:
        """Reset test metric and visual QA state."""

        self.evaluator.reset("test")
        self.visual_qa_samples["test"].clear()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Adapt channel dimensions if loading 3-channel weights into 1-channel models."""

        state_dict = checkpoint.get("state_dict", {})
        current_state = self.state_dict()

        first_weight = "model.backbone.conv_first.weight"
        if first_weight in state_dict and first_weight in current_state:
            pt_w = state_dict[first_weight]
            tgt_w = current_state[first_weight]
            if pt_w.shape[1] == 3 and tgt_w.shape[1] == 1:
                state_dict[first_weight] = pt_w.sum(dim=1, keepdim=True)

        last_weight = "model.backbone.conv_last.weight"
        if last_weight in state_dict and last_weight in current_state:
            pt_w = state_dict[last_weight]
            tgt_w = current_state[last_weight]
            if pt_w.shape[0] == 3 and tgt_w.shape[0] == 1:
                state_dict[last_weight] = pt_w.mean(dim=0, keepdim=True)

        last_bias = "model.backbone.conv_last.bias"
        if last_bias in state_dict and last_bias in current_state:
            pt_b = state_dict[last_bias]
            tgt_b = current_state[last_bias]
            if pt_b.shape[0] == 3 and tgt_b.shape[0] == 1:
                state_dict[last_bias] = pt_b.mean(dim=0, keepdim=True)

    def on_validation_epoch_end(self) -> None:
        """Log and export validation metrics."""

        self._log_stage_metrics("val")
        self._export_metrics()

    def on_test_epoch_end(self) -> None:
        """Log and export test metrics."""

        self._log_stage_metrics("test")
        self._export_metrics()

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        """Create optimizer and optional LR scheduler from Hydra config."""

        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found for optimizer setup.")

        optimizer = _build_optimizer(self.config.optimizer, trainable_parameters)
        scheduler_config = _get_optional_config_node(self.config, "scheduler")
        scheduler_enabled = scheduler_config is not None and bool(
            scheduler_config.get("enabled", True)
        )
        if not scheduler_enabled:
            return optimizer

        assert scheduler_config is not None
        scheduler = _build_scheduler(scheduler_config, optimizer)
        lightning_scheduler = _build_lightning_scheduler_config(
            scheduler_config,
            scheduler,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lightning_scheduler,
        }

    def pop_visual_qa_samples(self, stage: str) -> list[VisualQASample]:
        """Return and clear collected visual QA samples for a stage.

        Args:
            stage: Stage name, for example ``"val"`` or ``"test"``.

        Returns:
            Collected visual QA samples.
        """

        normalized_stage = stage.strip().lower()
        samples = list(self.visual_qa_samples.get(normalized_stage, []))
        self.visual_qa_samples.setdefault(normalized_stage, []).clear()
        return samples

    def _compute_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute configured pixel loss."""

        _validate_sr_hr_shapes(sr, hr)
        loss_type = str(self.loss_config.type).strip().lower()
        reduction = str(self.loss_config.reduction)
        if loss_type in {"l1", "mae"}:
            loss = F.l1_loss(sr, hr, reduction=reduction)
        elif loss_type in {"mse", "l2"}:
            loss = F.mse_loss(sr, hr, reduction=reduction)
        elif loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(sr, hr, reduction=reduction)
        elif loss_type == "charbonnier":
            diff = sr - hr  # [B, C, H, W]
            eps = float(self.loss_config.charbonnier_eps)
            pixel_loss = torch.sqrt(diff * diff + eps * eps)  # [B, C, H, W]
            loss = _reduce_loss(pixel_loss, reduction)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_config.type}.")
        return loss * float(self.loss_config.weight)

    def _log_stage_metrics(self, stage: str) -> None:
        """Compute and log evaluator metrics for one stage."""

        metrics = self.evaluator.compute(stage)[stage]
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{stage}/{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=metric_name in {"psnr", "ssim"},
                sync_dist=True,
            )

    def _export_metrics(self) -> None:
        """Export evaluator metrics when an output directory is configured."""

        output_dir = _resolve_output_dir(self.config)
        if output_dir is None:
            return
        self.evaluator.export_metrics(output_dir)

    def _collect_visual_qa_samples(
        self,
        stage: str,
        batch: Batch,
        sr: torch.Tensor,
    ) -> None:
        """Collect detached samples for callback-driven visual QA export."""

        if not _visual_qa_enabled(self.config):
            return
        if not _batch_has_hr(batch):
            return

        stage_samples = self.visual_qa_samples.setdefault(stage, [])
        max_samples = _visual_qa_sample_limit(self.config)
        if len(stage_samples) >= max_samples:
            return

        lr = _extract_tensor(batch, "lr")
        hr = _extract_tensor(batch, "hr")
        baseline = _extract_optional_baseline(batch, sr)
        image_ids = _extract_image_ids(batch, batch_size=lr.shape[0])
        remaining = max_samples - len(stage_samples)
        for index in range(min(lr.shape[0], remaining)):
            stage_samples.append(
                VisualQASample(
                    sample_id=image_ids[index],
                    lr=lr[index].detach(),
                    baseline=baseline[index].detach(),
                    adaptation=sr[index].detach(),
                    hr=hr[index].detach(),
                )
            )


def build_lightning_trainer(
    config: DictConfig,
    callbacks: Sequence[L.Callback] | None = None,
) -> L.Trainer:
    """Create ``lightning.Trainer`` from ``config.trainer``.

    Args:
        config: Root Hydra config with a ``trainer`` section.
        callbacks: Optional callbacks built from ``core.events``.

    Returns:
        Configured Lightning trainer.
    """

    trainer_config = config.trainer
    kwargs = OmegaConf.to_container(trainer_config, resolve=True)
    if not isinstance(kwargs, dict):
        raise TypeError("`config.trainer` must resolve to a mapping.")
    kwargs["callbacks"] = list(callbacks or [])
    return L.Trainer(**kwargs)


def _extract_supervised_tensors(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract LR and HR tensors for supervised steps."""

    lr = _extract_tensor(batch, "lr")
    if not _batch_has_hr(batch):
        raise ValueError("Supervised training requires batch['hr'] with HR tensors.")
    hr = _extract_tensor(batch, "hr")
    return lr, hr


def _extract_tensor(batch: Batch, key: str) -> torch.Tensor:
    """Extract a tensor from a batch mapping."""

    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"`batch[{key!r}]` must be a torch.Tensor.")
    return value


def _batch_has_hr(batch: Batch) -> bool:
    """Return whether a batch contains non-empty HR tensors."""

    hr = batch.get("hr")
    return isinstance(hr, torch.Tensor) and hr.ndim == 4 and hr.numel() > 0


def _validate_sr_hr_shapes(sr: torch.Tensor, hr: torch.Tensor) -> None:
    """Validate SR and HR tensors before loss computation."""

    if sr.ndim != 4 or hr.ndim != 4:
        raise ValueError(
            "SR and HR tensors must have shape [B, C, H, W], got "
            f"sr={tuple(sr.shape)} and hr={tuple(hr.shape)}."
        )
    if sr.shape != hr.shape:
        raise ValueError(
            "SR and HR tensors must have identical shapes for loss, got "
            f"sr={tuple(sr.shape)} and hr={tuple(hr.shape)}."
        )


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply a PyTorch-style reduction to a loss tensor."""

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported loss reduction: {reduction}.")


def _visual_qa_enabled(config: DictConfig) -> bool:
    """Return whether visual QA collection is enabled."""

    metrics_enabled = bool(config.metrics.visual_qa.enabled)
    callback_enabled = bool(config.callbacks.visualization.enabled)
    return metrics_enabled and callback_enabled


def _visual_qa_sample_limit(config: DictConfig) -> int:
    """Resolve visual QA sample limit from callbacks and metrics configs."""

    callback_limit = int(config.callbacks.visualization.num_samples)
    metrics_limit = int(config.metrics.visual_qa.max_samples)
    return max(0, min(callback_limit, metrics_limit))


def _extract_optional_baseline(batch: Batch, sr: torch.Tensor) -> torch.Tensor:
    """Return a baseline SR tensor if present, otherwise use current SR."""

    for key in ("baseline", "baseline_sr", "sr_baseline"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return value  # [B, C, H, W]
    return sr.detach()  # [B, C, H, W]


def _extract_image_ids(batch: Batch, *, batch_size: int) -> list[str]:
    """Return image identifiers for a batch."""

    image_ids = batch.get("image_id")
    if isinstance(image_ids, Sequence) and not isinstance(image_ids, str):
        return [str(image_id) for image_id in image_ids]
    return [f"sample_{index}" for index in range(batch_size)]


def _resolve_output_dir(config: DictConfig) -> Path | None:
    """Resolve experiment output directory from root config."""

    output_dir = config.get("output_dir")
    if output_dir is None:
        return None
    return Path(str(output_dir))


def _import_symbol(target: str) -> Any:
    """Import a Python symbol from a dotted path."""

    module_name, separator, symbol_name = target.rpartition(".")
    if not separator:
        raise ValueError(f"Target path must include module and symbol: {target!r}.")
    module = import_module(module_name)
    return getattr(module, symbol_name)


def _build_optimizer(
    optimizer_config: DictConfig,
    trainable_parameters: Sequence[nn.Parameter],
) -> torch.optim.Optimizer:
    """Create an optimizer from ``config.optimizer``."""

    optimizer_class = _import_symbol(str(optimizer_config.target))
    kwargs = _filter_constructor_kwargs(
        optimizer_class,
        OmegaConf.to_container(optimizer_config, resolve=True),
        metadata_keys=("target", "name", "params"),
    )
    if "betas" in kwargs:
        kwargs["betas"] = tuple(float(value) for value in kwargs["betas"])
    return optimizer_class(trainable_parameters, **kwargs)


def _build_scheduler(
    scheduler_config: DictConfig,
    optimizer: torch.optim.Optimizer,
) -> object:
    """Create an LR scheduler from ``config.scheduler``."""

    scheduler_class = _import_symbol(str(scheduler_config.target))
    kwargs = _filter_constructor_kwargs(
        scheduler_class,
        OmegaConf.to_container(scheduler_config, resolve=True),
        metadata_keys=(
            "target",
            "name",
            "enabled",
            "optimizer",
            "interval",
            "frequency",
            "monitor",
            "strict",
        ),
    )
    return scheduler_class(optimizer, **kwargs)


def _build_lightning_scheduler_config(
    scheduler_config: DictConfig,
    scheduler: object,
) -> dict[str, object]:
    """Create Lightning's ``lr_scheduler`` config mapping."""

    lightning_config: dict[str, object] = {"scheduler": scheduler}
    if "interval" in scheduler_config:
        lightning_config["interval"] = str(scheduler_config.interval)
    if "frequency" in scheduler_config:
        lightning_config["frequency"] = int(scheduler_config.frequency)
    if "monitor" in scheduler_config and scheduler_config.monitor is not None:
        lightning_config["monitor"] = str(scheduler_config.monitor)
    if "strict" in scheduler_config:
        lightning_config["strict"] = bool(scheduler_config.strict)
    if "name" in scheduler_config and scheduler_config.name is not None:
        lightning_config["name"] = str(scheduler_config.name)
    return lightning_config


def _get_optional_config_node(config: DictConfig, key: str) -> DictConfig | None:
    """Return an optional DictConfig node from the root config."""

    node = config.get(key)
    if node is None:
        return None
    if not isinstance(node, DictConfig):
        raise TypeError(f"`config.{key}` must be a DictConfig when provided.")
    return node


def _filter_constructor_kwargs(
    target: Any,
    raw_kwargs: object,
    *,
    metadata_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Filter config values by a callable constructor signature."""

    if not isinstance(raw_kwargs, Mapping):
        raise TypeError("Constructor config must resolve to a mapping.")

    constructor_signature = signature(target)
    parameters = constructor_signature.parameters
    accepts_kwargs = any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    filtered: dict[str, Any] = {}
    for key, value in raw_kwargs.items():
        if key in metadata_keys:
            continue
        if accepts_kwargs or key in parameters:
            filtered[str(key)] = value
    return filtered


__all__ = [
    "SuperResolutionLightningModule",
    "VisualQASample",
    "build_lightning_trainer",
]
