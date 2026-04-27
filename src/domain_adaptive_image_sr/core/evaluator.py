"""Evaluation component for super-resolution metrics."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from domain_adaptive_image_sr.utils.reporting import export_metrics_csv


class SuperResolutionEvaluator(nn.Module):
    """Aggregate SR metrics by stage with optional border cropping.

    Args:
        config: Hydra metrics config.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.crop_border = int(config.crop_border)
        self.stages = tuple(str(stage).strip().lower() for stage in config.stages)
        if not self.stages:
            raise ValueError("`metrics.stages` must contain at least one stage.")

        metric_factories = _build_metric_factories(config)
        if not metric_factories:
            raise ValueError("At least one metric must be enabled.")

        self._metric_names = tuple(metric_factories)
        self._stage_update_counts = {stage: 0 for stage in self.stages}
        self.metrics_by_stage = nn.ModuleDict(
            {
                stage: nn.ModuleDict(
                    {
                        metric_name: metric_factory()
                        for metric_name, metric_factory in metric_factories.items()
                    }
                )
                for stage in self.stages
            }
        )

    def update(self, stage: str, sr: torch.Tensor, hr: torch.Tensor) -> None:
        """Update metrics for one stage.

        Args:
            stage: Evaluation stage, for example ``"val"`` or ``"test"``.
            sr: Super-resolved batch with shape ``[B, C, H, W]``.
            hr: High-resolution target batch with shape ``[B, C, H, W]``.
        """

        stage_metrics = self._get_stage_metrics(stage)

        # Torchmetrics image metrics expect normalized image tensors.
        sr = sr.clamp(0.0, 1.0)
        hr = hr.clamp(0.0, 1.0)

        sr_cropped, hr_cropped = crop_border(sr, hr, self.crop_border)
        for metric_name, metric in stage_metrics.items():
            if metric_name == "lpips" and sr_cropped.shape[1] == 1:
                # LPIPS has no native single-channel mode.
                metric.update(
                    sr_cropped.repeat(1, 3, 1, 1), hr_cropped.repeat(1, 3, 1, 1)
                )
            else:
                metric.update(sr_cropped, hr_cropped)
        self._stage_update_counts[stage.strip().lower()] += 1

    def compute(self, stage: str | None = None) -> dict[str, dict[str, float]]:
        """Compute aggregated metrics.

        Args:
            stage: Optional single stage to compute. When omitted, all stages
                are computed.

        Returns:
            Mapping from stage names to metric values.
        """

        stages = (stage,) if stage is not None else self.stages
        return {stage_name: self._compute_stage(stage_name) for stage_name in stages}

    def reset(self, stage: str | None = None) -> None:
        """Reset metric states.

        Args:
            stage: Optional stage to reset. When omitted, all stages are reset.
        """

        stages = (stage,) if stage is not None else self.stages
        for stage_name in stages:
            normalized_stage = stage_name.strip().lower()
            for metric in self._get_stage_metrics(normalized_stage).values():
                metric.reset()
            self._stage_update_counts[normalized_stage] = 0

    def export_metrics(
        self,
        output_dir: str | Path,
        filename: str | None = None,
    ) -> Path:
        """Export computed metrics to CSV.

        Args:
            output_dir: Experiment output directory.
            filename: Optional CSV filename. Defaults to config reporting name.

        Returns:
            Path to the written CSV file.
        """

        report_filename = filename or str(self.config.reporting.filename)
        return export_metrics_csv(self.compute(), output_dir, report_filename)

    def _get_stage_metrics(self, stage: str) -> nn.ModuleDict:
        """Return metrics for a configured stage."""

        normalized_stage = stage.strip().lower()
        if normalized_stage not in self.metrics_by_stage:
            available = ", ".join(self.metrics_by_stage.keys())
            raise KeyError(f"Unknown metric stage {stage!r}. Available: {available}.")
        return self.metrics_by_stage[normalized_stage]

    def _compute_stage(self, stage: str) -> dict[str, float]:
        """Compute one stage, returning NaN values if it has no updates."""

        normalized_stage = stage.strip().lower()
        stage_metrics = self._get_stage_metrics(normalized_stage)
        if self._stage_update_counts[normalized_stage] == 0:
            return {metric_name: float("nan") for metric_name in self._metric_names}

        return {
            metric_name: _metric_value_to_float(metric.compute())
            for metric_name, metric in stage_metrics.items()
        }


def crop_border(
    sr: torch.Tensor,
    hr: torch.Tensor,
    border: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop metric tensors by a border width.

    Args:
        sr: Super-resolved batch with shape ``[B, C, H, W]``.
        hr: High-resolution target batch with shape ``[B, C, H, W]``.
        border: Number of pixels to crop on each side.

    Returns:
        Tuple of cropped ``(sr, hr)`` tensors with shape
        ``[B, C, H - 2 * border, W - 2 * border]``.
    """

    _validate_metric_tensors(sr, hr)
    if border < 0:
        raise ValueError(f"`crop_border` must be >= 0, got {border}.")
    if border == 0:
        return sr, hr
    if sr.shape[-2] <= 2 * border or sr.shape[-1] <= 2 * border:
        raise ValueError(
            "`crop_border` is too large for tensor shape "
            f"{tuple(sr.shape)} and border={border}."
        )

    sr_cropped = sr[..., border:-border, border:-border] # [B, C, Hc, Wc]
    hr_cropped = hr[..., border:-border, border:-border]
    return sr_cropped, hr_cropped


def _build_metric_factories(config: DictConfig) -> dict[str, Callable[[], Metric]]:
    """Create metric factories from config."""

    metric_factories: dict[str, Callable[[], Metric]] = {}

    if bool(config.psnr.enabled):
        psnr = PeakSignalNoiseRatio(data_range=float(config.psnr.data_range))
        metric_factories["psnr"] = lambda psnr=psnr: deepcopy(psnr)

    if bool(config.ssim.enabled):
        ssim = StructuralSimilarityIndexMeasure(
            data_range=float(config.ssim.data_range)
        )
        metric_factories["ssim"] = lambda ssim=ssim: deepcopy(ssim)

    if bool(config.lpips.enabled):
        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=str(config.lpips.net_type),
            normalize=bool(config.lpips.normalize),
        )
        metric_factories["lpips"] = lambda lpips=lpips: deepcopy(lpips)

    return metric_factories


def _validate_metric_tensors(sr: torch.Tensor, hr: torch.Tensor) -> None:
    """Validate SR and HR tensors for metric computation."""

    if sr.ndim != 4 or hr.ndim != 4:
        raise ValueError(
            "Metric tensors must have shape [B, C, H, W], got "
            f"sr={tuple(sr.shape)} and hr={tuple(hr.shape)}."
        )
    if sr.shape != hr.shape:
        raise ValueError(
            "SR and HR tensors must have identical shapes for metrics, got "
            f"sr={tuple(sr.shape)} and hr={tuple(hr.shape)}."
        )


def _metric_value_to_float(value: torch.Tensor | float) -> float:
    """Convert a torchmetrics result into a Python float."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    return float(value)


__all__ = ["SuperResolutionEvaluator", "crop_border"]
