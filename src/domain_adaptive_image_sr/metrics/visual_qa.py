"""Helpers for qualitative visual QA of SR outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from domain_adaptive_image_sr.utils.visualization import CropRegion, draw_crops


@dataclass(frozen=True)
class VisualQAResult:
    """Result of saving one visual QA panel.

    Attributes:
        path: Output image path.
        sample_id: Stable sample identifier.
    """

    path: Path
    sample_id: str


def build_visual_qa_panel(
    lr: torch.Tensor,
    baseline: torch.Tensor,
    adaptation: torch.Tensor,
    hr: torch.Tensor,
    config: DictConfig,
) -> Image.Image:
    """Build a visual QA comparison panel from tensors.

    Args:
        lr: LR image tensor with shape ``[C, H, W]`` or ``[1, C, H, W]``.
        baseline: Baseline SR tensor with shape ``[C, H, W]`` or
            ``[1, C, H, W]``.
        adaptation: Adapted SR tensor with shape ``[C, H, W]`` or
            ``[1, C, H, W]``.
        hr: HR tensor with shape ``[C, H, W]`` or ``[1, C, H, W]``.
        config: Metrics config containing ``visual_qa`` settings.

    Returns:
        PIL comparison panel.
    """

    qa_config = config.visual_qa
    regions = parse_crop_regions(qa_config.regions)
    border_color = tuple(int(value) for value in qa_config.crop_border_color)
    if len(border_color) != 3:
        raise ValueError("`visual_qa.crop_border_color` must contain 3 integers.")

    return draw_crops(
        lr=lr,
        baseline=baseline,
        adaptation=adaptation,
        hr=hr,
        regions=regions,
        labels=tuple(str(label) for label in qa_config.panel_labels),
        crop_padding=int(qa_config.crop_padding),
        crop_border_width=int(qa_config.crop_border_width),
        crop_border_color=border_color,
    )


def save_visual_qa_panel(
    lr: torch.Tensor,
    baseline: torch.Tensor,
    adaptation: torch.Tensor,
    hr: torch.Tensor,
    config: DictConfig,
    output_dir: str | Path,
    sample_id: str,
) -> VisualQAResult:
    """Build and save one visual QA comparison panel.

    Args:
        lr: LR image tensor with shape ``[C, H, W]`` or ``[1, C, H, W]``.
        baseline: Baseline SR tensor with shape ``[C, H, W]`` or
            ``[1, C, H, W]``.
        adaptation: Adapted SR tensor with shape ``[C, H, W]`` or
            ``[1, C, H, W]``.
        hr: HR tensor with shape ``[C, H, W]`` or ``[1, C, H, W]``.
        config: Metrics config containing ``visual_qa`` settings.
        output_dir: Directory where the panel should be written.
        sample_id: Stable sample identifier used in the filename.

    Returns:
        Saved visual QA result metadata.
    """

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    safe_sample_id = _sanitize_sample_id(sample_id)
    panel = build_visual_qa_panel(lr, baseline, adaptation, hr, config)
    panel_path = output_path / f"{safe_sample_id}.png"
    panel.save(panel_path)
    return VisualQAResult(path=panel_path, sample_id=safe_sample_id)


def parse_crop_regions(
    regions: Sequence[CropRegion | Mapping[str, object] | DictConfig],
) -> tuple[CropRegion, ...]:
    """Parse visual QA crop regions from config-like objects.

    Args:
        regions: Region sequence from Hydra config.

    Returns:
        Tuple of normalized crop regions.
    """

    parsed_regions: list[CropRegion] = []
    for region in regions:
        if isinstance(region, CropRegion):
            parsed_regions.append(region)
            continue
        if isinstance(region, DictConfig):
            region = OmegaConf.to_container(region, resolve=True)
        if not isinstance(region, Mapping):
            raise TypeError(f"Unsupported crop region type: {type(region)!r}.")
        parsed_regions.append(
            CropRegion(
                name=str(region.get("name", "crop")),
                x=float(region["x"]),
                y=float(region["y"]),
                width=float(region["width"]),
                height=float(region["height"]),
                relative=bool(region.get("relative", False)),
            )
        )
    return tuple(parsed_regions)


def should_save_visual_sample(index: int, config: DictConfig) -> bool:
    """Return whether a sample index should be saved for visual QA.

    Args:
        index: Zero-based sample index.
        config: Metrics config containing ``visual_qa`` settings.

    Returns:
        ``True`` if the sample should be saved.
    """

    qa_config = config.visual_qa
    return bool(qa_config.enabled) and index < int(qa_config.max_samples)


def _sanitize_sample_id(sample_id: str) -> str:
    """Convert a sample identifier into a safe filename stem."""

    safe_chars = [
        char if char.isalnum() or char in {"-", "_"} else "_" for char in sample_id
    ]
    safe_id = "".join(safe_chars).strip("_")
    return safe_id or "sample"


__all__ = [
    "VisualQAResult",
    "build_visual_qa_panel",
    "parse_crop_regions",
    "save_visual_qa_panel",
    "should_save_visual_sample",
]
