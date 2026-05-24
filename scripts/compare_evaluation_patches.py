"""Save LR/base/checkpoint/HR comparison patches for SR evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from domain_adaptive_image_sr.core.evaluator import SuperResolutionEvaluator
from domain_adaptive_image_sr.core.trainer import SuperResolutionLightningModule
from domain_adaptive_image_sr.data.datamodule import SuperResolutionDataModule
from domain_adaptive_image_sr.models.factory import get_model
from domain_adaptive_image_sr.utils.reproducibility import seed_experiment
from domain_adaptive_image_sr.utils.visualization import tensor_to_pil_image

console = Console()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Create visual comparison patches for a configured evaluation setup."""

    OmegaConf.set_struct(config, False)
    seed = seed_experiment(int(config.seed))
    _force_patch_config(config, patch_size=_comparison_patch_size(config))

    output_dir = _comparison_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Seeded comparison with seed={seed}.")
    console.print(f"Writing comparison patches to: {output_dir}")

    datamodule = SuperResolutionDataModule(config)
    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    device = _resolve_device(config)

    import copy
    base_config = copy.deepcopy(config)
    base_upsampler = _get_optional_string(config, ("base_upsampler",))
    if base_upsampler is not None:
        base_config.model.upsampler = base_upsampler
        base_config.model.strict_load = False
        console.print(f"Overriding initialized model upsampler to: {base_upsampler}")

    initialized_model = get_model(base_config).to(device).eval()
    configured_model = _build_configured_model(config, device).eval()

    max_samples = _comparison_num_samples(config)
    saved = 0
    with torch.inference_mode():
        for batch in dataloader:
            tensor_batch = _move_tensor_batch(batch, device)
            lr = _require_tensor(tensor_batch, "lr")
            hr = _require_tensor(tensor_batch, "hr")
            if hr.ndim != 4 or hr.numel() == 0:
                raise ValueError("Comparison requires paired test data with HR tensors.")

            # Apply absolute patch crop logic to center or offset ROI
            target_h, target_w = hr.shape[-2:]
            crop_size = _comparison_patch_size(config)

            top, left = _comparison_patch_position(config, target_h, target_w, crop_size)

            hr = hr[:, :, top:top+crop_size, left:left+crop_size]

            scale = config.data.get("scale", 2)
            lr_top, lr_left = top // scale, left // scale
            lr_crop_size = crop_size // scale

            lr = lr[:, :, lr_top:lr_top+lr_crop_size, lr_left:lr_left+lr_crop_size]

            initialized_sr = initialized_model(lr).clamp(0.0, 1.0)
            configured_sr = configured_model(lr).clamp(0.0, 1.0)
            image_ids = _extract_image_ids(batch, batch_size=lr.shape[0])

            for index in range(lr.shape[0]):
                if saved >= max_samples:
                    console.print(f"Saved {saved} comparison patch panel(s).")
                    return

                panel = _build_panel(
                    lr=lr[index],
                    initialized_sr=initialized_sr[index],
                    configured_sr=configured_sr[index],
                    hr=hr[index],
                )
                sample_id = _sanitize_filename(image_ids[index])
                output_path = output_dir / f"{saved:03d}_{sample_id}.png"
                panel.save(output_path)
                console.print(f"Wrote {output_path}")
                saved += 1

    console.print(f"Saved {saved} comparison patch panel(s).")


def _build_configured_model(config: DictConfig, device: torch.device) -> torch.nn.Module:
    """Build the model selected by config and optionally load Lightning checkpoint."""

    checkpoint_path = _resolve_checkpoint_path(config)
    if checkpoint_path in {"base", "null"}:
        console.print("Configured model uses base weights only.")
        return get_model(config).to(device)

    console.print(f"Loading configured model from checkpoint: {checkpoint_path}")
    model = get_model(config)
    evaluator = SuperResolutionEvaluator(config.metrics)
    lightning_module = SuperResolutionLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        model=model,
        evaluator=evaluator,
        map_location=device,
    )
    return lightning_module.model.to(device)


def _force_patch_config(config: DictConfig, *, patch_size: int) -> None:
    """Force deterministic test patches with the requested HR patch size."""

    config.data.patch.enabled = False
    config.data.patch.size = int(patch_size)
    config.data.patch.random = False
    config.data.batch_size = int(config.data.get("batch_size", 1))


def _build_panel(
    *,
    lr: torch.Tensor,
    initialized_sr: torch.Tensor,
    configured_sr: torch.Tensor,
    hr: torch.Tensor,
) -> Image.Image:
    """Build a four-column visual comparison panel."""

    target_h, target_w = hr.shape[-2:]
    lr_display = _resize_to_hw(lr, height=target_h, width=target_w)
    initialized_sr = _resize_to_hw(initialized_sr, height=target_h, width=target_w)
    configured_sr = _resize_to_hw(configured_sr, height=target_h, width=target_w)

    images = [
        tensor_to_pil_image(lr_display.detach().cpu()),
        tensor_to_pil_image(initialized_sr.detach().cpu()),
        tensor_to_pil_image(configured_sr.detach().cpu()),
        tensor_to_pil_image(hr.detach().cpu()),
    ]
    labels = ("LR", "Initialized", "Configured", "HR")
    return _compose_labeled_panel(images, labels)


def _compose_labeled_panel(
    images: Sequence[Image.Image],
    labels: Sequence[str],
) -> Image.Image:
    """Compose images into one row with labels."""

    if len(images) != len(labels):
        raise ValueError("Image and label counts must match.")

    padding = 8
    label_height = 18
    font = ImageFont.load_default()
    image_width = max(image.width for image in images)
    image_height = max(image.height for image in images)
    width = len(images) * image_width + (len(images) + 1) * padding
    height = label_height + image_height + 2 * padding

    panel = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)
    x = padding
    for label, image in zip(labels, images, strict=True):
        draw.text((x, padding), label, fill=(0, 0, 0), font=font)
        panel.paste(image, (x, padding + label_height))
        x += image_width + padding
    return panel


def _resize_to_hw(image: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Resize CHW image tensor to the requested height and width."""

    if image.shape[-2:] == (height, width):
        return image
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _resolve_device(config: DictConfig) -> torch.device:
    """Resolve a torch device from trainer config."""

    accelerator = str(config.trainer.get("accelerator", "cpu")).lower()
    if accelerator in {"gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _comparison_patch_size(config: DictConfig) -> int:
    """Return configured HR patch size for comparison panels."""

    node = _get_optional_node(config, "comparison")
    if node is None:
        return 128
    return int(node.get("patch_size", 128))


def _comparison_num_samples(config: DictConfig) -> int:
    """Return the number of comparison panels to save."""

    node = _get_optional_node(config, "comparison")
    if node is None:
        return 4
    return int(node.get("num_samples", 4))


def _comparison_output_dir(config: DictConfig) -> Path:
    """Return output directory for comparison panels."""

    node = _get_optional_node(config, "comparison")
    if node is not None and node.get("output_dir") is not None:
        return Path(str(node.output_dir))
    return Path(str(config.output_dir)) / "comparison_patches"


def _comparison_patch_position(config: DictConfig, height: int, width: int, crop_size: int) -> tuple[int, int]:
    """Calculate the top-left coordinate for an extraction patch based on config offset."""
    node = _get_optional_node(config, "comparison")
    x_offset, y_offset = 0.5, 0.5
    if node is not None:
        x_offset = float(node.get("patch_x", 0.5))
        y_offset = float(node.get("patch_y", 0.5))

    max_y = height - crop_size
    max_x = width - crop_size

    top = int(max_y * y_offset)
    left = int(max_x * x_offset)

    return max(0, min(top, max_y)), max(0, min(left, max_x))


def _resolve_checkpoint_path(config: DictConfig) -> str:
    """Resolve checkpoint path using the same convention as run_evaluation."""

    checkpoint_path = _get_optional_string(config, ("evaluation", "checkpoint_path"))
    checkpoint_path = _get_optional_string(config, ("checkpoint_path",), checkpoint_path)
    if checkpoint_path is None or checkpoint_path.lower() == "base":
        return "base"
    return checkpoint_path


def _move_tensor_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move tensor values in a batch mapping to device."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    """Return a required tensor from a batch."""

    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch key {key!r} must contain a tensor.")
    return value


def _extract_image_ids(batch: Mapping[str, Any], *, batch_size: int) -> list[str]:
    """Extract stable image identifiers from a batch."""

    image_ids = batch.get("image_id")
    if isinstance(image_ids, Sequence) and not isinstance(image_ids, str):
        return [str(image_id) for image_id in image_ids]
    return [f"sample_{index}" for index in range(batch_size)]


def _sanitize_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""

    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return safe.strip("_") or "sample"


def _get_optional_node(config: DictConfig, key: str) -> DictConfig | None:
    """Return an optional DictConfig child node."""

    node = config.get(key)
    if node is None:
        return None
    if not isinstance(node, DictConfig):
        raise TypeError(f"config.{key} must be a DictConfig when provided.")
    return node


def _get_optional_string(
    config: DictConfig,
    path: Sequence[str],
    default: str | None = None,
) -> str | None:
    """Read an optional string from a nested config path."""

    node: object = config
    for key in path:
        if not isinstance(node, DictConfig) or key not in node:
            return default
        node = node[key]
    if node is None:
        return None
    text = str(node)
    return text if text else None


if __name__ == "__main__":
    main()
