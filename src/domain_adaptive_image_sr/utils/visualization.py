"""Visualization utilities for super-resolution qualitative assessment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class CropRegion:
    """Region of interest for visual crop comparison.

    Args:
        name: Human-readable region name.
        x: Left coordinate or relative left fraction.
        y: Top coordinate or relative top fraction.
        width: Region width in pixels or relative fraction.
        height: Region height in pixels or relative fraction.
        relative: Whether coordinates are relative fractions of image size.
    """

    name: str
    x: float
    y: float
    width: float
    height: float
    relative: bool = False


def draw_crops(
    lr: torch.Tensor,
    baseline: torch.Tensor,
    adaptation: torch.Tensor,
    hr: torch.Tensor,
    regions: Sequence[CropRegion | Mapping[str, object]],
    *,
    labels: Sequence[str] = ("LR", "Baseline", "Adaptation", "HR"),
    crop_padding: int = 4,
    crop_border_width: int = 2,
    crop_border_color: tuple[int, int, int] = (255, 64, 64),
) -> Image.Image:
    """Build a comparison panel ``[LR, Baseline, Adaptation, HR]`` with crops.

    Args:
        lr: Low-resolution image tensor with shape ``[C, H, W]`` or
            ``[1, C, H, W]``.
        baseline: Baseline SR tensor with shape ``[C, H, W]`` or
            ``[1, C, H, W]``.
        adaptation: Adapted SR tensor with shape ``[C, H, W]`` or
            ``[1, C, H, W]``.
        hr: High-resolution image tensor with shape ``[C, H, W]`` or
            ``[1, C, H, W]``.
        regions: Regions of interest in HR/SR coordinates.
        labels: Panel labels for the four columns.
        crop_padding: Padding between crop strips.
        crop_border_width: Width of crop and source-region outlines.
        crop_border_color: RGB outline color.

    Returns:
        PIL image containing the comparison panel and crop strips.
    """

    images_to_draw = [lr, baseline]
    labels_to_draw = list(labels[:2])

    # A shared storage pointer means baseline and adaptation are the same panel.
    if baseline.data_ptr() != adaptation.data_ptr():
        images_to_draw.append(adaptation)
        labels_to_draw.append(labels[2])

    images_to_draw.append(hr)
    labels_to_draw.append(labels[3])

    normalized_regions = [_coerce_region(region) for region in regions]
    images = [_squeeze_image(tensor) for tensor in images_to_draw]
    hr_height, hr_width = images[-1].shape[-2:]
    resized_images = [
        _resize_to_hw(image, height=hr_height, width=hr_width) for image in images
    ]
    pil_images = [_tensor_to_pil(image) for image in resized_images]

    for pil_image in pil_images:
        _draw_region_boxes(
            pil_image,
            normalized_regions,
            border_width=crop_border_width,
            border_color=crop_border_color,
        )

    crop_rows: list[list[Image.Image]] = []
    for region in normalized_regions:
        crop_box = _region_to_box(region, width=hr_width, height=hr_height)
        crop_rows.append(
            [
                _add_border(
                    pil_image.crop(crop_box),
                    width=crop_border_width,
                    color=crop_border_color,
                )
                for pil_image in pil_images
            ]
        )

    return _compose_panel(
        pil_images,
        crop_rows,
        labels=labels_to_draw,
        padding=int(crop_padding),
    )


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    """Convert an image tensor in ``[0, 1]`` to a PIL image.

    Args:
        image: Tensor with shape ``[C, H, W]`` or ``[1, C, H, W]``.

    Returns:
        RGB PIL image.
    """

    return _tensor_to_pil(_squeeze_image(image))


def _compose_panel(
    images: Sequence[Image.Image],
    crop_rows: Sequence[Sequence[Image.Image]],
    *,
    labels: Sequence[str],
    padding: int,
) -> Image.Image:
    """Compose full images and crop rows into one panel."""

    if len(labels) != len(images):
        raise ValueError(f"Expected {len(images)} labels, got {len(labels)}.")

    font = ImageFont.load_default()
    label_height = 16
    image_width = max(image.width for image in images)
    image_height = max(image.height for image in images)
    column_width = image_width
    full_width = len(images) * column_width + (len(images) + 1) * padding

    crop_heights = [max(crop.height for crop in row) for row in crop_rows]
    full_height = (
        padding
        + label_height
        + image_height
        + padding
        + sum(crop_heights)
        + max(0, len(crop_heights) - 1) * padding
        + padding
    )
    panel = Image.new("RGB", (full_width, full_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    x = padding
    for label, image in zip(labels, images, strict=True):
        draw.text((x, padding), str(label), fill=(0, 0, 0), font=font)
        panel.paste(image, (x, padding + label_height))
        x += column_width + padding

    y = padding + label_height + image_height + padding
    for row, row_height in zip(crop_rows, crop_heights, strict=True):
        x = padding
        for crop in row:
            panel.paste(crop, (x, y))
            x += column_width + padding
        y += row_height + padding

    return panel


def _draw_region_boxes(
    image: Image.Image,
    regions: Sequence[CropRegion],
    *,
    border_width: int,
    border_color: tuple[int, int, int],
) -> None:
    """Draw source crop boxes on an image."""

    draw = ImageDraw.Draw(image)
    for region in regions:
        box = _region_to_box(region, width=image.width, height=image.height)
        for offset in range(border_width):
            expanded = (
                box[0] - offset,
                box[1] - offset,
                box[2] + offset,
                box[3] + offset,
            )
            draw.rectangle(expanded, outline=border_color)


def _add_border(
    image: Image.Image,
    *,
    width: int,
    color: tuple[int, int, int],
) -> Image.Image:
    """Return a copy of an image with an RGB border."""

    if width <= 0:
        return image.copy()

    bordered = Image.new(
        "RGB",
        (image.width + 2 * width, image.height + 2 * width),
        color=color,
    )
    bordered.paste(image, (width, width))
    return bordered


def _region_to_box(
    region: CropRegion,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a crop region into a valid PIL crop box."""

    if region.relative:
        left = round(region.x * width)
        top = round(region.y * height)
        crop_width = round(region.width * width)
        crop_height = round(region.height * height)
    else:
        left = round(region.x)
        top = round(region.y)
        crop_width = round(region.width)
        crop_height = round(region.height)

    right = left + max(1, crop_width)
    bottom = top + max(1, crop_height)
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def _resize_to_hw(image: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Resize an image tensor to a target height and width."""

    if image.shape[-2:] == (height, width):
        return image

    batched = image.unsqueeze(0)  # [1, C, H, W]
    resized = F.interpolate(
        batched,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # [1, C, target_h, target_w]
    return resized.squeeze(0)  # [C, target_h, target_w]


def _squeeze_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor shape to ``[C, H, W]``."""

    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)  # [C, H, W]
    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C, H, W], got {tuple(image.shape)}.")
    if image.shape[0] not in {1, 3}:
        raise ValueError(f"Expected 1 or 3 image channels, got {image.shape[0]}.")
    return image.detach().clamp(0.0, 1.0)  # [C, H, W]


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor with shape ``[C, H, W]`` to RGB PIL."""

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # [3, H, W]
    image_uint8 = image.mul(255.0).round().to(torch.uint8)  # [3, H, W]
    image_hwc = image_uint8.permute(1, 2, 0).contiguous()  # [H, W, 3]
    return Image.fromarray(image_hwc.numpy(force=True), mode="RGB")


def _coerce_region(region: CropRegion | Mapping[str, object]) -> CropRegion:
    """Convert a mapping config into a ``CropRegion``."""

    if isinstance(region, CropRegion):
        return region

    return CropRegion(
        name=str(region.get("name", "crop")),
        x=float(region["x"]),
        y=float(region["y"]),
        width=float(region["width"]),
        height=float(region["height"]),
        relative=bool(region.get("relative", False)),
    )


__all__ = ["CropRegion", "draw_crops", "tensor_to_pil_image"]
