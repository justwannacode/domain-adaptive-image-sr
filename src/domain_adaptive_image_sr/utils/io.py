"""Safe filesystem and image IO utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor


def list_image_files(
    root: str | Path,
    extensions: Sequence[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    """List image files under a directory in deterministic order.

    Args:
        root: Directory to scan.
        extensions: Allowed file extensions, with or without leading dots.
        recursive: Whether to scan nested directories.

    Returns:
        Sorted list of image paths.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
        ValueError: If no extensions are provided.
    """

    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {root_path}")
    if not extensions:
        raise ValueError("At least one image extension must be configured.")

    normalized_extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    iterator = root_path.rglob("*") if recursive else root_path.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def read_image(path: str | Path, *, mode: str = "rgb") -> Tensor:
    """Read an image file as a floating point tensor in ``[0, 1]``.

    Args:
        path: Image file path.
        mode: Output color mode. Supported values are ``"rgb"`` and
            ``"grayscale"``.

    Returns:
        Image tensor with shape ``[C, H, W]``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``mode`` is unsupported or the image cannot be decoded.
    """

    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    if image_path.suffix.lower() == ".dcm":
        try:
            import pydicom
            import numpy as np
        except ImportError as error:
            raise ImportError("pydicom is required to read .dcm files") from error

        try:
            dicom = pydicom.dcmread(str(image_path))
            pixel_array = dicom.pixel_array.astype(np.float32)
            tensor = torch.from_numpy(pixel_array)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)  # [1, H, W]
            elif tensor.ndim == 3:
                tensor = tensor.permute(2, 0, 1).contiguous()  # [C, H, W]

            # Float arrays may arrive outside [0, 1]; normalize before conversion.
            t_min = tensor.min()
            t_max = tensor.max()
            if t_max > t_min:
                tensor = (tensor - t_min) / (t_max - t_min)
            else:
                tensor = tensor - t_min

            normalized_mode = mode.lower()
            if normalized_mode == "rgb" and tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)  # [3, H, W]
            elif normalized_mode in {"grayscale", "gray", "l"} and tensor.shape[0] == 3:
                tensor = tensor.mean(dim=0, keepdim=True)  # [1, H, W]

            return tensor
        except Exception as error:
            raise ValueError(f"Failed to read DICOM file: {image_path}") from error

    pil_mode = _to_pil_mode(mode)
    try:
        with Image.open(image_path) as image:
            image = image.convert(pil_mode)
            image_bytes = torch.ByteTensor(bytearray(image.tobytes()))
            channel_count = len(image.getbands())
            height, width = image.height, image.width
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Failed to read image file: {image_path}") from error

    tensor = image_bytes.view(height, width, channel_count)  # [H, W, C]
    tensor = tensor.permute(2, 0, 1).contiguous()  # [C, H, W]
    return tensor.to(dtype=torch.float32).div(255.0)  # [C, H, W]


def _to_pil_mode(mode: str) -> str:
    """Convert a config image mode into a PIL mode."""

    normalized_mode = mode.lower()
    if normalized_mode == "rgb":
        return "RGB"
    if normalized_mode in {"grayscale", "gray", "l"}:
        return "L"
    raise ValueError(f"Unsupported image mode: {mode}")


__all__ = ["list_image_files", "read_image"]
