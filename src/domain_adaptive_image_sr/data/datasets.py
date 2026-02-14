"""Dataset implementations for super-resolution data loading."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from domain_adaptive_image_sr.data.degradation import DegradationPipeline
from domain_adaptive_image_sr.utils.io import list_image_files, read_image

Sample = dict[str, Tensor | str | dict[str, Any]]


class SyntheticSRDataset(Dataset[Sample]):
    """Synthetic SR dataset that generates LR tensors from HR images.

    Args:
        config: Hydra data config section available as ``config.data``.
        split: Dataset split name. Expected values are ``"train"``, ``"val"``,
            or ``"test"`` when those sections exist in config.
        degradation_pipeline: Optional degradation pipeline. When omitted, a
            pipeline is created from ``config.degradation``.
        generator: Optional random generator used for patch sampling and
            degradation randomness.
    """

    def __init__(
        self,
        config: DictConfig,
        split: str,
        degradation_pipeline: DegradationPipeline | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.split_config = self._get_split_config(config, split)
        self.generator = generator
        self.degradation_pipeline = degradation_pipeline or DegradationPipeline(
            config.degradation,
            generator=generator,
            domain=str(config.get("name", "")) or None,
        )

        hr_dir = self.split_config.hr_dir
        if hr_dir is None:
            raise ValueError(f"`config.data.{split}.hr_dir` must be set.")

        self.hr_paths = list_image_files(
            hr_dir,
            list(config.extensions),
            recursive=bool(config.recursive),
        )
        if not self.hr_paths:
            raise ValueError(f"No HR images found in configured directory: {hr_dir}")

    def __len__(self) -> int:
        """Return the number of HR images."""

        return len(self.hr_paths)

    def __getitem__(self, index: int) -> Sample:
        """Return one synthetic SR sample."""

        hr_path = self.hr_paths[index]
        hr = read_image(hr_path, mode=str(self.config.image_mode))  # [C, H, W]
        sample_generator = self._make_sample_generator(index)
        hr = self._crop_patch(hr, sample_generator)  # [C, patch_h, patch_w]

        hr_batch = hr.unsqueeze(0)  # [1, C, H, W]
        lr = self.degradation_pipeline(
            hr_batch,
            generator=sample_generator,
        ).squeeze(0)  # [C, H // scale, W // scale]

        return {
            "lr": lr,
            "hr": hr,
            "image_id": hr_path.stem,
            "metadata": {
                "mode": "synthetic",
                "split": self.split,
                "hr_path": str(hr_path),
                "lr_path": None,
                "has_hr": True,
                "scale": int(self.config.scale),
            },
        }

    def _crop_patch(
        self,
        image: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Crop a configured patch from a single image tensor."""

        if not bool(self.config.patch.enabled):
            scale = int(self.config.scale)
            _, height, width = image.shape
            aligned_h = (height // scale) * scale
            aligned_w = (width // scale) * scale
            return image[:, :aligned_h, :aligned_w]

        patch_size = int(self.config.patch.size)
        if patch_size < 1:
            raise ValueError(
                f"`config.data.patch.size` must be >= 1, got {patch_size}."
            )

        _, height, width = image.shape
        crop_h = min(patch_size, height)
        crop_w = min(patch_size, width)
        top, left = _sample_crop_origin(
            height=height,
            width=width,
            crop_h=crop_h,
            crop_w=crop_w,
            random_crop=bool(self.config.patch.random),
            generator=generator,
        )
        return image[:, top : top + crop_h, left : left + crop_w]  # [C, crop_h, crop_w]

    def _make_sample_generator(self, index: int) -> torch.Generator | None:
        """Create a deterministic per-sample generator when configured."""

        if not bool(self.config.patch.seed_per_sample):
            return self.generator
        if self.config.seed is None:
            return self.generator

        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed) + index)
        return generator

    @staticmethod
    def _get_split_config(config: DictConfig, split: str) -> DictConfig:
        """Resolve a split section from the dataset config."""

        if split not in config:
            raise ValueError(f"`config.data.{split}` section is not configured.")
        return config[split]


class UnpairedDomainDataset(Dataset[Sample]):
    """Unpaired low-quality dataset for domain adaptation.

    Args:
        config: Hydra data config section available as ``config.data``.
        split: Dataset split name. Expected values are ``"train"``, ``"val"``,
            or ``"test"`` when those sections exist in config.
        generator: Optional random generator used for patch sampling.
    """

    def __init__(
        self,
        config: DictConfig,
        split: str,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.split_config = self._get_split_config(config, split)
        self.generator = generator

        lq_dir = self.split_config.lq_dir or self.split_config.lr_dir
        if lq_dir is None:
            raise ValueError(
                f"`config.data.{split}.lq_dir` or `config.data.{split}.lr_dir` "
                "must be set."
            )

        self.lq_paths = list_image_files(
            lq_dir,
            list(config.extensions),
            recursive=bool(config.recursive),
        )
        if not self.lq_paths:
            raise ValueError(
                f"No low-quality images found in configured directory: {lq_dir}"
            )

    def __len__(self) -> int:
        """Return the number of low-quality domain images."""

        return len(self.lq_paths)

    def __getitem__(self, index: int) -> Sample:
        """Return one unpaired domain adaptation sample."""

        lq_path = self.lq_paths[index]
        lr = read_image(lq_path, mode=str(self.config.image_mode))  # [C, H, W]
        sample_generator = self._make_sample_generator(index)
        lr = self._crop_patch(lr, sample_generator)  # [C, patch_h, patch_w]
        hr = torch.empty(0, dtype=lr.dtype)  # [0]

        return {
            "lr": lr,
            "hr": hr,
            "image_id": lq_path.stem,
            "metadata": {
                "mode": "unpaired",
                "split": self.split,
                "hr_path": None,
                "lr_path": str(lq_path),
                "has_hr": False,
                "scale": int(self.config.scale),
            },
        }

    def _crop_patch(
        self,
        image: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Crop a configured patch from a single low-quality image tensor."""

        if not bool(self.config.patch.enabled):
            return image

        patch_size = int(self.config.patch.size)
        if patch_size < 1:
            raise ValueError(
                f"`config.data.patch.size` must be >= 1, got {patch_size}."
            )

        _, height, width = image.shape
        crop_h = min(patch_size, height)
        crop_w = min(patch_size, width)
        top, left = _sample_crop_origin(
            height=height,
            width=width,
            crop_h=crop_h,
            crop_w=crop_w,
            random_crop=bool(self.config.patch.random),
            generator=generator,
        )
        return image[:, top : top + crop_h, left : left + crop_w]  # [C, crop_h, crop_w]

    def _make_sample_generator(self, index: int) -> torch.Generator | None:
        """Create a deterministic per-sample generator when configured."""

        if not bool(self.config.patch.seed_per_sample):
            return self.generator
        if self.config.seed is None:
            return self.generator

        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed) + index)
        return generator

    @staticmethod
    def _get_split_config(config: DictConfig, split: str) -> DictConfig:
        """Resolve a split section from the dataset config."""

        if split not in config:
            raise ValueError(f"`config.data.{split}` section is not configured.")
        return config[split]


def _sample_crop_origin(
    *,
    height: int,
    width: int,
    crop_h: int,
    crop_w: int,
    random_crop: bool,
    generator: torch.Generator | None,
) -> tuple[int, int]:
    """Sample or compute a crop origin for an image tensor."""

    max_top = height - crop_h
    max_left = width - crop_w
    if max_top == 0 and max_left == 0:
        return 0, 0
    if not random_crop:
        return max_top // 2, max_left // 2

    top = _randint_inclusive(max_top, generator)
    left = _randint_inclusive(max_left, generator)
    return top, left


def _randint_inclusive(
    upper_bound: int,
    generator: torch.Generator | None,
) -> int:
    """Sample an integer from ``[0, upper_bound]``."""

    if upper_bound <= 0:
        return 0
    value = torch.randint(
        low=0,
        high=upper_bound + 1,
        size=(),
        generator=generator,
    )
    return int(value.item())


__all__ = ["SyntheticSRDataset", "UnpairedDomainDataset"]
