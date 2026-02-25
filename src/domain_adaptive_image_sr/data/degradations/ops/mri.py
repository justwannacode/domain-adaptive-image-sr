"""MRI-specific tensor degradation operations."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import Tensor

from domain_adaptive_image_sr.data.degradations.random import RandomMixin


class MRIDegradationOps(RandomMixin):
    """MRI operations configured from a Hydra degradation section."""

    def __init__(self, config: DictConfig) -> None:
        self.inhomogeneity_config = getattr(config, "inhomogeneity", {})
        self.ghosting_config = getattr(config, "ghosting", {})

    def apply_field_inhomogeneity(
        self, x: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        """Simulate B1 field inhomogeneity with a smooth spatial gradient."""
        H, W = x.shape[-2:]
        device = x.device
        strength_range = self.inhomogeneity_config.get("strength_range", [0.1, 0.4])
        strength = self._random_uniform(
            float(strength_range[0]),
            float(strength_range[1]),
            device=device,
            dtype=x.dtype,
            generator=generator,
        )
        center_x = self._random_uniform(
            -0.5, 0.5, device=device, dtype=x.dtype, generator=generator
        )
        center_y = self._random_uniform(
            -0.5, 0.5, device=device, dtype=x.dtype, generator=generator
        )

        y, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )

        field = 1.0 - strength * ((x_grid - center_x) ** 2 + (y - center_y) ** 2)
        field = field.clamp(0.1, 1.0)
        return x * field[None, None, :, :]

    def apply_ghosting(self, x: Tensor, generator: torch.Generator | None) -> Tensor:
        """Simulate N/2 ghosting artifact common in Echo Planar Imaging (EPI)."""
        intensity_range = self.ghosting_config.get("intensity_range", [0.05, 0.2])
        intensity = self._random_uniform(
            float(intensity_range[0]),
            float(intensity_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        shift = x.shape[-2] // 2
        ghost = torch.roll(x, shifts=shift, dims=-2)
        return torch.clamp(x + intensity * ghost, 0.0, 1.0)
