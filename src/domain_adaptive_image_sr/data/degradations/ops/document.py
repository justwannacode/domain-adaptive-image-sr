"""Document scan degradation operations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from domain_adaptive_image_sr.data.degradations.ops.common import CommonDegradationOps


class DocumentDegradationOps(CommonDegradationOps):
    """Document scan operations.

    Args:
        config: Hydra degradation config for the document domain.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.bleedthrough_config = getattr(config, "bleedthrough", {})
        self.motor_blur_config = getattr(config, "motor_blur", {})
        self.contact_blur_config = getattr(config, "contact_blur", {})
        self.impulse_noise_config = getattr(config, "impulse_noise", {})
        self.poisson_noise_config = getattr(config, "poisson_noise", {})

    def apply_bleedthrough(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Blend a mirrored, heavily blurred back-side page impression."""

        opacity_range = self.bleedthrough_config.get("opacity_range", [0.03, 0.12])
        sigma_range = self.bleedthrough_config.get("sigma_range", [2.0, 5.0])
        kernel_size = int(self.bleedthrough_config.get("kernel_size", 17))
        axis = self._sample_axis(
            str(self.bleedthrough_config.get("mirror_axis", "horizontal")),
            x.device,
            generator,
        )
        opacity = self._random_uniform(
            float(opacity_range[0]),
            float(opacity_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        sigma = self._random_uniform(
            float(sigma_range[0]),
            float(sigma_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        flip_dim = -1 if axis == "horizontal" else -2
        back_side = torch.flip(x, dims=(flip_dim,))  # [B, C, H, W]
        back_side = self._depthwise_gaussian_blur(
            back_side,
            kernel_size=kernel_size,
            sigma=sigma,
        )  # [B, C, H, W]
        return torch.lerp(x, back_side, opacity)  # [B, C, H, W]

    def apply_motor_blur(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply strict 1D scan motor blur along X or Y."""

        kernel_range = self.motor_blur_config.get("kernel_size_range", [3, 9])
        strength_range = self.motor_blur_config.get("strength_range", [0.25, 0.8])
        axis = self._sample_axis(
            str(self.motor_blur_config.get("axis", "random")),
            x.device,
            generator,
        )
        kernel_size = self._sample_odd_int(
            int(kernel_range[0]),
            int(kernel_range[1]),
            x.device,
            generator,
        )
        strength = self._random_uniform(
            float(strength_range[0]),
            float(strength_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        blurred = self._depthwise_box_blur_1d(
            x,
            kernel_size=kernel_size,
            axis=axis,
        )  # [B, C, H, W]
        return torch.lerp(x, blurred, strength)  # [B, C, H, W]

    def apply_contact_blur(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply isotropic blur globally or through smooth local masks."""

        sigma_range = self.contact_blur_config.get("sigma_range", [0.4, 1.4])
        kernel_size = int(self.contact_blur_config.get("kernel_size", 9))
        strength_range = self.contact_blur_config.get("strength_range", [0.2, 0.7])
        local_probability = float(
            self.contact_blur_config.get("local_probability", 0.6)
        )
        sigma = self._random_uniform(
            float(sigma_range[0]),
            float(sigma_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        strength = self._random_uniform(
            float(strength_range[0]),
            float(strength_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        blurred = self._depthwise_gaussian_blur(
            x,
            kernel_size=kernel_size,
            sigma=sigma,
        )  # [B, C, H, W]

        if not self._should_apply(
            enabled=True,
            probability=local_probability,
            device=x.device,
            generator=generator,
        ):
            return torch.lerp(x, blurred, strength)  # [B, C, H, W]

        mask = self._smooth_random_mask(x, generator)  # [B, 1, H, W]
        return x * (1.0 - mask * strength) + blurred * (mask * strength)

    def apply_impulse_noise(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply sparse impulse dust and micro-scratch noise."""

        density_range = self.impulse_noise_config.get("density_range", [0.0005, 0.003])
        dark_probability = float(self.impulse_noise_config.get("dark_probability", 0.7))
        density = self._random_uniform(
            float(density_range[0]),
            float(density_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        mask_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
        dust_mask = (
            torch.rand(
                mask_shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            < density
        )
        dark_mask = (
            torch.rand(
                mask_shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            < dark_probability
        )
        impulse_values = torch.where(
            dark_mask,
            torch.zeros_like(x),
            torch.ones_like(x),
        )
        dust_mask = dust_mask.expand(-1, x.shape[1], -1, -1)
        return torch.where(dust_mask, impulse_values, x)  # [B, C, H, W]

    def apply_poisson_noise(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply sensor-like Poisson shot noise."""

        peak_range = self.poisson_noise_config.get("peak_range", [30.0, 120.0])
        peak = self._random_uniform(
            float(peak_range[0]),
            float(peak_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        positive = x.clamp_min(0.0)  # [B, C, H, W]
        noisy = torch.poisson(positive * peak, generator=generator) / peak
        return noisy.to(dtype=x.dtype)  # [B, C, H, W]

    def _depthwise_box_blur_1d(
        self,
        x: Tensor,
        *,
        kernel_size: int,
        axis: str,
    ) -> Tensor:
        """Apply a depthwise 1D box blur along one spatial axis."""

        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if axis == "horizontal":
            kernel = x.new_ones((x.shape[1], 1, 1, kernel_size)) / float(kernel_size)
            padding = (kernel_size // 2, kernel_size // 2, 0, 0)
        else:
            kernel = x.new_ones((x.shape[1], 1, kernel_size, 1)) / float(kernel_size)
            padding = (0, 0, kernel_size // 2, kernel_size // 2)
        padded = F.pad(x, padding, mode="replicate")  # [B, C, H + p, W + p]
        return F.conv2d(padded, kernel, groups=x.shape[1])  # [B, C, H, W]

    def _depthwise_gaussian_blur(
        self,
        x: Tensor,
        *,
        kernel_size: int,
        sigma: float,
    ) -> Tensor:
        """Apply a depthwise isotropic Gaussian blur."""

        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = self._build_gaussian_kernel(
            kernel_size=kernel_size,
            sigma=sigma,
            device=x.device,
            dtype=x.dtype,
        )
        kernel = kernel.expand(x.shape[1], 1, -1, -1)  # [C, 1, K, K]
        padding = kernel_size // 2
        padded = F.pad(
            x,
            (padding, padding, padding, padding),
            mode="reflect" if min(x.shape[-2:]) > padding else "replicate",
        )  # [B, C, H + 2p, W + 2p]
        return F.conv2d(padded, kernel, groups=x.shape[1])  # [B, C, H, W]

    def _smooth_random_mask(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Create a smooth local blur mask."""

        grid_size = int(self.contact_blur_config.get("mask_grid_size", 8))
        threshold = float(self.contact_blur_config.get("mask_threshold", 0.45))
        grid_h = max(1, min(grid_size, x.shape[-2]))
        grid_w = max(1, min(grid_size, x.shape[-1]))
        coarse = torch.rand(
            (x.shape[0], 1, grid_h, grid_w),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        mask = F.interpolate(
            coarse,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # [B, 1, H, W]
        mask = ((mask - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
        return mask

    def _sample_axis(
        self,
        axis: str,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> str:
        """Sample a spatial axis name."""

        if axis in {"horizontal", "vertical"}:
            return axis
        index = torch.randint(0, 2, (), device=device, generator=generator)
        return "horizontal" if int(index.item()) == 0 else "vertical"

    def _sample_odd_int(
        self,
        low: int,
        high: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> int:
        """Sample an odd integer from an inclusive range."""

        low = max(1, low)
        high = max(low, high)
        value = torch.randint(low, high + 1, (), device=device, generator=generator)
        sampled = int(value.item())
        if sampled % 2 == 0:
            sampled = sampled + 1 if sampled < high else sampled - 1
        return max(1, sampled)
