"""Retro-photo degradation operations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from domain_adaptive_image_sr.data.degradations.ops.common import CommonDegradationOps


class RetroDegradationOps(CommonDegradationOps):
    """Retro face photo operations for 1980s-1990s photo scans.

    Args:
        config: Hydra degradation config for the retro photo domain.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.optics_config = getattr(config, "optics", {})
        self.chromatic_aberration_config = getattr(config, "chromatic_aberration", {})
        self.color_shift_config = getattr(config, "color_shift", {})
        self.digitization_noise_config = getattr(config, "digitization_noise", {})

    def apply_optics(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply camera optics blur: anisotropic defocus or motion blur."""

        mode = self._sample_mode(
            tuple(str(item) for item in self.optics_config.get("modes", [])),
            x.device,
            generator,
        )
        if mode == "motion":
            return self._apply_motion_blur(x, generator)  # [B, C, H, W]
        return self._apply_anisotropic_gaussian_blur(x, generator)  # [B, C, H, W]

    def apply_chromatic_aberration(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply mild opposite red/blue channel shifts."""

        if x.shape[1] != 3:
            return x

        shift_range = self.chromatic_aberration_config.get("shift_range", [1, 2])
        strength_range = self.chromatic_aberration_config.get(
            "strength_range",
            [0.15, 0.45],
        )
        shift = self._sample_int(
            int(shift_range[0]),
            int(shift_range[1]),
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
        if shift <= 0 or strength <= 0.0:
            return x

        shifted = x.clone()  # [B, 3, H, W]
        shifted[:, 0] = torch.roll(x[:, 0], shifts=(0, shift), dims=(-2, -1))
        shifted[:, 2] = torch.roll(x[:, 2], shifts=(0, -shift), dims=(-2, -1))
        return torch.lerp(x, shifted, strength)  # [B, C, H, W]

    def apply_digitization_noise(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply cheap scanner matrix noise using Poisson noise and a luminance mask."""

        peak_range = self.digitization_noise_config.get("peak_range", [80.0, 300.0])
        peak = self._random_uniform(
            float(peak_range[0]),
            float(peak_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        if peak <= 0.0:
            return x

        # The offset keeps Poisson noise active in near-black regions.
        base_signal = x.clamp(0.0, 1.0) + 0.1
        noisy = (
            torch.poisson(
                base_signal * peak,
                generator=generator,
            ).to(dtype=x.dtype)
            / peak
        )
        delta = noisy - base_signal

        if x.shape[1] == 3:
            luminance = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        else:
            luminance = x

        # Low-cost scanner noise is strongest in darker regions.
        mask = (1.0 - luminance).clamp_min(0.2)
        return (x + delta * mask).clamp_(0.0, 1.0)  # [B, C, H, W]

    def apply_color_shift(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply brightness, contrast, saturation and white-balance shifts."""

        brightness_range = self.color_shift_config.get("brightness_range", [0.95, 1.05])
        contrast_range = self.color_shift_config.get("contrast_range", [0.9, 1.1])
        saturation_range = self.color_shift_config.get("saturation_range", [0.85, 1.1])
        white_balance_range = self.color_shift_config.get(
            "white_balance_range",
            [0.95, 1.05],
        )

        brightness = self._random_uniform(
            float(brightness_range[0]),
            float(brightness_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        contrast = self._random_uniform(
            float(contrast_range[0]),
            float(contrast_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        saturation = self._random_uniform(
            float(saturation_range[0]),
            float(saturation_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        out = (x - 0.5) * contrast + 0.5
        out = out * brightness

        if out.shape[1] == 3:
            luminance = (
                out[:, 0:1] * 0.299 + out[:, 1:2] * 0.587 + out[:, 2:3] * 0.114
            )  # [B, 1, H, W]
            out = luminance + (out - luminance) * saturation
            white_balance = self._sample_white_balance(
                white_balance_range,
                x.device,
                x.dtype,
                generator,
            )
            out = out * white_balance.view(1, 3, 1, 1)

        return out  # [B, C, H, W]

    def _apply_anisotropic_gaussian_blur(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply rotated anisotropic Gaussian blur."""

        cfg = self.optics_config.get("anisotropic_gaussian", {})
        kernel_size = int(cfg.get("kernel_size", 9))
        sigma_x_range = cfg.get("sigma_x_range", [0.4, 1.4])
        sigma_y_range = cfg.get("sigma_y_range", [0.2, 0.8])
        sigma_x = self._random_uniform(
            float(sigma_x_range[0]),
            float(sigma_x_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        sigma_y = self._random_uniform(
            float(sigma_y_range[0]),
            float(sigma_y_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        angle = self._random_uniform(
            0.0,
            math.pi,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        kernel = self._build_anisotropic_gaussian_kernel(
            kernel_size=kernel_size,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            angle=angle,
            device=x.device,
            dtype=x.dtype,
        )
        return self._depthwise_filter(x, kernel)  # [B, C, H, W]

    def _apply_motion_blur(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply hand-shake-like linear motion blur."""

        cfg = self.optics_config.get("motion", {})
        kernel_range = cfg.get("kernel_size_range", [3, 9])
        strength_range = cfg.get("strength_range", [0.15, 0.55])
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
        angle = self._random_uniform(
            0.0,
            math.pi,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        kernel = self._build_motion_kernel(
            kernel_size=kernel_size,
            angle=angle,
            device=x.device,
            dtype=x.dtype,
        )
        blurred = self._depthwise_filter(x, kernel)  # [B, C, H, W]
        return torch.lerp(x, blurred, strength)  # [B, C, H, W]

    def _build_anisotropic_gaussian_kernel(
        self,
        *,
        kernel_size: int,
        sigma_x: float,
        sigma_y: float,
        angle: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build a rotated anisotropic Gaussian kernel."""

        kernel_size = self._ensure_odd(kernel_size)
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords = coords - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_rot = xx * cos_a + yy * sin_a
        y_rot = -xx * sin_a + yy * cos_a
        sigma_x_t = torch.tensor(sigma_x, device=device, dtype=dtype).clamp_min(1e-6)
        sigma_y_t = torch.tensor(sigma_y, device=device, dtype=dtype).clamp_min(1e-6)
        kernel = torch.exp(
            -0.5 * ((x_rot / sigma_x_t).square() + (y_rot / sigma_y_t).square())
        )
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel[None, None, :, :]

    def _build_motion_kernel(
        self,
        *,
        kernel_size: int,
        angle: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build a soft line kernel for motion blur."""

        kernel_size = self._ensure_odd(kernel_size)
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords = coords - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        along = xx * cos_a + yy * sin_a
        perpendicular = -xx * sin_a + yy * cos_a
        half_length = float(kernel_size) / 2.0
        line = (along.abs() <= half_length).to(dtype)
        kernel = line * torch.exp(-perpendicular.square() / 0.5)
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel[None, None, :, :]

    def _depthwise_filter(self, x: Tensor, kernel: Tensor) -> Tensor:
        """Apply a single 2D kernel independently to every channel."""

        kernel_size = kernel.shape[-1]
        expanded_kernel = kernel.expand(x.shape[1], 1, -1, -1)  # [C, 1, K, K]
        padding = kernel_size // 2
        padded = F.pad(
            x,
            (padding, padding, padding, padding),
            mode="reflect" if min(x.shape[-2:]) > padding else "replicate",
        )  # [B, C, H + 2p, W + 2p]
        return F.conv2d(padded, expanded_kernel, groups=x.shape[1])  # [B, C, H, W]

    def _sample_white_balance(
        self,
        value_range: list[float] | tuple[float, float],
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Sample per-channel white balance multipliers."""

        low, high = float(value_range[0]), float(value_range[1])
        return torch.empty((3,), device=device, dtype=dtype).uniform_(
            low,
            high,
            generator=generator,
        )

    def _sample_mode(
        self,
        modes: tuple[str, ...],
        device: torch.device,
        generator: torch.Generator | None,
    ) -> str:
        """Sample one blur mode from a configured mode list."""

        if not modes:
            return "anisotropic_gaussian"
        index = self._sample_int(0, len(modes) - 1, device, generator)
        return modes[index]

    def _sample_int(
        self,
        low: int,
        high: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> int:
        """Sample an integer from an inclusive range."""

        high = max(low, high)
        value = torch.randint(low, high + 1, (), device=device, generator=generator)
        return int(value.item())

    def _sample_odd_int(
        self,
        low: int,
        high: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> int:
        """Sample an odd integer from an inclusive range."""

        sampled = self._sample_int(max(1, low), max(1, high), device, generator)
        if sampled % 2 == 0:
            sampled = sampled + 1 if sampled < high else sampled - 1
        return max(1, sampled)

    def _ensure_odd(self, value: int) -> int:
        """Return a positive odd integer."""

        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1
