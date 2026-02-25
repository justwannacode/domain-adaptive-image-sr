"""Common tensor degradation operations shared by image domains."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from domain_adaptive_image_sr.data.degradations.random import RandomMixin

try:
    import kornia
except ImportError:  # pragma: no cover - runtime dependency fallback
    kornia = None


InterpolationMode = Literal["nearest", "bilinear", "bicubic", "area", "kspace"]


class CommonDegradationOps(RandomMixin):
    """Common degradation operations configured from a Hydra degradation section."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.blur_config = config.blur
        self.resize_config = config.resize
        self.noise_config = config.noise
        self.jpeg_config = config.jpeg

    def apply_blur(self, x: Tensor, generator: torch.Generator | None) -> Tensor:
        """Apply Gaussian blur with a randomly sampled sigma."""

        sigma = self._random_uniform(
            float(self.blur_config.sigma_range[0]),
            float(self.blur_config.sigma_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        if kornia is not None:
            return kornia.filters.gaussian_blur2d(
                x,
                kernel_size=(
                    int(self.blur_config.kernel_size),
                    int(self.blur_config.kernel_size),
                ),
                sigma=(float(sigma), float(sigma)),
                border_type="reflect",
            )  # [B, C, H, W]

        return self._gaussian_blur_fallback(x, float(sigma))

    def apply_resize(self, x: Tensor) -> Tensor:
        """Resize the tensor to the target LR size."""

        scale_factor = int(self.resize_config.scale_factor)
        interpolation = str(self.resize_config.interpolation)

        if interpolation == "kspace":
            target_h = max(1, x.shape[-2] // scale_factor)
            target_w = max(1, x.shape[-1] // scale_factor)

            fft = torch.fft.fft2(x, norm="ortho")
            fft = torch.fft.fftshift(fft, dim=(-2, -1))

            H, W = x.shape[-2:]
            start_h = (H - target_h) // 2
            start_w = (W - target_w) // 2

            cropped_fft = fft[
                ..., start_h : start_h + target_h, start_w : start_w + target_w
            ]
            cropped_fft = torch.fft.ifftshift(cropped_fft, dim=(-2, -1))
            ifft = torch.fft.ifft2(cropped_fft, norm="ortho")

            out = ifft.real / float(scale_factor)

            return out

        target_h = max(1, x.shape[-2] // scale_factor)
        target_w = max(1, x.shape[-1] // scale_factor)
        target_size = (target_h, target_w)
        antialias = bool(self.resize_config.antialias)

        if kornia is not None:
            align_corners = False if interpolation in {"bilinear", "bicubic"} else None
            return kornia.geometry.transform.resize(
                x,
                size=target_size,
                interpolation=interpolation,
                align_corners=align_corners,
                antialias=antialias,
            )  # [B, C, H // scale, W // scale]

        kwargs: dict[str, object] = {"size": target_size, "mode": interpolation}
        if interpolation in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
            kwargs["antialias"] = antialias
        return F.interpolate(x, **kwargs)  # [B, C, H // scale, W // scale]

    def apply_noise(self, x: Tensor, generator: torch.Generator | None) -> Tensor:
        """Add noise to the tensor based on configured type."""

        sigma = self._random_uniform(
            float(self.noise_config.sigma_range[0]),
            float(self.noise_config.sigma_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )

        is_color_noise = bool(self.noise_config.get("color_noise", True))
        noise_shape = (
            x.shape if is_color_noise else (x.shape[0], 1, x.shape[2], x.shape[3])
        )

        noise_real = torch.randn(
            noise_shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        if not is_color_noise and x.shape[1] > 1:
            noise_real = noise_real.expand(-1, x.shape[1], -1, -1)

        if str(self.noise_config.type) == "rician":
            noise_imag = torch.randn(
                noise_shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            if not is_color_noise and x.shape[1] > 1:
                noise_imag = noise_imag.expand(-1, x.shape[1], -1, -1)
            return torch.sqrt(
                (x + noise_real * sigma).square() + (noise_imag * sigma).square()
            )

        # Unknown noise aliases fall back to additive Gaussian noise.
        return x + noise_real * sigma  # [B, C, H, W]

    def apply_jpeg(
        self,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Approximate JPEG artifacts with blockiness and quantization.

        This is a lightweight tensor-only approximation and not a true JPEG
        codec. It intentionally keeps all operations on tensors so the pipeline
        remains compatible with GPU training loops.
        """

        quality = self._random_uniform(
            float(self.jpeg_config.quality_range[0]),
            float(self.jpeg_config.quality_range[1]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        compression_strength = (100.0 - quality) / 100.0

        if (
            int(self.jpeg_config.block_size) > 1
            and float(self.jpeg_config.blockiness) > 0.0
        ):
            x = self._apply_block_artifacts(
                x,
                block_size=int(self.jpeg_config.block_size),
                strength=float(
                    compression_strength * float(self.jpeg_config.blockiness)
                ),
            )  # [B, C, H, W]

        levels = int(round(8.0 + (1.0 - compression_strength) * 248.0))
        levels = max(8, levels)
        return torch.round(x * (levels - 1)) / float(levels - 1)  # [B, C, H, W]

    def _apply_block_artifacts(
        self,
        x: Tensor,
        block_size: int,
        strength: float,
    ) -> Tensor:
        """Blend the image with a block-averaged version."""

        if strength <= 0.0:
            return x

        height, width = x.shape[-2:]
        pad_h = (block_size - height % block_size) % block_size
        pad_w = (block_size - width % block_size) % block_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")  # [B, C, Hp, Wp]

        blocky = F.avg_pool2d(
            x_padded,
            kernel_size=block_size,
            stride=block_size,
        )  # [B, C, Hp // bs, Wp // bs]
        blocky = F.interpolate(
            blocky,
            size=x_padded.shape[-2:],
            mode="nearest",
        )  # [B, C, Hp, Wp]
        blocky = blocky[..., :height, :width]  # [B, C, H, W]
        return torch.lerp(x, blocky, strength)

    def _gaussian_blur_fallback(self, x: Tensor, sigma: float) -> Tensor:
        """Fallback Gaussian blur when Kornia is unavailable."""

        kernel_size = int(self.blur_config.kernel_size)
        kernel = self._build_gaussian_kernel(
            kernel_size=kernel_size,
            sigma=sigma,
            device=x.device,
            dtype=x.dtype,
        )
        kernel = kernel.expand(x.shape[1], 1, -1, -1)  # [C, 1, K, K]
        padding = kernel_size // 2
        pad_mode = (
            "reflect"
            if x.shape[-2] > padding and x.shape[-1] > padding
            else "replicate"
        )
        x = F.pad(
            x,
            (padding, padding, padding, padding),
            mode=pad_mode,
        )  # [B, C, H + 2p, W + 2p]
        return F.conv2d(x, kernel, groups=x.shape[1])  # [B, C, H, W]

    def _build_gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build a normalized 2D Gaussian kernel."""

        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords = coords - (kernel_size - 1) / 2.0
        sigma_tensor = torch.tensor(sigma, device=device, dtype=dtype).clamp_min(1e-6)
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma_tensor.square()))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d[None, None, :, :]
