"""Tensor-based image degradation pipeline for super-resolution."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from domain_adaptive_image_sr.data.degradations.ops.common import InterpolationMode
from domain_adaptive_image_sr.data.degradations.random import (
    validate_input,
    validate_probability,
    validate_range,
)
from domain_adaptive_image_sr.data.degradations.registry import (
    RegisteredOperation,
    build_operation_registry,
)


class DegradationPipeline(nn.Module):
    """Apply synthetic degradations to HR tensors for SR training.

    Args:
        config: Hydra config section available as ``config.data.degradation``.
        generator: Optional random generator. When provided, it overrides the
            internal seed-based generator and can be managed by the caller.
        domain: Optional domain name used to select an operation registry.
    """

    def __init__(
        self,
        config: DictConfig,
        generator: torch.Generator | None = None,
        domain: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.enabled = bool(config.enabled)
        self.clamp_output = bool(config.clamp_output)
        self.seed = int(config.seed) if config.seed is not None else None
        self.generator = generator
        self._device_generators: dict[str, torch.Generator] = {}

        self.blur_config = config.blur
        self.resize_config = config.resize
        self.noise_config = config.noise
        self.jpeg_config = config.jpeg

        self._validate_config()
        self.registry = build_operation_registry(config, domain=domain)
        self.operations: list[RegisteredOperation] = self.registry.build()

    def forward(
        self,
        hr: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate an LR tensor from an HR tensor.

        Args:
            hr: High-resolution tensor in `[0, 1]` with shape `[B, C, H, W]`.
            generator: Optional per-call generator used for reproducible random
                sampling inside all stochastic stages.

        Returns:
            Low-resolution tensor with shape
            `[B, C, H // scale_factor, W // scale_factor]`.
        """

        validate_input(hr)
        if not self.enabled:
            return hr

        active_generator = generator or self._get_generator(hr.device)

        x = hr  # [B, C, H, W]
        for operation in self.operations:
            if operation.should_apply(x.device, active_generator):
                x = operation.apply(x, active_generator)

        if self.clamp_output:
            x = x.clamp(0.0, 1.0)  # [B, C, H // scale, W // scale]

        return x

    def _get_generator(self, device: torch.device) -> torch.Generator | None:
        """Return a device-compatible generator if one is configured."""

        if self.generator is not None:
            return self.generator
        if self.seed is None:
            return None

        device_key = str(device)
        if device_key not in self._device_generators:
            try:
                generator = torch.Generator(device=device.type)
            except RuntimeError:
                return None
            generator.manual_seed(self.seed)
            self._device_generators[device_key] = generator
        return self._device_generators[device_key]

    def _validate_config(self) -> None:
        """Validate degradation config values."""

        validate_probability(
            "config.blur.probability",
            float(self.blur_config.probability),
        )
        validate_probability(
            "config.resize.probability",
            float(self.resize_config.probability),
        )
        validate_probability(
            "config.noise.probability",
            float(self.noise_config.probability),
        )
        validate_probability(
            "config.jpeg.probability",
            float(self.jpeg_config.probability),
        )
        validate_range(
            "config.blur.sigma_range",
            tuple(self.blur_config.sigma_range),
        )
        validate_range(
            "config.noise.sigma_range",
            tuple(self.noise_config.sigma_range),
        )
        validate_range(
            "config.jpeg.quality_range",
            tuple(self.jpeg_config.quality_range),
        )
        self._validate_optional_document_config()
        self._validate_optional_retro_config()

        if int(self.resize_config.scale_factor) < 1:
            raise ValueError(
                "`config.resize.scale_factor` must be >= 1, "
                f"got {self.resize_config.scale_factor}."
            )
        if (
            int(self.blur_config.kernel_size) < 1
            or int(self.blur_config.kernel_size) % 2 == 0
        ):
            raise ValueError(
                "`config.blur.kernel_size` must be a positive odd integer, "
                f"got {self.blur_config.kernel_size}."
            )
        if str(self.noise_config.type) not in {"gaussian", "rician"}:
            raise ValueError(
                "`config.noise.type` currently supports only 'gaussian' and 'rician', "
                f"got {self.noise_config.type}."
            )
        if int(self.jpeg_config.block_size) < 1:
            raise ValueError(
                "`config.jpeg.block_size` must be >= 1, "
                f"got {self.jpeg_config.block_size}."
            )
        if not 0.0 <= float(self.jpeg_config.blockiness) <= 1.0:
            raise ValueError(
                "`config.jpeg.blockiness` must be in [0, 1], "
                f"got {self.jpeg_config.blockiness}."
            )
        self._validate_interpolation_mode(
            str(self.resize_config.interpolation),
        )

    def _validate_interpolation_mode(self, mode: str) -> None:
        """Validate the configured resize interpolation mode."""

        valid_modes = {"nearest", "bilinear", "bicubic", "area", "kspace"}
        if mode not in valid_modes:
            raise ValueError(
                "`config.resize.interpolation` must be one of "
                f"{sorted(valid_modes)}, got {mode}."
            )

    def _validate_optional_document_config(self) -> None:
        """Validate optional document-specific degradation config blocks."""

        self._validate_optional_stage("bleedthrough")
        self._validate_optional_stage("motor_blur")
        self._validate_optional_stage("contact_blur")
        self._validate_optional_stage("impulse_noise")
        self._validate_optional_stage("poisson_noise")

        if "bleedthrough" in self.config:
            cfg = self.config.bleedthrough
            validate_range(
                "config.bleedthrough.opacity_range",
                tuple(cfg.opacity_range),
            )
            validate_range("config.bleedthrough.sigma_range", tuple(cfg.sigma_range))
            self._validate_positive_odd_int(
                "config.bleedthrough.kernel_size",
                int(cfg.kernel_size),
            )
        if "motor_blur" in self.config:
            cfg = self.config.motor_blur
            validate_range(
                "config.motor_blur.kernel_size_range",
                tuple(cfg.kernel_size_range),
            )
            validate_range(
                "config.motor_blur.strength_range",
                tuple(cfg.strength_range),
            )
        if "contact_blur" in self.config:
            cfg = self.config.contact_blur
            validate_range("config.contact_blur.sigma_range", tuple(cfg.sigma_range))
            validate_range(
                "config.contact_blur.strength_range",
                tuple(cfg.strength_range),
            )
            validate_probability(
                "config.contact_blur.local_probability",
                float(cfg.local_probability),
            )
            self._validate_positive_odd_int(
                "config.contact_blur.kernel_size",
                int(cfg.kernel_size),
            )
        if "impulse_noise" in self.config:
            cfg = self.config.impulse_noise
            validate_range(
                "config.impulse_noise.density_range",
                tuple(cfg.density_range),
            )
            validate_probability(
                "config.impulse_noise.dark_probability",
                float(cfg.dark_probability),
            )
        if "poisson_noise" in self.config:
            cfg = self.config.poisson_noise
            validate_range("config.poisson_noise.peak_range", tuple(cfg.peak_range))

    def _validate_optional_stage(self, stage_name: str) -> None:
        """Validate a document-specific probabilistic stage when present."""

        if stage_name not in self.config:
            return
        validate_probability(
            f"config.{stage_name}.probability",
            float(self.config[stage_name].probability),
        )

    def _validate_positive_odd_int(self, name: str, value: int) -> None:
        """Validate a positive odd integer config value."""

        if value < 1 or value % 2 == 0:
            raise ValueError(f"`{name}` must be a positive odd integer, got {value}.")

    def _validate_optional_retro_config(self) -> None:
        """Validate optional retro-specific degradation config blocks."""

        self._validate_optional_stage("optics")
        self._validate_optional_stage("chromatic_aberration")
        self._validate_optional_stage("color_shift")
        self._validate_optional_stage("digitization_noise")

        if "optics" in self.config:
            cfg = self.config.optics
            if "anisotropic_gaussian" in cfg:
                anisotropic = cfg.anisotropic_gaussian
                self._validate_positive_odd_int(
                    "config.optics.anisotropic_gaussian.kernel_size",
                    int(anisotropic.kernel_size),
                )
                validate_range(
                    "config.optics.anisotropic_gaussian.sigma_x_range",
                    tuple(anisotropic.sigma_x_range),
                )
                validate_range(
                    "config.optics.anisotropic_gaussian.sigma_y_range",
                    tuple(anisotropic.sigma_y_range),
                )
            if "motion" in cfg:
                motion = cfg.motion
                validate_range(
                    "config.optics.motion.kernel_size_range",
                    tuple(motion.kernel_size_range),
                )
                validate_range(
                    "config.optics.motion.strength_range",
                    tuple(motion.strength_range),
                )
        if "chromatic_aberration" in self.config:
            cfg = self.config.chromatic_aberration
            validate_range(
                "config.chromatic_aberration.shift_range",
                tuple(cfg.shift_range),
            )
            validate_range(
                "config.chromatic_aberration.strength_range",
                tuple(cfg.strength_range),
            )
        if "digitization_noise" in self.config:
            cfg = self.config.digitization_noise
            validate_range(
                "config.digitization_noise.peak_range",
                tuple(cfg.peak_range),
            )
        if "color_shift" in self.config:
            cfg = self.config.color_shift
            validate_range(
                "config.color_shift.brightness_range",
                tuple(cfg.brightness_range),
            )
            validate_range(
                "config.color_shift.contrast_range",
                tuple(cfg.contrast_range),
            )
            validate_range(
                "config.color_shift.saturation_range",
                tuple(cfg.saturation_range),
            )
            validate_range(
                "config.color_shift.white_balance_range",
                tuple(cfg.white_balance_range),
            )


__all__ = ["DegradationPipeline", "InterpolationMode"]
