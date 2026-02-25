"""Domain operation registries for degradation pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import Tensor

from domain_adaptive_image_sr.data.degradations.ops.common import CommonDegradationOps
from domain_adaptive_image_sr.data.degradations.ops.document import (
    DocumentDegradationOps,
)
from domain_adaptive_image_sr.data.degradations.ops.mri import MRIDegradationOps
from domain_adaptive_image_sr.data.degradations.ops.retro import RetroDegradationOps

OperationFn = Callable[[Tensor, torch.Generator | None], Tensor]
PredicateFn = Callable[[torch.device, torch.Generator | None], bool]


@dataclass(frozen=True)
class RegisteredOperation:
    """A named degradation operation plus its probability predicate."""

    name: str
    should_apply: PredicateFn
    apply: OperationFn


class DomainOperationRegistry:
    """Build an ordered list of operations for a degradation domain."""

    def __init__(self, domain: str, config: DictConfig) -> None:
        self.domain = domain
        self.config = config
        self.common_ops = CommonDegradationOps(config)

    def build(self) -> list[RegisteredOperation]:
        """Return operations in the configured domain order."""

        return self._common_order(self.common_ops)

    def _common_order(self, ops: CommonDegradationOps) -> list[RegisteredOperation]:
        """Return the generic SR operation order."""

        return [
            self._operation(
                "blur",
                enabled=bool(ops.blur_config.enabled),
                probability=float(ops.blur_config.probability),
                apply=ops.apply_blur,
            ),
            self._operation(
                "resize",
                enabled=bool(ops.resize_config.enabled),
                probability=float(ops.resize_config.probability),
                apply=lambda x, generator: ops.apply_resize(x),
            ),
            self._operation(
                "jpeg",
                enabled=bool(ops.jpeg_config.enabled),
                probability=float(ops.jpeg_config.probability),
                apply=ops.apply_jpeg,
            ),
        ]

    def _operation(
        self,
        name: str,
        *,
        enabled: bool,
        probability: float,
        apply: OperationFn,
    ) -> RegisteredOperation:
        """Create a registered operation using shared probability logic."""

        def should_apply(
            device: torch.device,
            generator: torch.Generator | None,
        ) -> bool:
            return self.common_ops._should_apply(
                enabled=enabled,
                probability=probability,
                device=device,
                generator=generator,
            )

        return RegisteredOperation(name=name, should_apply=should_apply, apply=apply)


class RetroOperationRegistry(DomainOperationRegistry):
    """Operation registry for retro face photos."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__("retro", config)
        self.common_ops = RetroDegradationOps(config)

    def build(self) -> list[RegisteredOperation]:
        """Return the retro photo operation order."""

        ops = self.common_ops
        return [
            self._operation_from_config(
                "optics",
                ops.optics_config,
                ops.apply_optics,
            ),
            self._operation_from_config(
                "chromatic_aberration",
                ops.chromatic_aberration_config,
                ops.apply_chromatic_aberration,
            ),
            self._operation_from_config(
                "color_shift",
                ops.color_shift_config,
                ops.apply_color_shift,
            ),
            self._operation(
                "resize",
                enabled=bool(ops.resize_config.enabled),
                probability=float(ops.resize_config.probability),
                apply=lambda x, generator: ops.apply_resize(x),
            ),
            self._operation_from_config(
                "digitization_noise",
                ops.digitization_noise_config,
                ops.apply_digitization_noise,
            ),
            self._operation(
                "jpeg",
                enabled=bool(ops.jpeg_config.enabled),
                probability=float(ops.jpeg_config.probability),
                apply=ops.apply_jpeg,
            ),
        ]

    def _operation_from_config(
        self,
        name: str,
        config: DictConfig | dict[str, object],
        apply: OperationFn,
    ) -> RegisteredOperation:
        """Create an operation from a retro-specific config block."""

        return self._operation(
            name,
            enabled=bool(config.get("enabled", False)),
            probability=float(config.get("probability", 1.0)),
            apply=apply,
        )


class DocumentOperationRegistry(DomainOperationRegistry):
    """Operation registry for document scans."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__("docs", config)
        self.common_ops = DocumentDegradationOps(config)

    def build(self) -> list[RegisteredOperation]:
        """Return the document scan operation order."""

        ops = self.common_ops
        return [
            self._operation_from_config(
                "bleedthrough",
                ops.bleedthrough_config,
                ops.apply_bleedthrough,
            ),
            self._operation_from_config(
                "motor_blur",
                ops.motor_blur_config,
                ops.apply_motor_blur,
            ),
            self._operation_from_config(
                "contact_blur",
                ops.contact_blur_config,
                ops.apply_contact_blur,
            ),
            self._operation(
                "resize",
                enabled=bool(ops.resize_config.enabled),
                probability=float(ops.resize_config.probability),
                apply=lambda x, generator: ops.apply_resize(x),
            ),
            self._operation_from_config(
                "impulse_noise",
                ops.impulse_noise_config,
                ops.apply_impulse_noise,
            ),
            self._operation_from_config(
                "poisson_noise",
                ops.poisson_noise_config,
                ops.apply_poisson_noise,
            ),
            self._operation(
                "jpeg",
                enabled=bool(ops.jpeg_config.enabled),
                probability=float(ops.jpeg_config.probability),
                apply=ops.apply_jpeg,
            ),
        ]

    def _operation_from_config(
        self,
        name: str,
        config: DictConfig | dict[str, object],
        apply: OperationFn,
    ) -> RegisteredOperation:
        """Create an operation from a document-specific config block."""

        return self._operation(
            name,
            enabled=bool(config.get("enabled", False)),
            probability=float(config.get("probability", 1.0)),
            apply=apply,
        )


class MRIOperationRegistry(DomainOperationRegistry):
    """Operation registry for MRI T2 scans."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__("mri", config)
        self.mri_ops = MRIDegradationOps(config)

    def build(self) -> list[RegisteredOperation]:
        """Return the MRI operation order exactly as in the original pipeline."""

        ghosting_cfg = self.mri_ops.ghosting_config
        inhomogeneity_cfg = self.mri_ops.inhomogeneity_config
        ops = self.common_ops

        return [
            self._operation(
                "ghosting",
                enabled=bool(ghosting_cfg.get("enabled", False)),
                probability=float(ghosting_cfg.get("probability", 1.0)),
                apply=self.mri_ops.apply_ghosting,
            ),
            self._operation(
                "blur",
                enabled=bool(ops.blur_config.enabled),
                probability=float(ops.blur_config.probability),
                apply=ops.apply_blur,
            ),
            self._operation(
                "resize",
                enabled=bool(ops.resize_config.enabled),
                probability=float(ops.resize_config.probability),
                apply=lambda x, generator: ops.apply_resize(x),
            ),
            self._operation(
                "inhomogeneity",
                enabled=bool(inhomogeneity_cfg.get("enabled", False)),
                probability=float(inhomogeneity_cfg.get("probability", 1.0)),
                apply=self.mri_ops.apply_field_inhomogeneity,
            ),
            self._operation(
                "noise",
                enabled=bool(ops.noise_config.enabled),
                probability=float(ops.noise_config.probability),
                apply=ops.apply_noise,
            ),
            self._operation(
                "jpeg",
                enabled=bool(ops.jpeg_config.enabled),
                probability=float(ops.jpeg_config.probability),
                apply=ops.apply_jpeg,
            ),
        ]


def build_operation_registry(
    config: DictConfig,
    *,
    domain: str | None = None,
) -> DomainOperationRegistry:
    """Create the operation registry for a data domain."""

    resolved_domain = _resolve_domain(config, domain)
    if resolved_domain == "mri":
        return MRIOperationRegistry(config)
    if resolved_domain in {"docs", "documents"}:
        return DocumentOperationRegistry(config)
    return RetroOperationRegistry(config)


def _resolve_domain(config: DictConfig, domain: str | None) -> str:
    """Resolve a degradation domain from explicit input or config shape."""

    if domain is not None:
        return domain
    config_domain = str(config.get("domain", ""))
    if config_domain:
        return config_domain
    if "inhomogeneity" in config or "ghosting" in config:
        return "mri"
    return "retro"
