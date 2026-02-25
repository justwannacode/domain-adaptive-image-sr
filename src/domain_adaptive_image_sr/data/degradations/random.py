"""Random sampling helpers for tensor degradations."""

from __future__ import annotations

import torch
from torch import Tensor


class RandomMixin:
    """Shared random utilities for degradation pipelines and operations."""

    def _should_apply(
        self,
        *,
        enabled: bool,
        probability: float,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> bool:
        """Sample whether a stage should be applied."""

        if not enabled:
            return False
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        random_value = torch.rand(
            (),
            device=device,
            generator=generator,
        )
        return bool((random_value < probability).item())

    def _random_uniform(
        self,
        low: float,
        high: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> float:
        """Sample a scalar uniformly from `[low, high]`."""

        if low == high:
            return low
        sample = torch.empty(
            (),
            device=device,
            dtype=dtype,
        ).uniform_(low, high, generator=generator)
        return float(sample.item())


def validate_input(hr: Tensor) -> None:
    """Validate the input tensor shape and type."""

    if hr.ndim != 4:
        raise ValueError(f"`hr` must have shape [B, C, H, W], got {tuple(hr.shape)}.")
    if hr.shape[1] not in {1, 3}:
        raise ValueError(
            "`hr` must have 1 or 3 channels for grayscale/RGB support, "
            f"got {hr.shape[1]}."
        )
    if not torch.is_floating_point(hr):
        raise TypeError(
            f"`hr` must be a floating point tensor in [0, 1], got {hr.dtype}."
        )
    if hr.shape[-2] < 1 or hr.shape[-1] < 1:
        raise ValueError(
            f"`hr` spatial size must be positive, got {tuple(hr.shape[-2:])}."
        )


def validate_probability(name: str, value: float) -> None:
    """Validate a probability value."""

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be in [0, 1], got {value}.")


def validate_range(
    name: str,
    value_range: tuple[float, float] | tuple[int, int],
) -> None:
    """Validate a `(min, max)` range."""

    low, high = value_range
    if low > high:
        raise ValueError(
            f"`{name}` must be an ordered pair `(min, max)`, got {value_range}."
        )
