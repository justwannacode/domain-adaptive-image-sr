"""Backward-compatible degradation pipeline import."""

from domain_adaptive_image_sr.data.degradations.pipeline import (
    DegradationPipeline,
    InterpolationMode,
)

__all__ = ["DegradationPipeline", "InterpolationMode"]
