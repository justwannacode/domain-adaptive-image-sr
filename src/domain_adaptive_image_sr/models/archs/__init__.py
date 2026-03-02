"""Model architecture exports."""

from domain_adaptive_image_sr.models.archs.realesrgan import RealESRGANGenerator
from domain_adaptive_image_sr.models.archs.swinir import SwinIR

__all__ = ["RealESRGANGenerator", "SwinIR"]
