"""Thin wrapper around the official Real-ESRGAN RRDBNet generator."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch import nn

_UPSTREAM_IMPORT_CANDIDATES = (
    "basicsr.archs.rrdbnet_arch",
    "realesrgan.archs.rrdbnet_arch",
    "domain_adaptive_image_sr.vendor.basicsr.archs.rrdbnet_arch",
    "domain_adaptive_image_sr.vendor.realesrgan.basicsr.archs.rrdbnet_arch",
)

_CHECKPOINT_STATE_KEYS = ("params_ema", "params", "state_dict")
_STATE_DICT_PREFIXES = (
    "backbone.",
    "generator.",
    "model.backbone.",
    "model.generator.",
    "module.backbone.",
    "module.generator.",
    "module.",
    "model.",
)


class RealESRGANGenerator(nn.Module):
    """Project wrapper for the official Real-ESRGAN RRDBNet generator.

    The wrapper owns only architecture construction, optional pretrained
    weight loading, and target-module discovery metadata. It intentionally
    excludes discriminator, GAN-loss, optimizer, metric, trainer, and evaluator
    logic.

    Args:
        scale: Super-resolution scale factor used by upstream RRDBNet.
        in_chans: Number of input image channels.
        out_chans: Number of output image channels.
        num_feat: Number of intermediate feature channels.
        num_block: Number of RRDB blocks in the trunk.
        num_grow_ch: Growth channels inside each residual dense block.
        weights_path: Optional checkpoint path to load into the generator.
        strict_load: Passed to ``load_state_dict`` for pretrained weights.
        implementation: Must be ``"upstream"`` for this wrapper.
        upstream_source: Human-readable upstream source URL or vendor path.
        upstream_commit: Optional upstream commit hash pinned by configuration.
        upstream_version: Optional upstream release or package version pinned by
            configuration.
        finetune_target_modules: Module-name prefixes or suffixes future
            factory/strategy code can use for partial generator fine-tuning.
        lora_target_modules: Optional module-name suffixes reserved for future
            adapter strategies.
        name: Optional Hydra metadata accepted for full-config instantiation.
        target: Optional Hydra metadata accepted for full-config instantiation.
        upscale: Backward-compatible alias for ``scale``.
        img_range: Optional Hydra metadata retained for config compatibility.
        body_res_scale: Optional Hydra metadata retained for compatibility with
            earlier local architecture configs.
        block_res_scale: Optional Hydra metadata retained for compatibility with
            earlier local architecture configs.
        upsample_mode: Optional Hydra metadata retained for compatibility with
            earlier local architecture configs.
        leaky_relu_negative_slope: Optional Hydra metadata retained for
            compatibility with earlier local architecture configs.
    """

    def __init__(
        self,
        scale: int,
        in_chans: int = 3,
        out_chans: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        weights_path: str | Path | None = None,
        strict_load: bool = True,
        implementation: str = "upstream",
        upstream_source: str = "",
        upstream_commit: str | None = None,
        upstream_version: str | None = None,
        finetune_target_modules: Sequence[str] | None = None,
        lora_target_modules: Sequence[str] | None = None,
        name: str = "realesrgan",
        target: str | None = None,
        upscale: int | None = None,
        img_range: float = 1.0,
        body_res_scale: float | None = None,
        block_res_scale: float | None = None,
        upsample_mode: str | None = None,
        leaky_relu_negative_slope: float | None = None,
    ) -> None:
        super().__init__()

        resolved_scale = int(scale if upscale is None else upscale)
        if resolved_scale != int(scale):
            raise ValueError(
                "`scale` and backward-compatible `upscale` must match when both "
                f"are provided, got scale={scale} and upscale={upscale}."
            )
        if implementation != "upstream":
            raise ValueError(
                "`implementation` must be 'upstream' for RealESRGANGenerator, "
                f"got {implementation!r}."
            )

        self.scale = resolved_scale
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.img_range = float(img_range)
        self.upstream_source = upstream_source
        self.upstream_commit = upstream_commit
        self.upstream_version = upstream_version
        self.name = name
        self.target = target
        self.body_res_scale = body_res_scale
        self.block_res_scale = block_res_scale
        self.upsample_mode = upsample_mode
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self._finetune_target_modules = tuple(
            finetune_target_modules
            or ("body", "conv_body", "conv_up1", "conv_up2", "conv_hr", "conv_last")
        )
        self._lora_target_modules = tuple(lora_target_modules or ())
        self.pretrained_load_report: dict[str, list[str]] | None = None

        upstream_rrdbnet = _resolve_upstream_rrdbnet()
        self.backbone = upstream_rrdbnet(
            num_in_ch=self.in_chans,
            num_out_ch=self.out_chans,
            scale=self.scale,
            num_feat=int(num_feat),
            num_block=int(num_block),
            num_grow_ch=int(num_grow_ch),
        )

        if weights_path is not None and str(weights_path):
            self.pretrained_load_report = _load_pretrained_weights(
                self.backbone,
                Path(weights_path),
                strict=bool(strict_load),
            )

    @property
    def finetune_target_modules(self) -> tuple[str, ...]:
        """Module-name patterns intended for partial generator fine-tuning."""

        return self._finetune_target_modules

    @property
    def lora_target_modules(self) -> tuple[str, ...]:
        """Module-name suffixes reserved for future adapter injection."""

        return self._lora_target_modules

    def iter_finetune_target_modules(self) -> Iterator[tuple[str, nn.Module]]:
        """Yield modules matching configured fine-tuning target names.

        Yields:
            Tuples of ``(module_name, module)`` whose names match
            ``finetune_target_modules``.
        """

        for module_name, module in self.named_modules():
            if module_name and _matches_target_module(
                module_name,
                self._finetune_target_modules,
            ):
                yield module_name, module

    def iter_lora_target_modules(self) -> Iterator[tuple[str, nn.Module]]:
        """Yield current modules matching configured LoRA target suffixes.

        Yields:
            Tuples of ``(module_name, module)`` for modules whose names match
            ``lora_target_modules``.
        """

        for module_name, module in self.named_modules():
            if module_name and _matches_target_module(
                module_name,
                self._lora_target_modules,
            ):
                yield module_name, module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run super-resolution on an LR tensor.

        Args:
            x: Low-resolution image batch with shape ``[B, C, H, W]``.

        Returns:
            Super-resolved image batch with shape
            ``[B, C, H * scale, W * scale]``.
        """

        if x.ndim != 4:
            raise ValueError(f"`x` must have shape [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"`x` must have {self.in_chans} channels, got {x.shape[1]}."
            )

        sr = self.backbone(x)  # [B, C, H * scale, W * scale]
        return sr


def _resolve_upstream_rrdbnet() -> type[nn.Module]:
    """Import the official RRDBNet class from installed or vendored upstream."""

    import_errors: list[str] = []
    for module_name in _UPSTREAM_IMPORT_CANDIDATES:
        try:
            module = import_module(module_name)
        except ImportError as error:
            import_errors.append(f"{module_name}: {error}")
            continue

        rrdbnet_class = getattr(module, "RRDBNet", None)
        if isinstance(rrdbnet_class, type) and issubclass(rrdbnet_class, nn.Module):
            return rrdbnet_class
        import_errors.append(f"{module_name}: missing nn.Module class `RRDBNet`")

    expected_sources = ", ".join(_UPSTREAM_IMPORT_CANDIDATES)
    details = "; ".join(import_errors)
    raise ImportError(
        "Official Real-ESRGAN RRDBNet generator is not importable. Install "
        "`basicsr` from the Real-ESRGAN dependency stack, or vendor BasicSR so "
        "that `basicsr/archs/rrdbnet_arch.py` is on PYTHONPATH. Accepted import "
        f"paths: {expected_sources}. Import attempts: {details}"
    )


def _load_pretrained_weights(
    model: nn.Module,
    weights_path: Path,
    strict: bool,
) -> dict[str, list[str]]:
    """Load pretrained weights into the RealESRGAN model."""

    if not weights_path.is_file():
        raise FileNotFoundError(f"RealESRGAN weights file missing: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    normalized_state_dict = _strip_state_dict_prefixes(state_dict)
    adapted_state_dict = _adapt_state_dict_channels(normalized_state_dict, model)
    load_result = model.load_state_dict(adapted_state_dict, strict=strict)
    return {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _adapt_state_dict_channels(
    state_dict: dict[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    """Adapt 3-channel pretrained weights to 1-channel models if needed."""

    adapted = dict(state_dict)

    if "conv_first.weight" in adapted and "conv_first.weight" in model.state_dict():
        pretrained_weight = adapted["conv_first.weight"]
        target_weight = model.state_dict()["conv_first.weight"]
        if pretrained_weight.shape[1] == 3 and target_weight.shape[1] == 1:
            # Preserve responses for grayscale inputs repeated across RGB channels.
            adapted["conv_first.weight"] = pretrained_weight.sum(dim=1, keepdim=True)

    if "conv_last.weight" in adapted and "conv_last.weight" in model.state_dict():
        pretrained_weight = adapted["conv_last.weight"]
        target_weight = model.state_dict()["conv_last.weight"]
        if pretrained_weight.shape[0] == 3 and target_weight.shape[0] == 1:
            # Collapse RGB output filters into a single grayscale predictor.
            adapted["conv_last.weight"] = pretrained_weight.mean(dim=0, keepdim=True)

        if "conv_last.bias" in adapted and "conv_last.bias" in model.state_dict():
            pretrained_bias = adapted["conv_last.bias"]
            target_bias = model.state_dict()["conv_last.bias"]
            if pretrained_bias.shape[0] == 3 and target_bias.shape[0] == 1:
                adapted["conv_last.bias"] = pretrained_bias.mean(dim=0, keepdim=True)

    return adapted


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Extract a model state dict from common Real-ESRGAN checkpoint formats."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Real-ESRGAN checkpoint must be a mapping containing `params_ema`, "
            "`params`, `state_dict`, or a plain state dict."
        )

    for key in _CHECKPOINT_STATE_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    if all(isinstance(key, str) for key in checkpoint):
        return checkpoint

    raise KeyError(
        "Real-ESRGAN checkpoint mapping does not contain `params_ema`, `params`, "
        "`state_dict`, or plain string state-dict keys."
    )


def _strip_state_dict_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Strip common wrapper prefixes before loading into upstream RRDBNet."""

    normalized: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        previous_key = None
        while previous_key != key:
            previous_key = key
            for prefix in _STATE_DICT_PREFIXES:
                if key.startswith(prefix):
                    key = key.removeprefix(prefix)
        normalized[key] = value
    return normalized


def _matches_target_module(module_name: str, target_modules: Sequence[str]) -> bool:
    """Return whether a module name matches a configured target pattern."""

    for target in target_modules:
        normalized_target = target.removeprefix(".")
        if module_name == normalized_target:
            return True
        if module_name.startswith(f"{normalized_target}."):
            return True
        if module_name.endswith(f".{normalized_target}"):
            return True
    return False
