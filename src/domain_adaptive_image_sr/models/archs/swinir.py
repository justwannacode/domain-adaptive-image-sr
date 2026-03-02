"""Thin wrapper around the official SwinIR architecture."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch import nn

_UPSTREAM_IMPORT_CANDIDATES = (
    "models.network_swinir",
    "network_swinir",
    "SwinIR.models.network_swinir",
    "domain_adaptive_image_sr.vendor.swinir.models.network_swinir",
    "domain_adaptive_image_sr.vendor.SwinIR.models.network_swinir",
)

_CHECKPOINT_STATE_KEYS = ("params_ema", "params", "state_dict")
_STATE_DICT_PREFIXES = (
    "backbone.",
    "model.backbone.",
    "module.backbone.",
    "module.",
    "model.",
)


class SwinIR(nn.Module):
    """Project wrapper for the upstream SwinIR implementation.

    The wrapper owns no training, optimizer, loss, metric, or evaluator logic.
    It only builds the official SwinIR network, optionally loads pretrained
    weights, and exposes stable target-module names for later LoRA injection.

    Args:
        scale: Super-resolution scale factor.
        img_size: Training patch size used by the upstream SwinIR constructor.
        patch_size: Patch size used by the upstream SwinIR constructor.
        in_chans: Number of input image channels.
        out_chans: Expected number of output channels. Official SwinIR uses the
            same channel count for input and output, so this must match
            ``in_chans`` when provided.
        window_size: Local attention window size.
        img_range: Pixel range multiplier used by upstream SwinIR.
        depths: Number of transformer blocks per residual Swin Transformer
            stage.
        embed_dim: Feature embedding dimension.
        num_heads: Number of attention heads per stage.
        mlp_ratio: Hidden-size ratio in transformer MLP blocks.
        upsampler: Upstream reconstruction head name.
        resi_connection: Upstream residual connection mode.
        qkv_bias: Whether QKV linear projections use bias.
        qk_scale: Optional QK attention scale.
        drop_rate: Dropout probability.
        attn_drop_rate: Attention dropout probability.
        drop_path_rate: Stochastic depth probability.
        ape: Whether to use absolute position embeddings.
        patch_norm: Whether patch embeddings use normalization.
        use_checkpoint: Whether upstream transformer blocks use gradient
            checkpointing.
        weights_path: Optional checkpoint path to load into the upstream model.
        strict_load: Passed to ``load_state_dict`` for pretrained weights.
        implementation: Must be ``"upstream"`` for this wrapper.
        upstream_source: Human-readable upstream source URL or vendor path.
        upstream_commit: Optional upstream commit hash pinned by configuration.
        upstream_version: Optional upstream release or tag pinned by
            configuration.
        lora_target_modules: Module-name suffixes that future adapters can use
            to find Q/K/V projection layers.
        name: Optional Hydra metadata accepted for full-config instantiation.
        target: Optional Hydra metadata accepted for full-config instantiation.
        upscale: Backward-compatible alias for ``scale``.
    """

    def __init__(
        self,
        scale: int,
        img_size: int | Sequence[int] = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        out_chans: int | None = None,
        window_size: int = 8,
        img_range: float = 1.0,
        depths: Sequence[int] = (6, 6, 6, 6),
        embed_dim: int = 96,
        num_heads: Sequence[int] = (6, 6, 6, 6),
        mlp_ratio: float = 2.0,
        upsampler: str = "pixelshuffle",
        resi_connection: str = "1conv",
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        weights_path: str | Path | None = None,
        strict_load: bool = True,
        implementation: str = "upstream",
        upstream_source: str = "",
        upstream_commit: str | None = None,
        upstream_version: str | None = None,
        lora_target_modules: Sequence[str] | None = None,
        name: str = "swinir",
        target: str | None = None,
        upscale: int | None = None,
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
                "`implementation` must be 'upstream' for the SwinIR wrapper, "
                f"got {implementation!r}."
            )
        if out_chans is not None and int(out_chans) != int(in_chans):
            raise ValueError(
                "Official SwinIR uses the same input and output channel count; "
                f"got in_chans={in_chans} and out_chans={out_chans}."
            )

        self.scale = resolved_scale
        self.in_chans = int(in_chans)
        self.img_range = float(img_range)
        self.upstream_source = upstream_source
        self.upstream_commit = upstream_commit
        self.upstream_version = upstream_version
        self.name = name
        self.target = target
        self._lora_target_modules = tuple(lora_target_modules or ("attn.qkv",))
        self.pretrained_load_report: dict[str, list[str]] | None = None

        upstream_swinir = _resolve_upstream_swinir()
        self.backbone = upstream_swinir(
            img_size=_normalize_img_size(img_size),
            patch_size=int(patch_size),
            in_chans=self.in_chans,
            embed_dim=int(embed_dim),
            depths=_normalize_int_sequence(depths, "depths"),
            num_heads=_normalize_int_sequence(num_heads, "num_heads"),
            window_size=int(window_size),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=float(drop_path_rate),
            ape=bool(ape),
            patch_norm=bool(patch_norm),
            use_checkpoint=bool(use_checkpoint),
            upscale=self.scale,
            img_range=self.img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
        )

        if weights_path is not None and str(weights_path):
            self.pretrained_load_report = _load_pretrained_weights(
                self.backbone,
                Path(weights_path),
                strict=bool(strict_load),
            )

    @property
    def lora_target_modules(self) -> tuple[str, ...]:
        """Module-name suffixes intended for future LoRA adapter injection."""

        return self._lora_target_modules

    def iter_lora_target_modules(self) -> Iterator[tuple[str, nn.Module]]:
        """Yield current modules matching configured LoRA target suffixes.

        Yields:
            Tuples of ``(module_name, module)`` for linear modules whose names
            match ``lora_target_modules``.
        """

        for module_name, module in self.named_modules():
            if isinstance(module, nn.Linear) and _matches_target_module(
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


def _resolve_upstream_swinir() -> type[nn.Module]:
    """Import the official SwinIR class from an installed or vendored source."""

    import_errors: list[str] = []
    for module_name in _UPSTREAM_IMPORT_CANDIDATES:
        try:
            module = import_module(module_name)
        except ImportError as error:
            import_errors.append(f"{module_name}: {error}")
            continue

        swinir_class = getattr(module, "SwinIR", None)
        if isinstance(swinir_class, type) and issubclass(swinir_class, nn.Module):
            return swinir_class
        import_errors.append(f"{module_name}: missing nn.Module class `SwinIR`")

    expected_sources = ", ".join(_UPSTREAM_IMPORT_CANDIDATES)
    details = "; ".join(import_errors)
    raise ImportError(
        "Official SwinIR implementation is not importable. Vendor "
        "`JingyunLiang/SwinIR` so that `models/network_swinir.py` is on "
        "PYTHONPATH, or provide one of these import paths: "
        f"{expected_sources}. The official file also depends on `timm`. "
        f"Import attempts: {details}"
    )


def _normalize_img_size(img_size: int | Sequence[int]) -> int | tuple[int, int]:
    """Normalize Hydra image-size values for the upstream constructor."""

    if isinstance(img_size, int):
        return img_size

    values = tuple(int(value) for value in img_size)
    if len(values) != 2:
        raise ValueError(f"`img_size` must be an int or [height, width], got {values}.")
    return values


def _normalize_int_sequence(values: Sequence[int], field_name: str) -> list[int]:
    """Convert a Hydra sequence into a plain list of integers."""

    normalized = [int(value) for value in values]
    if not normalized:
        raise ValueError(f"`{field_name}` must contain at least one value.")
    return normalized


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


def _load_pretrained_weights(
    model: nn.Module,
    weights_path: Path,
    strict: bool,
) -> dict[str, list[str]]:
    """Load pretrained SwinIR weights into the upstream model.

    Args:
        model: Upstream SwinIR model instance.
        weights_path: Path to a checkpoint file.
        strict: Whether all checkpoint keys must exactly match.

    Returns:
        Missing and unexpected keys reported by ``load_state_dict``.
    """

    if not weights_path.is_file():
        raise FileNotFoundError(f"SwinIR weights file does not exist: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    normalized_state_dict = _strip_state_dict_prefixes(state_dict)
    adapted_state_dict = _adapt_state_dict_channels(normalized_state_dict, model)
    load_result = model.load_state_dict(adapted_state_dict, strict=strict)
    return {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Extract a model state dict from common SwinIR checkpoint formats."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "SwinIR checkpoint must be a mapping containing `params_ema`, "
            "`params`, `state_dict`, or a plain state dict."
        )

    for key in _CHECKPOINT_STATE_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    if all(isinstance(key, str) for key in checkpoint):
        return checkpoint

    raise KeyError(
        "SwinIR checkpoint mapping does not contain `params_ema`, `params`, "
        "`state_dict`, or plain string state-dict keys."
    )


def _strip_state_dict_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Strip common wrapper prefixes before loading into upstream SwinIR."""

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
    """Return whether a module name matches a configured target suffix."""

    for target in target_modules:
        normalized_target = target.removeprefix(".")
        if module_name == normalized_target or module_name.endswith(
            f".{normalized_target}"
        ):
            return True
    return False
