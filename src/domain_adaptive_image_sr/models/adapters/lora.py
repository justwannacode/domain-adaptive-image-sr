"""LoRA layers and injection utilities for linear attention projections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

from domain_adaptive_image_sr.models.adapters import register_adapter_strategy


@dataclass(frozen=True)
class LoRAInjectionReport:
    """Summary of LoRA adapter injection.

    Attributes:
        target_modules: Configured target module patterns.
        injected_modules: Fully qualified module names replaced by LoRA.
    """

    target_modules: tuple[str, ...]
    injected_modules: tuple[str, ...]

    @property
    def num_injected(self) -> int:
        """Number of modules replaced with LoRA wrappers."""

        return len(self.injected_modules)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for ``torch.nn.Linear``.

    The base linear layer remains available as ``base_layer`` and can be frozen
    independently from the trainable LoRA matrices.

    Args:
        base_layer: Existing linear projection to wrap.
        rank: Low-rank bottleneck dimension.
        alpha: LoRA scaling numerator. Effective scale is ``alpha / rank``.
        dropout: Dropout probability applied before the LoRA down projection.
        freeze_base: Whether to freeze the wrapped linear layer parameters.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"`rank` must be >= 1, got {rank}.")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"`dropout` must be in [0, 1], got {dropout}.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(p=float(dropout))
        self.lora_down = nn.Linear(
            base_layer.in_features,
            self.rank,
            bias=False,
        )
        self.lora_up = nn.Linear(
            self.rank,
            base_layer.out_features,
            bias=False,
        )
        self.reset_parameters()

        if freeze_base:
            for parameter in self.base_layer.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        """Initialize LoRA matrices with a no-op initial residual."""

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection plus trainable low-rank residual.

        Args:
            x: Input tensor with shape ``[..., in_features]``.

        Returns:
            Output tensor with shape ``[..., out_features]``.
        """

        base = self.base_layer(x)  # [..., out_features]
        down = self.lora_down(self.dropout(x))  # [..., rank]
        residual = self.lora_up(down) * self.scaling  # [..., out_features]
        return base + residual  # [..., out_features]


class LoRAConv2d(nn.Module):
    """Low-rank adapter wrapper for ``torch.nn.Conv2d``.

    Args:
        base_layer: Existing Conv2d layer to wrap.
        rank: Low-rank bottleneck dimension.
        alpha: LoRA scaling numerator.
        dropout: Dropout probability.
        freeze_base: Whether to freeze the wrapped base layer.
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int,
        alpha: float,
        dropout: float,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"`rank` must be >= 1, got {rank}.")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"`dropout` must be in [0, 1], got {dropout}.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout2d(p=float(dropout))

        # Preserve the wrapped convolution geometry in the down projection.
        self.lora_down = nn.Conv2d(
            in_channels=base_layer.in_channels,
            out_channels=self.rank,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
            bias=False,
        )

        # Restore the original channel count without changing spatial size.
        self.lora_up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.reset_parameters()

        if freeze_base:
            for parameter in self.base_layer.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        """Initialize LoRA matrices with a no-op initial residual."""
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection plus trainable low-rank residual."""
        base = self.base_layer(x)
        down = self.lora_down(self.dropout(x))
        residual = self.lora_up(down) * self.scaling
        return base + residual


def inject_lora_adapters(
    model: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    freeze_base: bool = True,
    allow_missing: bool = False,
) -> LoRAInjectionReport:
    """Replace matching linear and conv modules with LoRA wrappers.

    Args:
        model: Model whose submodules should be adapted.
        target_modules: Module-name patterns, usually suffixes like
            ``"attn.qkv"`` for transformer Q/K/V projections.
        rank: Low-rank bottleneck dimension.
        alpha: LoRA scaling numerator.
        dropout: Dropout probability used in LoRA branch.
        freeze_base: Whether wrapped base linear projections should be frozen.
        allow_missing: If ``False``, raise when no matching modules are found.

    Returns:
        Injection report with replaced module names.
    """

    normalized_targets = _normalize_targets(target_modules)
    injected_modules: list[str] = []

    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue

        if isinstance(module, nn.Linear):
            wrapper_cls = LoRALinear
        elif isinstance(module, nn.Conv2d):
            wrapper_cls = LoRAConv2d
        else:
            continue

        if not _matches_target_module(module_name, normalized_targets):
            continue

        parent, child_name = _get_parent_module(model, module_name)
        setattr(
            parent,
            child_name,
            wrapper_cls(
                base_layer=module,
                rank=int(rank),
                alpha=float(alpha),
                dropout=float(dropout),
                freeze_base=bool(freeze_base),
            ),
        )
        injected_modules.append(module_name)

    if not injected_modules and not allow_missing:
        raise ValueError(
            "LoRA injection did not find any matching `nn.Linear` or `nn.Conv2d` modules for "
            f"targets: {', '.join(normalized_targets)}."
        )

    return LoRAInjectionReport(
        target_modules=tuple(normalized_targets),
        injected_modules=tuple(injected_modules),
    )


@register_adapter_strategy("lora")
def apply_lora_strategy(model: nn.Module, strategy_config: DictConfig) -> nn.Module:
    """Apply LoRA adapters according to ``config.strategy``.

    Args:
        model: Base model built by the model factory.
        strategy_config: Hydra strategy config with LoRA hyperparameters.

    Returns:
        The same model instance after in-place adapter injection.
    """

    target_modules = _resolve_strategy_targets(model, strategy_config)
    report = inject_lora_adapters(
        model=model,
        target_modules=target_modules,
        rank=int(strategy_config.rank),
        alpha=float(strategy_config.alpha),
        dropout=float(strategy_config.dropout),
        freeze_base=bool(
            strategy_config.get("freeze_base", strategy_config.freeze_backbone)
        ),
        allow_missing=bool(strategy_config.get("allow_missing_targets", False)),
    )
    setattr(model, "adapter_report", report)
    return model


def _resolve_strategy_targets(
    model: nn.Module,
    strategy_config: DictConfig,
) -> tuple[str, ...]:
    """Resolve LoRA targets from strategy config or model metadata."""

    configured_targets = tuple(str(item) for item in strategy_config.target_modules)
    if configured_targets:
        return configured_targets

    model_targets = getattr(model, "lora_target_modules", ())
    resolved_targets = tuple(str(item) for item in model_targets)
    if resolved_targets:
        return resolved_targets

    raise ValueError(
        "`strategy.target_modules` is empty and the model does not expose "
        "`lora_target_modules`."
    )


def _normalize_targets(target_modules: Sequence[str]) -> tuple[str, ...]:
    """Normalize target module patterns from Hydra config."""

    normalized = tuple(
        target.strip().removeprefix(".")
        for target in target_modules
        if target and target.strip()
    )
    if not normalized:
        raise ValueError("At least one LoRA target module must be configured.")
    return normalized


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    """Return parent module and child name for a dotted module path."""

    path = module_name.split(".")
    parent = model
    for part in path[:-1]:
        parent = getattr(parent, part)
    return parent, path[-1]


def _matches_target_module(module_name: str, target_modules: Sequence[str]) -> bool:
    """Return whether a module name matches a configured target pattern."""

    for target in target_modules:
        if module_name == target:
            return True
        if module_name.startswith(f"{target}."):
            return True
        if f".{target}." in module_name:
            return True
        if module_name.endswith(f".{target}"):
            return True
    return False
