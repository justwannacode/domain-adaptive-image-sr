"""Model factory and adaptation strategy orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from inspect import Parameter, signature
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import nn

from domain_adaptive_image_sr.models.adapters import get_adapter_strategy

_NO_ADAPTER_STRATEGIES = {"baseline", "none", "full_finetune", "partial_finetune"}
_MODEL_METADATA_KEYS = {"target"}


def get_model(config: DictConfig) -> nn.Module:
    """Build a model and apply the configured adaptation strategy.

    Args:
        config: Root Hydra config containing ``model`` and ``strategy`` nodes.

    Returns:
        Model ready to be passed to the trainer.
    """

    model_config = _get_required_node(config, "model")
    strategy_config = _get_optional_node(config, "strategy")
    model = _build_model(model_config)

    if strategy_config is None:
        return model

    if bool(strategy_config.freeze_backbone):
        freeze_parameters(model)

    strategy_type = str(strategy_config.type).strip().lower()
    if strategy_type in _NO_ADAPTER_STRATEGIES:
        _unfreeze_configured_targets(model, strategy_config)
        return model

    adapter_strategy = get_adapter_strategy(strategy_type)
    return adapter_strategy(model, strategy_config)


class ModelFactory:
    """Small callable factory wrapper around ``get_model``."""

    def get_model(self, config: DictConfig) -> nn.Module:
        """Build a model from a root Hydra config.

        Args:
            config: Root Hydra config containing model and strategy sections.

        Returns:
            Configured PyTorch model.
        """

        return get_model(config)


def freeze_parameters(model: nn.Module) -> None:
    """Freeze all parameters in a model."""

    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_target_modules(
    model: nn.Module,
    target_modules: Sequence[str],
) -> tuple[str, ...]:
    """Unfreeze parameters for modules matching target patterns.

    Args:
        model: Model whose modules should be partially unfrozen.
        target_modules: Module-name prefixes or suffixes to unfreeze.

    Returns:
        Names of modules whose direct parameters were unfrozen.
    """

    normalized_targets = _normalize_targets(target_modules)
    if not normalized_targets:
        return ()

    unfrozen_modules: list[str] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not _matches_target_module(module_name, normalized_targets):
            continue

        changed = False
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = True
            changed = True
        if changed:
            unfrozen_modules.append(module_name)

    if not unfrozen_modules:
        raise ValueError(
            "No modules were unfrozen for target patterns: "
            f"{', '.join(normalized_targets)}."
        )

    return tuple(unfrozen_modules)


def _build_model(model_config: DictConfig) -> nn.Module:
    """Instantiate the architecture declared by ``config.model.target``."""

    target = str(model_config.target)
    model_class = _import_symbol(target)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise TypeError(f"`config.model.target` must point to nn.Module, got {target}.")

    raw_kwargs = OmegaConf.to_container(model_config, resolve=True)
    if not isinstance(raw_kwargs, Mapping):
        raise TypeError("`config.model` must resolve to a mapping.")

    kwargs = _filter_constructor_kwargs(model_class, raw_kwargs)
    return model_class(**kwargs)


def _filter_constructor_kwargs(
    model_class: type[nn.Module],
    raw_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Drop factory metadata and unsupported constructor keys."""

    constructor_signature = signature(model_class.__init__)
    parameters = constructor_signature.parameters
    accepts_kwargs = any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    filtered: dict[str, Any] = {}
    for key, value in raw_kwargs.items():
        if key in _MODEL_METADATA_KEYS:
            continue
        if accepts_kwargs or key in parameters:
            filtered[str(key)] = value
    return filtered


def _import_symbol(target: str) -> object:
    """Import a Python symbol from a dotted path."""

    module_name, separator, symbol_name = target.rpartition(".")
    if not separator:
        raise ValueError(f"Target path must include a module and symbol: {target!r}.")

    module = import_module(module_name)
    return getattr(module, symbol_name)


def _get_required_node(config: DictConfig, key: str) -> DictConfig:
    """Return a required DictConfig node."""

    node = config.get(key)
    if not isinstance(node, DictConfig):
        raise TypeError(f"`config.{key}` must be a DictConfig.")
    return node


def _get_optional_node(config: DictConfig, key: str) -> DictConfig | None:
    """Return an optional DictConfig node."""

    node = config.get(key)
    if node is None:
        return None
    if not isinstance(node, DictConfig):
        raise TypeError(f"`config.{key}` must be a DictConfig when provided.")
    return node


def _unfreeze_configured_targets(
    model: nn.Module,
    strategy_config: DictConfig,
) -> None:
    """Unfreeze explicitly configured strategy target modules."""

    target_modules = tuple(str(item) for item in strategy_config.target_modules)
    strategy_type = str(strategy_config.type).strip().lower()
    if not target_modules and strategy_type == "partial_finetune":
        target_modules = tuple(
            str(item) for item in getattr(model, "finetune_target_modules", ())
        )
    if not target_modules:
        return

    unfrozen_modules = unfreeze_target_modules(model, target_modules)
    setattr(model, "unfrozen_target_modules", unfrozen_modules)


def _normalize_targets(target_modules: Sequence[str]) -> tuple[str, ...]:
    """Normalize module target patterns."""

    return tuple(
        target.strip().removeprefix(".")
        for target in target_modules
        if target and target.strip()
    )


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
