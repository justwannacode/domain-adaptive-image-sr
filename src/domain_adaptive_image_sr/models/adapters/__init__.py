"""Adapter strategy registry for model adaptation."""

from __future__ import annotations

from collections.abc import Callable

from omegaconf import DictConfig
from torch import nn

AdapterStrategy = Callable[[nn.Module, DictConfig], nn.Module]

_ADAPTER_STRATEGIES: dict[str, AdapterStrategy] = {}


def register_adapter_strategy(
    name: str,
) -> Callable[[AdapterStrategy], AdapterStrategy]:
    """Register an adapter strategy by config name.

    Args:
        name: Strategy name used in ``config.strategy.type``.

    Returns:
        Decorator that stores the strategy function in the adapter registry.
    """

    normalized_name = name.strip().lower()

    def decorator(strategy: AdapterStrategy) -> AdapterStrategy:
        """Store a strategy function in the adapter registry."""

        if normalized_name in _ADAPTER_STRATEGIES:
            raise ValueError(f"Adapter strategy {normalized_name!r} is already set.")
        _ADAPTER_STRATEGIES[normalized_name] = strategy
        return strategy

    return decorator


def get_adapter_strategy(name: str) -> AdapterStrategy:
    """Fetch a registered adapter strategy.

    Args:
        name: Strategy name from ``config.strategy.type``.

    Returns:
        Registered strategy callable.
    """

    normalized_name = name.strip().lower()
    try:
        return _ADAPTER_STRATEGIES[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(_ADAPTER_STRATEGIES)) or "<none>"
        raise KeyError(
            f"Unknown adapter strategy {name!r}. Available strategies: {available}."
        ) from error


def list_adapter_strategies() -> tuple[str, ...]:
    """Return registered adapter strategy names."""

    return tuple(sorted(_ADAPTER_STRATEGIES))


from domain_adaptive_image_sr.models.adapters.lora import (  # noqa: E402
    LoRALinear,
    LoRAInjectionReport,
    inject_lora_adapters,
)

__all__ = [
    "AdapterStrategy",
    "LoRALinear",
    "LoRAInjectionReport",
    "get_adapter_strategy",
    "inject_lora_adapters",
    "list_adapter_strategies",
    "register_adapter_strategy",
]
