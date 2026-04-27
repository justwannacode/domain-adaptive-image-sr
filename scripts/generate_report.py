"""Aggregate experiment metrics into a single CSV report."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

console = Console()


def _patch_hydra_argparse_for_python314() -> None:
    """Patch argparse help handling for Hydra on Python 3.14."""

    original_expand_help = argparse.HelpFormatter._expand_help

    def expand_help(self: argparse.HelpFormatter, action: argparse.Action) -> str:
        """Convert Hydra lazy help objects to strings before argparse checks."""

        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return original_expand_help(self, action)

    argparse.HelpFormatter._expand_help = expand_help


_patch_hydra_argparse_for_python314()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Aggregate ``outputs/*/metrics.csv`` files.

    Args:
        config: Root Hydra config. Report-specific values can be passed with
            Hydra overrides under ``+report``.
    """

    search_dir = _resolve_path(
        _get_optional_string(config, ("report", "search_dir"), "outputs")
    )
    metrics_filename = _get_optional_string(
        config,
        ("report", "metrics_filename"),
        "metrics.csv",
    )
    output_path = _resolve_path(
        _get_optional_string(
            config,
            ("report", "output_path"),
            "outputs/summary.csv",
        )
    )

    metrics_paths = sorted(search_dir.glob(f"*/{metrics_filename}"))
    rows = _load_metric_rows(metrics_paths)
    if not rows:
        raise ValueError(f"No metrics rows found under: {search_dir}")

    report_path = _write_summary(rows, output_path)
    _print_summary(rows, report_path)


def _load_metric_rows(metrics_paths: Sequence[Path]) -> list[dict[str, str]]:
    """Load metric rows and attach experiment names."""

    rows: list[dict[str, str]] = []
    for metrics_path in metrics_paths:
        experiment_name = metrics_path.parent.name
        with metrics_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                normalized_row = {key: value for key, value in row.items() if key}
                normalized_row["experiment"] = experiment_name
                rows.append(normalized_row)
    return rows


def _write_summary(rows: Sequence[Mapping[str, str]], output_path: Path) -> Path:
    """Write aggregated rows into a summary CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _collect_fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def _collect_fieldnames(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Collect CSV field names with experiment and stage first."""

    preferred = ["experiment", "stage"]
    seen = set(preferred)
    fieldnames = list(preferred)
    for row in rows:
        for fieldname in row:
            if fieldname in seen:
                continue
            fieldnames.append(fieldname)
            seen.add(fieldname)
    return fieldnames


def _print_summary(rows: Sequence[Mapping[str, str]], report_path: Path) -> None:
    """Print a compact Rich table for CLI feedback."""

    fieldnames = _collect_fieldnames(rows)
    preview_fields = fieldnames[: min(len(fieldnames), 6)]
    table = Table(title=f"Aggregated metrics: {report_path}")
    for fieldname in preview_fields:
        table.add_column(fieldname)
    for row in rows[:20]:
        table.add_row(*(str(row.get(fieldname, "")) for fieldname in preview_fields))
    console.print(table)
    console.print(f"Wrote {len(rows)} rows to {report_path}")


def _resolve_path(path: str | None) -> Path:
    """Resolve a path relative to the original working directory."""

    if path is None:
        raise ValueError("Path value cannot be null.")
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return Path(get_original_cwd()) / resolved


def _get_optional_string(
    config: DictConfig,
    path: Sequence[str],
    default: str | None,
) -> str | None:
    """Read an optional non-empty string from a nested Hydra config path."""

    value = _get_optional_value(config, path, default)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _get_optional_value(
    config: DictConfig,
    path: Sequence[str],
    default: object,
) -> object:
    """Read an optional value from a nested Hydra config path."""

    node: object = config
    for key in path:
        if not isinstance(node, DictConfig) or key not in node:
            return default
        node = node[key]
    return node


if __name__ == "__main__":
    main()
