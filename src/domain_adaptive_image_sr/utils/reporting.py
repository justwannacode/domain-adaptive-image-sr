"""Reporting helpers for experiment metric exports."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


MetricTable = Mapping[str, Mapping[str, float]]


def export_metrics_csv(
    metrics_by_stage: MetricTable,
    output_dir: str | Path,
    filename: str = "metrics.csv",
) -> Path:
    """Export stage-level metrics to ``outputs/[experiment_name]/metrics.csv``.

    Args:
        metrics_by_stage: Mapping from stage name to metric-name/value mapping.
        output_dir: Experiment output directory, usually
            ``outputs/[experiment_name]``.
        filename: CSV filename within ``output_dir``.

    Returns:
        Path to the written CSV file.
    """

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / filename

    metric_names = _collect_metric_names(metrics_by_stage)
    fieldnames = ["stage", *metric_names]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for stage, metrics in metrics_by_stage.items():
            row: dict[str, str | float] = {"stage": stage}
            for metric_name in metric_names:
                if metric_name in metrics:
                    row[metric_name] = float(metrics[metric_name])
                else:
                    row[metric_name] = ""
            writer.writerow(row)

    return csv_path


def _collect_metric_names(metrics_by_stage: MetricTable) -> list[str]:
    """Collect metric names in deterministic first-seen order."""

    metric_names: list[str] = []
    seen: set[str] = set()
    for metrics in metrics_by_stage.values():
        for metric_name in metrics:
            if metric_name not in seen:
                metric_names.append(metric_name)
                seen.add(metric_name)
    return metric_names


__all__ = ["MetricTable", "export_metrics_csv"]
