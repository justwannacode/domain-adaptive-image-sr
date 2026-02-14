"""LightningDataModule for super-resolution datasets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from domain_adaptive_image_sr.data.datasets import (
    Sample,
    SyntheticSRDataset,
    UnpairedDomainDataset,
)
from domain_adaptive_image_sr.data.degradation import DegradationPipeline

Stage = Literal["fit", "validate", "test", "predict"]
Batch = dict[str, Tensor | list[str] | dict[str, Any]]


class SuperResolutionDataModule(LightningDataModule):
    """Orchestrate SR datasets and dataloaders from Hydra config.

    Args:
        config: Root Hydra config with ``config.data`` or the ``config.data``
            section itself.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.data_config = config.data if "data" in config else config
        self.train_dataset: Dataset[Sample] | None = None
        self.val_dataset: Dataset[Sample] | None = None
        self.test_dataset: Dataset[Sample] | None = None
        self.predict_dataset: Dataset[Sample] | None = None
        self.degradation_pipeline: DegradationPipeline | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested Lightning stage.

        Args:
            stage: Lightning stage name. Supported values are ``"fit"``,
                ``"validate"``, ``"test"``, and ``"predict"``.
        """

        normalized_stage = self._normalize_stage(stage)

        if normalized_stage in {None, "fit"}:
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_optional_dataset("val")
        if normalized_stage in {None, "validate"}:
            self.val_dataset = self._build_dataset("val")
        if normalized_stage in {None, "test"}:
            self.test_dataset = self._build_dataset("test")
        if normalized_stage in {None, "predict"}:
            self.predict_dataset = self._build_optional_dataset("predict")
            if self.predict_dataset is None:
                self.predict_dataset = self._build_dataset("test")

    def train_dataloader(self) -> DataLoader[Batch]:
        """Return the training dataloader."""

        return self._build_dataloader(
            self._require_dataset(self.train_dataset, "train"),
            split="train",
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        """Return the validation dataloader."""

        return self._build_dataloader(
            self._require_dataset(self.val_dataset, "val"),
            split="val",
        )

    def test_dataloader(self) -> DataLoader[Batch]:
        """Return the test dataloader."""

        return self._build_dataloader(
            self._require_dataset(self.test_dataset, "test"),
            split="test",
        )

    def predict_dataloader(self) -> DataLoader[Batch]:
        """Return the prediction dataloader."""

        return self._build_dataloader(
            self._require_dataset(self.predict_dataset, "predict"),
            split="predict",
        )

    def _build_dataset(self, split: str) -> Dataset[Sample]:
        """Instantiate a dataset for a split based on ``config.data.mode``."""

        mode = str(self.data_config.mode)
        if mode == "synthetic":
            return SyntheticSRDataset(
                self.data_config,
                split=split,
                degradation_pipeline=self._get_degradation_pipeline(),
            )
        if mode == "unpaired":
            return UnpairedDomainDataset(self.data_config, split=split)

        raise ValueError(
            f"`config.data.mode` must be one of ['synthetic', 'unpaired'], got {mode}."
        )

    def _build_optional_dataset(self, split: str) -> Dataset[Sample] | None:
        """Build a dataset when the split exists and has a configured root."""

        if split not in self.data_config:
            return None

        split_config = self.data_config[split]
        mode = str(self.data_config.mode)
        if mode == "synthetic" and split_config.hr_dir is None:
            return None
        if (
            mode == "unpaired"
            and split_config.lq_dir is None
            and split_config.lr_dir is None
        ):
            return None
        return self._build_dataset(split)

    def _get_degradation_pipeline(self) -> DegradationPipeline:
        """Create or return the shared synthetic degradation pipeline."""

        if self.degradation_pipeline is None:
            self.degradation_pipeline = DegradationPipeline(
                self.data_config.degradation,
                domain=str(self.data_config.get("name", "")) or None,
            )
        return self.degradation_pipeline

    def _build_dataloader(
        self,
        dataset: Dataset[Sample],
        *,
        split: str,
    ) -> DataLoader[Batch]:
        """Create a DataLoader using Hydra dataloader settings."""

        split_loader_config = self._get_dataloader_split_config(split)
        return DataLoader(
            dataset,
            batch_size=int(self.data_config.batch_size),
            shuffle=bool(split_loader_config.shuffle),
            num_workers=int(self.data_config.num_workers),
            pin_memory=bool(self.data_config.pin_memory),
            persistent_workers=self._persistent_workers_enabled(),
            drop_last=bool(split_loader_config.drop_last),
            collate_fn=self._get_collate_fn(),
        )

    def _get_dataloader_split_config(self, split: str) -> DictConfig:
        """Resolve DataLoader settings for a split."""

        dataloader_config = self.data_config.dataloader
        if split in dataloader_config:
            return dataloader_config[split]
        if split == "predict" and "test" in dataloader_config:
            return dataloader_config.test
        raise ValueError(f"`config.data.dataloader.{split}` is not configured.")

    def _get_collate_fn(self) -> Callable[[list[Sample]], Batch]:
        """Return a collate function selected by Hydra config."""

        collate_name = str(self.data_config.dataloader.collate_fn)
        if collate_name == "super_resolution":
            return collate_super_resolution_samples
        raise ValueError(
            "`config.data.dataloader.collate_fn` must be 'super_resolution', "
            f"got {collate_name}."
        )

    def _persistent_workers_enabled(self) -> bool:
        """Return a valid persistent workers flag for current worker count."""

        return (
            bool(self.data_config.persistent_workers)
            and int(self.data_config.num_workers) > 0
        )

    @staticmethod
    def _require_dataset(
        dataset: Dataset[Sample] | None,
        split: str,
    ) -> Dataset[Sample]:
        """Return a dataset or raise a setup error."""

        if dataset is None:
            raise RuntimeError(
                f"The {split} dataset is not initialized. Call setup() first."
            )
        return dataset

    @staticmethod
    def _normalize_stage(stage: str | None) -> Stage | None:
        """Normalize Lightning stage names used by different entrypoints."""

        if stage is None:
            return None
        if stage in {"fit", "validate", "test", "predict"}:
            return cast(Stage, stage)
        if stage == "val":
            return "validate"
        raise ValueError(f"Unsupported datamodule stage: {stage}")


def collate_super_resolution_samples(samples: list[Sample]) -> Batch:
    """Collate SR samples into a batch dictionary.

    Args:
        samples: List of sample dictionaries returned by SR datasets.

    Returns:
        Batch dictionary with ``lr`` and ``hr`` tensors plus per-image metadata.
    """

    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    lr_tensors = [_get_tensor(sample, "lr") for sample in samples]
    lr = torch.stack(lr_tensors)  # [B, C, H_lr, W_lr]
    hr_tensors = [_get_tensor(sample, "hr") for sample in samples]
    hr = _collate_hr_tensors(hr_tensors)  # [B, C, H_hr, W_hr] or [B, 0]

    return {
        "lr": lr,
        "hr": hr,
        "image_id": [str(sample["image_id"]) for sample in samples],
        "metadata": {
            key: [sample["metadata"][key] for sample in samples]
            for key in samples[0]["metadata"]
        },
    }


def _get_tensor(sample: Sample, key: str) -> Tensor:
    """Return a tensor value from a sample."""

    value = sample[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"`{key}` sample value must be a tensor.")
    return value


def _collate_hr_tensors(hr_tensors: list[Tensor]) -> Tensor:
    """Stack HR tensors while supporting unpaired empty placeholders."""

    if all(tensor.numel() == 0 for tensor in hr_tensors):
        return torch.stack(hr_tensors)  # [B, 0]
    return torch.stack(hr_tensors)  # [B, C, H_hr, W_hr]


__all__ = ["SuperResolutionDataModule", "collate_super_resolution_samples"]
