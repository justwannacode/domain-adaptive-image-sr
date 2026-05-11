# Domain Adaptive Image SR

Research code for domain-adaptive image super-resolution. The project compares
baseline SR models with adaptation strategies for domain-specific degradation
shifts, including scanned documents, retro photos and MRI-like data.

The main hypothesis is that parameter-efficient fine-tuning methods such as
LoRA and partial layer training can approach full fine-tuning quality while
training fewer than 20% of the model parameters and using less compute.

## What Is Included

- Super-resolution training and evaluation with PyTorch Lightning.
- Hydra-based experiment configuration.
- SwinIR and Real-ESRGAN/RealESRNet model wrappers.
- Adaptation strategies: baseline, LoRA and partial fine-tuning.
- Synthetic and domain-specific degradation pipelines.
- Metrics and reporting for PSNR, SSIM, LPIPS and resource-oriented logs.
- Visual quality assessment outputs with SR comparison panels and crops.

## Repository Layout

```text
configs/                         Hydra configuration groups
docs/                            PRD, architecture notes and conventions
scripts/                         Training, evaluation and report entry points
src/domain_adaptive_image_sr/     Project package
  core/                          Lightning module, evaluator and callbacks
  data/                          datasets, datamodule and degradations
  metrics/                       visual QA helpers
  models/                        model factory, architectures and adapters
  utils/                         reporting, IO, reproducibility, visualization
tests/                           unit and component tests
weights/                         local pretrained weights, ignored by git
outputs/                         Hydra/Lightning experiment outputs, ignored
```

## Requirements

- Python `>=3.11,<3.13`
- PyTorch `>=2.0`
- CUDA-capable GPU for practical training

Install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you use `uv`, the checked-in `uv.lock` can be used instead:

```bash
uv sync --extra dev
```

## Data And Weights

Datasets are expected under `data/`. The
base data config accepts paired or synthetic SR inputs:

- `data.train.hr_dir`: high-resolution training images
- `data.train.lr_dir`: low-resolution training images for paired data
- `data.val.hr_dir`, `data.val.lr_dir`: validation split
- `data.test.hr_dir`, `data.test.lr_dir`: test split

Pretrained weights are expected under `weights/`.
The model configs currently reference:

- `weights/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth`
- `weights/realesrgan/RealESRNet_x4plus.pth`

Use `weights/download_weights.sh` or place compatible checkpoints manually.

## Running Experiments

The default entry point is `scripts/run_experiment.py`. Hydra writes each run to
`outputs/${experiment.name}` and stores the resolved config in `.hydra/`.

Fast smoke test:

```bash
python scripts/run_experiment.py trainer.fast_dev_run=true
```

Train SwinIR on the default synthetic setup:

```bash
python scripts/run_experiment.py experiment.name=swinir_baseline
```

Run LoRA adaptation on scanned documents:

```bash
python scripts/run_experiment.py \
  data=docs \
  model=swinir \
  strategy=lora \
  experiment.name=docs_swinir_lora
```

Run partial fine-tuning on MRI data:

```bash
python scripts/run_experiment.py \
  data=mri \
  model=swinir \
  strategy=partial \
  experiment.name=mri_swinir_partial
```

Override dataset paths from the command line:

```bash
python scripts/run_experiment.py \
  data=docs \
  data.train.hr_dir=/path/to/train/hr \
  data.val.hr_dir=/path/to/val/hr \
  data.test.hr_dir=/path/to/test/hr
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/run_evaluation.py \
  data=docs \
  model=swinir \
  strategy=lora \
  evaluation.checkpoint_path=outputs/docs_swinir_lora/checkpoints/last.ckpt \
  evaluation.stage=test \
  experiment.name=eval_docs_swinir_lora
```

Evaluate a base model without a project checkpoint:

```bash
python scripts/run_evaluation.py \
  model=realesrgan \
  evaluation.checkpoint_path=base \
  evaluation.stage=test \
  experiment.name=eval_realesrgan_base
```

Prediction mode saves batch tensors to `outputs/<experiment>/predictions` by
default:

```bash
python scripts/run_evaluation.py \
  evaluation.stage=predict \
  evaluation.checkpoint_path=outputs/docs_swinir_lora/checkpoints/last.ckpt
```

## Reports

Each experiment writes `metrics.csv` under its output directory. Aggregate all
available metrics into one summary:

```bash
python scripts/generate_report.py \
  +report.search_dir=outputs \
  +report.output_path=outputs/summary.csv
```

## Configuration Notes

The root config is `configs/config.yaml`. Main configurable groups:

- `data`: `base`, `docs`, `mri`, `retro`, `domain_adaptation`
- `model`: `swinir`, `realesrgan`
- `strategy`: `base`, `lora`, `partial`
- `trainer`, `optimizer`, `scheduler`, `loss`, `metrics`, `callbacks`

Reproducibility is handled through `lightning.seed_everything(seed,
workers=True)`. For strict deterministic runs, set the corresponding Lightning
trainer flags through Hydra.
