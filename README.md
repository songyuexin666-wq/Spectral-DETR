# Spectral-DETR

Spectral-DETR is a detector-internal reliability framework for object detection in degraded underground mine scenes. It is built on RF-DETR with a DINOv2-Small backbone and adds three reliability-oriented components:

- **DAFD** — Degradation-Aware Frequency Decomposition for feature reliability.
- **DQCD** — Degradation-adaptive Query Contrastive Denoising for query reliability.
- **SCU + LUE** — Salience-Calibrated Uncertainty with Learned Uncertainty Estimation for localization reliability.

The project accompanies the revised manuscript submitted to *Journal of Imaging*. The repository is intended to make the method implementation, dataset conversion utilities, evaluation scripts, and final configuration files publicly inspectable.

## Method Overview

Spectral-DETR addresses three failure points common in low-illumination, dusty, blurry, and cluttered mine imagery:

1. **Feature corruption**: DAFD applies learnable frequency-band decomposition, per-band FiLM conditioning, reliability gating, IFFT reconstruction, and residual spatial fusion before the multi-scale projection stage.
2. **Query entanglement**: DQCD applies a supervised contrastive objective over decoder query embeddings. Its InfoNCE temperature is modulated by DAFD gate statistics. DQCD is used only during training and adds no inference cost.
3. **Localization under-optimization**: LUE predicts coordinate-level log-variance, while SCU calibrates uncertainty using geometric salience. The calibrated uncertainty precision-weights the L1 localization term for small/medium objects while retaining standard GIoU supervision.

The core contribution is the cross-stage reliability pathway linking frequency processing, query regularization, and uncertainty-guided localization. Individual operators such as FFT filtering, contrastive learning, and uncertainty regression are established techniques; Spectral-DETR adapts and couples them inside a DETR-style detector for degraded underground detection.

## Repository Structure

```text
.
├── rfdetr/                         # RF-DETR-based model code with Spectral-DETR modules
│   ├── main.py                     # Training/evaluation entry points and argument population
│   ├── config.py                   # Model and module configuration definitions
│   ├── engine.py                   # Training/evaluation loop
│   ├── models/
│   │   └── backbone/projector.py   # DAFD and multi-scale projection integration
│   └── util/                       # Metrics, diagnostics, degradation utilities, checkpoints
├── configs/                        # Baseline and ablation YAML configurations
├── tools/                          # Table generation, ScienceDB evaluation, figure export helpers
├── scripts/                        # Analysis and visualization helpers
├── train_mine.py                   # YAML-driven training script for mine datasets
├── eval_test.py                    # Test-set evaluation from a trained checkpoint
├── benchmark_fps.py                # Batch-1 pure-forward FPS benchmark
├── convert_to_coco.py              # Mine-Objects conversion utility
├── convert_exdark_to_coco.py       # ExDark conversion utility
└── data.py                         # Dataset metadata and split helpers
```

## Installation

Create a Python environment and install the project in editable mode:

```bash
git clone https://github.com/songyuexin666-wq/Spectral-DETR.git
cd Spectral-DETR

conda create -n spectral-detr python=3.10 -y
conda activate spectral-detr

pip install -e .
```

Install PyTorch according to your CUDA version from the official PyTorch instructions. The experiments reported in the revised manuscript used a single RTX 3090 with 24 GB memory.

## Data

The manuscript uses three dataset-specific evaluation settings.

| Dataset | Role in manuscript | Public source |
|---|---|---|
| Mine-Objects | Self-built mine-domain benchmark with a fixed sequence-level 8:1:1 train/validation/test split | https://github.com/songyuexin666-wq/mine-datasets |
| ScienceDB V1 Coal Mine Underground Drilling Site Object Detection Dataset | Additional mine-domain benchmark converted to COCO format with a deterministic 80:20 train/validation partition | https://doi.org/10.57760/sciencedb.j00001.01020 |
| ExDark | Public low-light benchmark for external degraded-scene evaluation | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset |

Expected COCO-style layout:

```text
dataset_root/
├── train/
├── val/ or valid/
├── test/                 # required for held-out Mine-Objects test evaluation
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
```

Roboflow-style `_annotations.coco.json` layouts are also supported by `train_mine.py`.

## Main Configuration

The final Spectral-DETR configuration is:

```text
configs/lue_fafd_qcd.yaml
```

Despite the historical filename, the configuration corresponds to the revised terminology:

- `use_dafd`: enables Degradation-Aware Frequency Decomposition.
- `use_dqcd`: enables Degradation-adaptive Query Contrastive Denoising.
- `use_lue`: enables Learned Uncertainty Estimation.
- `use_scu`: enables Salience-Calibrated Uncertainty.

Key manuscript settings include:

- DINOv2-Small backbone.
- Input resolution: `560 × 560`.
- Three DAFD frequency bands.
- DAFD residual coefficient: `alpha = 0.15`.
- DQCD base temperature: `tau_base = 0.15`.
- DQCD memory/negative count: `K = 128`.
- DQCD starts after epoch 8 with a five-epoch warmup.
- LUE warmup: 15 epochs.
- SCU calibration coefficients: `c1 = -1.5`, `c0 = -4.2`.

## Training

Train a baseline or Spectral-DETR model with a YAML configuration:

```bash
python train_mine.py \
  --config configs/lue_fafd_qcd.yaml \
  --device cuda \
  --seed 42
```

For controlled ablations, use the corresponding files in `configs/`, for example:

```bash
python train_mine.py --config configs/baseline.yaml --device cuda
python train_mine.py --config configs/ablation_softnms_dafd.yaml --device cuda
python train_mine.py --config configs/ablation_softnms_dafd_dqcd.yaml --device cuda
python train_mine.py --config configs/ablation_softnms_dafd_dqcd_lue_scu.yaml --device cuda
```

The training script saves the effective configuration into the output directory for reproducibility.

## Evaluation

Evaluate a trained checkpoint on a held-out test split:

```bash
python eval_test.py \
  --checkpoint outputs/full/checkpoint_best_total.pth \
  --dataset_file coco \
  --coco_path /path/to/dataset_root \
  --device cuda \
  --output_dir outputs/full_eval
```

Measure pure model-forward FPS:

```bash
python benchmark_fps.py \
  --checkpoint outputs/full/checkpoint_best_total.pth \
  --device cuda \
  --warmup 20 \
  --iters 100
```

Evaluate a dedicated five-class ScienceDB checkpoint:

```bash
python tools/eval_sciencedb.py \
  --checkpoint outputs/sciencedb/checkpoint_best_total.pth \
  --coco-path /path/to/sciencedb_coco \
  --split val \
  --device cuda \
  --output outputs/sciencedb/sciencedb_results.json
```

## Reproducing Manuscript Tables and Figures

The revised manuscript reports dataset-specific results and controlled RF-DETR ablations. Table-generation and diagnostic helpers are provided in `tools/` and `scripts/`, including:

- `tools/make_tables.py` for manuscript table assembly.
- `tools/export_figure5.py` for matched qualitative examples.
- `tools/eval_sciencedb.py` for ScienceDB evaluation.
- `benchmark_fps.py` for the FPS protocol.

Some plotting scripts in `scripts/` are retained for historical analysis and may require paths to local experiment logs. The manuscript tables should be reproduced from the final trained checkpoints and generated result JSON files, not from hard-coded legacy plots.

## Reproducibility Notes

- DQCD and SCU are training-only objectives and do not add inference operations.
- LUE can be retained for diagnostic uncertainty output; final detections use the standard class scores and box predictions.
- DAFD is the main source of inference overhead because it uses FFT/IFFT operations and additional spatial fusion.
- ScienceDB is used as a dataset-specific mine-domain benchmark after COCO conversion; the repository DOI is not a source-code repository.
- Pretrained weights and trained checkpoints are not bundled in this repository unless explicitly released separately.

## Citation

If you use this code, please cite the associated manuscript after publication. Before formal publication, cite the repository as:

```bibtex
@misc{spectral_detr_code,
  title        = {Spectral-DETR: Detector-Internal Reliability Propagation for Degraded Underground Object Detection},
  author       = {Yuexin Song},
  year         = {2026},
  howpublished = {\url{https://github.com/songyuexin666-wq/Spectral-DETR}}
}
```

## Acknowledgements

This implementation builds on RF-DETR, DINOv2, LW-DETR, and Deformable DETR. We thank the authors and maintainers of these projects for their open-source contributions.

## License

This repository follows the license terms inherited from the RF-DETR codebase and included license file. Dataset licenses follow their respective public records.
