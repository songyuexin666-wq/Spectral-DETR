#!/usr/bin/env python3
"""Evaluate DAFD reliability signals under controlled image degradation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torchvision.transforms.functional import gaussian_blur

import rfdetr.util.misc as utils
from eval_test import build_args_from_checkpoint
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.checkpoint import filter_state_dict_by_shape


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEGRADATION_CODES = {"clean": 0, "low_light": 1, "noise": 2, "blur": 3, "contrast": 4}


class ControlledDegradationDataset(Dataset):
    def __init__(self, dataset, kind, severity):
        self.dataset = dataset
        self.kind = kind
        self.severity = float(severity)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self._degrade(image, index)
        target = dict(target)
        target["controlled_degradation_type"] = torch.tensor(
            DEGRADATION_CODES[self.kind], dtype=torch.int64
        )
        target["controlled_degradation_severity"] = torch.tensor(
            self.severity, dtype=torch.float32
        )
        return image, target

    def _degrade(self, image, index):
        mean = IMAGENET_MEAN.to(image)
        std = IMAGENET_STD.to(image)
        restored = (image * std + mean).clamp(0.0, 1.0)

        if self.kind == "clean":
            degraded = restored
        elif self.kind == "low_light":
            degraded = restored * (1.0 - self.severity)
        elif self.kind == "contrast":
            gray = restored.mean(dim=(1, 2), keepdim=True)
            degraded = gray + (restored - gray) * (1.0 - self.severity)
        elif self.kind == "noise":
            generator = torch.Generator(device=restored.device)
            generator.manual_seed(20260730 + index)
            noise = torch.randn(
                restored.shape,
                generator=generator,
                device=restored.device,
                dtype=restored.dtype,
            )
            degraded = restored + noise * self.severity
        elif self.kind == "blur":
            kernel = int(round(3 + 8 * self.severity))
            kernel += 1 - kernel % 2
            sigma = 0.5 + 2.5 * self.severity
            degraded = gaussian_blur(restored, [kernel, kernel], [sigma, sigma])
        else:
            raise ValueError(f"Unsupported degradation kind: {self.kind}")
        return (degraded.clamp(0.0, 1.0) - mean) / std


def _correlation_with_ci(x, y, seed=42, samples=1000):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(x.size), "pearson": None, "spearman": None, "pearson_ci95": None}

    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(scipy_stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        indexes = rng.integers(0, x.size, x.size)
        bx, by = x[indexes], y[indexes]
        if np.std(bx) > 0 and np.std(by) > 0:
            boot.append(float(np.corrcoef(bx, by)[0, 1]))
    ci = np.percentile(boot, [2.5, 97.5]).tolist() if boot else None
    return {
        "n": int(x.size),
        "pearson": pearson,
        "spearman": spearman,
        "pearson_ci95": ci,
    }


def _risk_coverage(records):
    pairs = [
        (record["mean_uncertainty"], record["localization_error"])
        for record in records
        if record.get("mean_uncertainty") is not None
    ]
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0])
    risks = {}
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, int(round(len(pairs) * coverage)))
        risks[str(coverage)] = float(np.mean([error for _, error in pairs[:count]]))
    return risks


def _settings():
    return [
        ("clean", 0.0),
        *(("low_light", level) for level in (0.25, 0.50, 0.75)),
        *(("noise", level) for level in (0.02, 0.05, 0.10)),
        *(("blur", level) for level in (0.25, 0.50, 0.75)),
        *(("contrast", level) for level in (0.25, 0.50, 0.75)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    cli = parser.parse_args()

    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    args = build_args_from_checkpoint(
        checkpoint.get("args"), cli.checkpoint, "coco", cli.coco_path,
        cli.coco_path, cli.device, str(Path(cli.output).parent),
    )
    args.num_classes = checkpoint["model"]["class_embed.bias"].shape[0] - 1
    args.batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    args.diagnostics_sample_records = True
    args.diagnostics = False
    args.fp16_eval = False

    device = torch.device(args.device)
    model = build_model(args).to(device)
    state, dropped = filter_state_dict_by_shape(model.state_dict(), checkpoint["model"])
    if dropped:
        raise RuntimeError(f"Checkpoint has {len(dropped)} incompatible tensors; refusing diagnostic evaluation")
    model.load_state_dict(state, strict=True)
    model.eval()
    criterion, postprocess = build_criterion_and_postprocessors(args)

    base_dataset = build_dataset("val", args, resolution=args.resolution)
    base_coco = get_coco_api_from_dataset(base_dataset)
    output = {"checkpoint": str(Path(cli.checkpoint).resolve()), "settings": []}
    all_records = []

    for kind, severity in _settings():
        dataset = ControlledDegradationDataset(base_dataset, kind, severity)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(dataset),
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )
        evaluation, _ = evaluate(
            model, criterion, postprocess, loader, base_coco, device, args=args
        )
        coco = evaluation.get("coco_eval_bbox") or [None] * 12
        records = evaluation.pop("diagnostics_sample_records", [])
        all_records.extend(records)
        output["settings"].append({
            "kind": kind,
            "severity": severity,
            "ap50_95": coco[0],
            "ap50": coco[1],
            "ap_s": coco[3],
            "gate_mean": evaluation.get("diag/dafd_gate_mean_unscaled"),
            "mean_localization_error": float(np.mean([
                item["localization_error"] for item in records
            ])),
            "records": records,
        })

    correlations = {}
    for kind in ("low_light", "noise", "blur", "contrast"):
        records = [
            item for item in all_records
            if item.get("controlled_degradation_type") == DEGRADATION_CODES[kind]
        ]
        severity = [item["controlled_degradation_severity"] for item in records]
        gate = [item["gate_mean"] for item in records]
        error = [item["localization_error"] for item in records]
        correlations[kind] = {
            "severity_vs_gate": _correlation_with_ci(severity, gate),
            "severity_vs_localization_error": _correlation_with_ci(severity, error),
            "gate_vs_localization_error": _correlation_with_ci(gate, error),
        }
    output["correlations"] = correlations
    output["risk_coverage"] = _risk_coverage(all_records)

    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved controlled-degradation evidence to {output_path}")


if __name__ == "__main__":
    main()
