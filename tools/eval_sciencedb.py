#!/usr/bin/env python3
"""
Evaluate a Spectral-DETR checkpoint on ScienceDB at multiple IoU thresholds.

Reports AP@0.5, AP@0.75, and AP@0.5:0.95 to help determine whether the
large AP@0.5→AP@0.5:0.95 gap is primarily a localization issue.

Usage:
  python tools/eval_sciencedb.py \
    --checkpoint outputs/full/checkpoint_best_total.pth \
    --coco-path /path/to/sciencedb \
    --output sciencedb_results.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

import rfdetr.util.misc as utils
from eval_test import build_args_from_checkpoint
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.checkpoint import filter_state_dict_by_shape


EXPECTED_SCIENCEDB_CLASSES = [
    "chuck",
    "coal miner",
    "drill pipe",
    "gripper",
    "mine safety helmet",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_path(coco_path, split):
    candidates = [
        coco_path / "annotations" / f"instances_{split}2017.json",
        coco_path / "annotations" / f"instances_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No COCO annotation file found for split {split!r}: {candidates}"
    )


def extract_coco_metrics(coco_eval_bbox):
    """Extract AP@0.5, AP@0.75, AP@0.5:0.95 from COCO eval result list.

    COCO eval bbox is a list of 12 values:
        [0] AP@IoU=0.50:0.95, area=all, maxDets=100
        [1] AP@IoU=0.50,      area=all, maxDets=100
        [2] AP@IoU=0.75,      area=all, maxDets=100
        ...
    """
    if coco_eval_bbox is None:
        return {"ap50_95": None, "ap50": None, "ap75": None}
    if isinstance(coco_eval_bbox, list):
        return {
            "ap50_95": coco_eval_bbox[0] if len(coco_eval_bbox) > 0 else None,
            "ap50": coco_eval_bbox[1] if len(coco_eval_bbox) > 1 else None,
            "ap75": coco_eval_bbox[2] if len(coco_eval_bbox) > 2 else None,
        }
    # dict form
    return {
        "ap50_95": coco_eval_bbox.get("AP"),
        "ap50": coco_eval_bbox.get("AP50"),
        "ap75": coco_eval_bbox.get("AP75"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to checkpoint_best_total.pth")
    parser.add_argument("--coco-path", required=True, type=Path,
                        help="Path to ScienceDB COCO directory")
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Converted dataset split to evaluate (default: val)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON path")
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = build_args_from_checkpoint(
        checkpoint.get("args"),
        str(args.checkpoint),
        "coco",
        str(args.coco_path),
        str(args.coco_path),
        args.device,
        str(args.output.parent),
    )
    ckpt_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    checkpoint_object_classes = ckpt_num_classes - 1
    if checkpoint_object_classes != len(EXPECTED_SCIENCEDB_CLASSES):
        raise ValueError(
            "ScienceDB evaluation requires a dedicated five-class checkpoint; "
            f"the supplied checkpoint represents {checkpoint_object_classes} object classes."
        )
    ckpt_args.num_classes = checkpoint_object_classes
    ckpt_args.batch_size = args.batch_size
    ckpt_args.num_workers = args.num_workers
    ckpt_args.diagnostics_sample_records = False
    ckpt_args.diagnostics = False
    ckpt_args.fp16_eval = False
    # Override split for evaluation
    ckpt_args.split = args.split

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.coco_path} ({args.split} split)")
    print(f"Classes    : {ckpt_args.num_classes}")
    print(f"Device     : {args.device}")

    device = torch.device(args.device)
    model = build_model(ckpt_args).to(device)
    state, dropped = filter_state_dict_by_shape(
        model.state_dict(), checkpoint["model"]
    )
    if dropped:
        print(f"Warning: {len(dropped)} incompatible tensors skipped (showing first 20):")
        for name, ckpt_shape, model_shape in dropped[:20]:
            print(f"  - {name}: ckpt{ckpt_shape} != model{model_shape}")
    model.load_state_dict(state, strict=False)
    model.eval()

    criterion, postprocess = build_criterion_and_postprocessors(ckpt_args)

    # Build dataset — use split from args
    dataset = build_dataset(args.split, ckpt_args, resolution=ckpt_args.resolution)
    base_coco = get_coco_api_from_dataset(dataset)
    cat_ids = sorted(base_coco.getCatIds())
    categories = base_coco.loadCats(cat_ids)
    dataset_class_names = [cat["name"].strip() for cat in categories]
    if sorted(name.casefold() for name in dataset_class_names) != sorted(
        name.casefold() for name in EXPECTED_SCIENCEDB_CLASSES
    ):
        raise ValueError(
            "ScienceDB category semantics do not match the official five classes. "
            f"Found: {dataset_class_names}"
        )
    checkpoint_class_names = getattr(checkpoint.get("args"), "class_names", None)
    if checkpoint_class_names and [name.casefold() for name in checkpoint_class_names] != [
        name.casefold() for name in dataset_class_names
    ]:
        raise ValueError(
            "Checkpoint class order does not match the ScienceDB annotation order: "
            f"checkpoint={checkpoint_class_names}, dataset={dataset_class_names}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    print(f"Samples    : {len(dataset)}")
    print(f"Categories : {list(zip(cat_ids, dataset_class_names))}")

    # Evaluate
    evaluation, _coco_evaluator = evaluate(
        model, criterion, postprocess, loader, base_coco, device, args=ckpt_args
    )

    coco_eval = evaluation.get("coco_eval_bbox")
    metrics = extract_coco_metrics(coco_eval)
    metrics["num_samples"] = len(dataset)
    metrics["num_classes"] = ckpt_args.num_classes
    metrics["split"] = args.split
    metrics["checkpoint"] = str(args.checkpoint.resolve())
    ann_path = annotation_path(args.coco_path, args.split)
    metrics["annotation_file"] = str(ann_path.resolve())
    metrics["annotation_sha256"] = file_sha256(ann_path)
    metrics["category_ids"] = cat_ids
    metrics["class_names"] = dataset_class_names
    manifest_path = args.coco_path / "split_manifest.json"
    if manifest_path.exists():
        metrics["split_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Compute localization gap
    if metrics["ap50"] is not None and metrics["ap50_95"] is not None:
        metrics["ap50_to_ap50_95_gap"] = float(metrics["ap50"] - metrics["ap50_95"])
    if metrics["ap50"] is not None and metrics["ap75"] is not None:
        metrics["ap50_to_ap75_gap"] = float(metrics["ap50"] - metrics["ap75"])

    # Per-class AP for localization analysis
    per_class = {}
    if _coco_evaluator is not None and "bbox" in _coco_evaluator.coco_eval:
        eval_obj = _coco_evaluator.coco_eval["bbox"]
        if hasattr(eval_obj, "eval") and eval_obj.eval:
            precisions = eval_obj.eval.get("precision")
            if precisions is not None and precisions.ndim == 5:
                # precision[T, R, K, A, M]
                # T=IoU thresholds, R=recall thresholds, K=category, A=area, M=maxDets
                try:
                    class_names = base_coco.loadCats(eval_obj.params.catIds)
                except Exception:
                    class_names = []
                for k_idx, cat in enumerate(class_names):
                    cat_name = cat["name"]
                    values = precisions[:, :, k_idx, 0, -1].ravel()
                    ap_cat = float(
                        sum(a for a in values if a >= 0)
                        / max(1, (values >= 0).sum())
                    )
                    values50 = precisions[0, :, k_idx, 0, -1]
                    ap50_cat = float(
                        sum(a for a in values50 if a >= 0)
                        / max(1, (values50 >= 0).sum())
                    )
                    per_class[cat_name] = {
                        "ap50_95": round(ap_cat, 4),
                        "ap50": round(ap50_cat, 4),
                    }
    metrics["per_class"] = per_class

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"ScienceDB Evaluation ({args.split})")
    print(f"  Samples  : {metrics['num_samples']}")
    print(f"  Classes  : {metrics['num_classes']}")
    print(f"  AP@0.5        : {metrics['ap50']}")
    print(f"  AP@0.75       : {metrics['ap75']}")
    print(f"  AP@0.5:0.95   : {metrics['ap50_95']}")
    print(f"  AP@0.5→0.5:0.95 gap: {metrics.get('ap50_to_ap50_95_gap')}")
    print(f"  AP@0.5→0.75   gap: {metrics.get('ap50_to_ap75_gap')}")
    print(f"\nResults → {args.output}")


if __name__ == "__main__":
    main()
