#!/usr/bin/env python3
"""
Export Figure 5: matched qualitative comparison under fixed parameters.

Runs inference with baseline and full Spectral-DETR checkpoints on the same
images, using identical score threshold and category mapping, then produces
side-by-side panels suitable for the paper.

Requirements:
  - Baseline checkpoint (RF-DETR)
  - Full Spectral-DETR checkpoint
  - List of image paths (one per row in Figure 5)
  - Common score threshold

Usage:
  python tools/export_figure5.py \
    --baseline outputs/baseline/checkpoint_best_total.pth \
    --full outputs/full/checkpoint_best_total.pth \
    --images img1.jpg img2.jpg img3.jpg img4.jpg \
    --score-thresh 0.3 \
    --category-mapping mine_objects \
    --output-dir figures/fig5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from rfdetr.models import build_model
from rfdetr.util.checkpoint import filter_state_dict_by_shape
from rfdetr.util.coco_classes import COCO_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
    "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
    "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF",
    "#FF95C8", "#FF37C7",
]


def load_model_and_class_map(ckpt_path: Path, device: str, num_classes: int):
    """Load model and return (model, class_map)."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args")
    if ckpt_args is None:
        raise RuntimeError("checkpoint has no args")

    ckpt_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    ckpt_args.num_classes = ckpt_num_classes - 1
    ckpt_args.device = device
    if num_classes is not None:
        ckpt_args.num_classes = num_classes

    if not hasattr(ckpt_args, "batch_size") or ckpt_args.batch_size is None:
        ckpt_args.batch_size = 1

    model = build_model(ckpt_args).to(torch.device(device))
    state, dropped = filter_state_dict_by_shape(model.state_dict(), checkpoint["model"])
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, ckpt_args


def infer_image(model, args, image_path: Path, device: str, score_thresh: float):
    """Run inference on a single image and return list of detections."""
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (W, H)

    # Resize to model resolution
    img = F.resize(image, (args.resolution, args.resolution))
    img = F.to_tensor(img)
    img = F.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        # outputs: {'pred_logits': ..., 'pred_boxes': ...}

    # Post-process: convert boxes to [x1,y1,x2,y2] in original image coords
    pred_logits = outputs["pred_logits"][0]  # [num_queries, num_classes+1]
    pred_boxes = outputs["pred_boxes"][0]    # [num_queries, 4] in cxcywh normalized

    # Convert to scores
    scores = pred_logits.sigmoid()  # [num_queries, num_classes+1]

    detections = []
    for cls_idx in range(args.num_classes):
        cls_scores = scores[:, cls_idx]
        keep = cls_scores > score_thresh
        if not keep.any():
            continue
        idxs = keep.nonzero(as_tuple=True)[0]
        for i in idxs:
            cx, cy, w, h = pred_boxes[i].tolist()
            score = cls_scores[i].item()
            # Convert to pixel coords
            x1 = (cx - w / 2) * orig_size[0]
            y1 = (cy - h / 2) * orig_size[1]
            x2 = (cx + w / 2) * orig_size[0]
            y2 = (cy + h / 2) * orig_size[1]
            detections.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "class_idx": cls_idx,
                "score": round(score, 4),
            })

    return detections


def draw_detections(image, detections, class_names, img_path):
    """Overlay boxes on image and return annotated PIL Image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_idx = det["class_idx"]
        score = det["score"]
        color = _COLORS[cls_idx % len(_COLORS)]

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label
        cls_name = class_names.get(cls_idx, class_names.get(str(cls_idx), f"cls{cls_idx}"))
        label = f"{cls_name} {score:.2f}"
        # Draw a background box for the label
        bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="#FFFFFF", font=font)

    return annotated


def build_comparison_panel(input_image, baseline_annot, full_annot):
    """
    Create a single panel: input | baseline | Spectral-DETR.
    Returns a PIL Image.
    """
    w0, h0 = input_image.size
    w1, h1 = baseline_annot.size
    w2, h2 = full_annot.size

    # Use max height
    max_h = max(h0, h1, h2)
    gap = 20
    total_w = w0 + gap + w1 + gap + w2

    panel = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
    panel.paste(input_image, (0, 0))
    panel.paste(baseline_annot, (w0 + gap, 0))
    panel.paste(full_annot, (w0 + gap + w1 + gap, 0))

    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path,
                        help="Baseline (RF-DETR) checkpoint")
    parser.add_argument("--full", required=True, type=Path,
                        help="Full Spectral-DETR checkpoint")
    parser.add_argument("--images", nargs="+", required=True, type=Path,
                        help="Image paths to compare")
    parser.add_argument("--score-thresh", type=float, default=0.3,
                        help="Score threshold (default: 0.3)")
    parser.add_argument("--num-classes", type=int, default=14,
                        help="Number of classes (14 for Mine-Objects)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/fig5"),
                        help="Output directory")
    parser.add_argument("--class-names", type=Path, default=None,
                        help="JSON file mapping class_idx → name")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Build class name mapping
    if args.class_names and args.class_names.exists():
        class_names = json.loads(args.class_names.read_text())
        class_names = {int(k): v for k, v in class_names.items()}
    else:
        # Fall back to COCO_CLASSES if num_classes <= 90
        if args.num_classes <= 91:
            class_names = {i: COCO_CLASSES[i] for i in range(args.num_classes)}
        else:
            class_names = {i: f"cls{i}" for i in range(args.num_classes)}

    print(f"Class names: {len(class_names)}")
    print(f"Images: {len(args.images)}")

    # Load models
    print(f"Loading baseline: {args.baseline}")
    model_base, args_base = load_model_and_class_map(args.baseline, device, args.num_classes)
    print(f"Loading full: {args.full}")
    model_full, args_full = load_model_and_class_map(args.full, device, args.num_classes)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "baseline_checkpoint": str(args.baseline.resolve()),
        "full_checkpoint": str(args.full.resolve()),
        "score_threshold": args.score_thresh,
        "num_classes": args.num_classes,
        "class_names": class_names,
        "images": [],
    }

    for i, img_path in enumerate(args.images):
        if not img_path.exists():
            print(f"  [SKIP] {img_path} not found")
            continue

        print(f"[{i+1}/{len(args.images)}] {img_path.name}")

        input_image = Image.open(img_path).convert("RGB")

        dets_base = infer_image(model_base, args_base, img_path, device, args.score_thresh)
        dets_full = infer_image(model_full, args_full, img_path, device, args.score_thresh)

        base_annot = draw_detections(input_image.copy(), dets_base, class_names, img_path)
        full_annot = draw_detections(input_image.copy(), dets_full, class_names, img_path)

        panel = build_comparison_panel(input_image, base_annot, full_annot)
        out_path = args.output_dir / f"fig5_row{i+1:02d}_{img_path.stem}.png"
        panel.save(out_path)
        print(f"  → {out_path}")

        manifest["images"].append({
            "input": str(img_path.resolve()),
            "baseline_detections": dets_base,
            "full_detections": dets_full,
            "output_panel": str(out_path.resolve()),
        })

    manifest_path = args.output_dir / "fig5_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest → {manifest_path}")
    print(f"Panels → {args.output_dir}/")


if __name__ == "__main__":
    main()
