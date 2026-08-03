#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Spectral-DETR inference script
# GitHub: https://github.com/songyuexin666-wq/Spectral-DETR
# ------------------------------------------------------------------------
"""
Single-image and batch inference for trained Spectral-DETR / RF-DETR checkpoints.

Supports:
  - Single image, directory glob, or list-of-paths input
  - Automatic model reconstruction from checkpoint args (num_classes, LUE, DAFD, DDQCD, SCU)
  - LUE uncertainty heatmap overlay
  - JSON-serialisable detection output
  - Batch GPU inference for throughput
  - COCO-compatible class name mapping

Examples:
  # Single image
  python infer.py --checkpoint outputs/spectral_detr_v5/checkpoint_best_total.pth \
                   --image datasets/coco/val2017/000000000139.jpg

  # Directory (all .jpg/.png recursively)
  python infer.py --checkpoint outputs/best.pth --image_dir datasets/coco/val2017

  # Lower threshold, custom output dir, batch 4
  python infer.py --checkpoint outputs/best.pth --image_dir images/ \
                   --score_thresh 0.3 --output_dir results/ --batch_size 4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from rfdetr.models import build_model, PostProcess
from rfdetr.util.checkpoint import filter_state_dict_by_shape
from rfdetr.util.coco_classes import COCO_CLASSES

# ImageNet normalisation (identical to training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Colour palette for per-class bounding boxes (24-colour cycle)
# ---------------------------------------------------------------------------
_COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
    "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
    "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF",
    "#FF95C8", "#FF37C7", "#FF0000", "#FFA500", "#FFFF00", "#00FF00",
]


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _get_class_color(cls_id: int) -> Tuple[int, int, int]:
    return _hex_to_rgb(_COLORS[cls_id % len(_COLORS)])


# ===========================================================================
# 1. Model loader
# ===========================================================================

def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "cuda",
) -> Tuple[torch.nn.Module, PostProcess, argparse.Namespace, Optional[List[str]]]:
    """Reconstruct a Spectral-DETR model from a training checkpoint.

    The checkpoint is expected to contain:
        - ``model`` : state_dict
        - ``args``  : argparse.Namespace with all hyper-parameters

    Returns:
        model       – eval-mode nn.Module on *device*
        postprocess – PostProcess(num_select)
        args        – reconstructed args (with corrected num_classes)
        class_names – list[str] or None
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "args" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'args'.  "
            "Re-save the checkpoint with `args` included, or provide a --config yaml."
        )

    args = checkpoint["args"]
    ckpt_state = checkpoint["model"]

    # ---- deduce num_classes from checkpoint head shape --------------------
    if "class_embed.bias" in ckpt_state:
        ckpt_num_classes = ckpt_state["class_embed.bias"].shape[0]
        args.num_classes = ckpt_num_classes - 1  # LW-DETR convention: D_out = C+1
        print(f"[INFO] Inferred num_classes = {args.num_classes} from checkpoint head (dim={ckpt_num_classes})")
    else:
        print("[WARN] class_embed.bias not found in checkpoint; using args.num_classes as-is")

    args.device = device

    # Ensure innovation flags exist on args (older checkpoints may lack them)
    for flag in ("use_lue", "use_dafd", "use_dqcd", "use_scu",
                 "lue_warmup_epochs", "lue_giou_weighting",
                 "dafd_sparsity_weight", "dafd_alpha", "dafd_n_bands",
                 "dqcd_temperature", "dqcd_weight", "dqcd_hard_negatives_k", "dqcd_start_epoch",
                 "scu_salience_weight", "scu_calib_slope", "scu_calib_center",
                 "ia_bce_loss", "use_varifocal_loss", "use_position_supervised_loss",
                 "bbox_reparam", "group_detr", "num_select"):
        if not hasattr(args, flag):
            setattr(args, flag, False if flag.startswith("use_") or flag in ("ia_bce_loss", "bbox_reparam") else
                    (0.0 if "weight" in flag or "coef" in flag else
                     (300 if flag == "num_select" else
                      (13 if flag == "group_detr" else 0))))

    print(f"[INFO] LUE={getattr(args, 'use_lue', False)}, "
          f"DAFD={getattr(args, 'use_dafd', False)}, "
          f"DDQCD={getattr(args, 'use_dqcd', False)}, "
          f"SCU={getattr(args, 'use_scu', False)}")

    # ---- build model ------------------------------------------------------
    print(f"[INFO] Building model with num_classes = {args.num_classes}")
    model = build_model(args)
    model.to(torch.device(device))

    # ---- load weights -----------------------------------------------------
    model_state = model.state_dict()
    ckpt_filtered, dropped = filter_state_dict_by_shape(model_state, ckpt_state)
    if dropped:
        print(f"[WARN] {len(dropped)} params skipped due to shape mismatch (showing ≤20):")
        for name, ckpt_shape, model_shape in dropped[:20]:
            print(f"       {name}: ckpt{ckpt_shape} != model{model_shape}")
    else:
        print("[INFO] All checkpoint params match model shapes.")

    model.load_state_dict(ckpt_filtered, strict=False)
    model.eval()

    # ---- post-processor ---------------------------------------------------
    num_select = getattr(args, "num_select", 300)
    postprocess = PostProcess(num_select=num_select)

    # ---- class names ------------------------------------------------------
    class_names = getattr(args, "class_names", None)

    # If not available, build from COCO_CLASSES up to num_classes
    if class_names is None:
        class_names = _build_class_names_from_coco(args.num_classes)
        print(f"[INFO] Built class_names from COCO mapping ({len(class_names)} classes)")

    return model, postprocess, args, class_names


def _build_class_names_from_coco(num_classes: int) -> List[str]:
    """Build a contiguous class-name list from the COCO_CLASSES dict."""
    names = []
    for i in range(num_classes):
        cid = i + 1  # COCO classes are 1-indexed
        if cid in COCO_CLASSES:
            names.append(COCO_CLASSES[cid])
        else:
            # Handle gaps (COCO skips some IDs)
            names.append(f"class_{cid}")
    return names


# ===========================================================================
# 2. Image dataset (directory / list / single)
# ===========================================================================

class ImageInferenceDataset(Dataset):
    """Lightweight dataset that loads images and keeps metadata."""

    def __init__(self, image_paths: List[Path]):
        self.paths = [Path(p) for p in image_paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


def collate_images(
    batch: List[Path],
    resolution: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Image.Image], List[Tuple[int, int]], List[Path]]:
    """Load a batch of images, preprocess, and stack.

    Returns:
        tensor      – (B, 3, H, W) normalised float tensor on *device*
        pil_images  – list of PIL originals (for visualisation)
        orig_sizes  – list of (h, w)
        paths       – list of Path
    """
    pil_images: List[Image.Image] = []
    tensors: List[torch.Tensor] = []
    orig_sizes: List[Tuple[int, int]] = []
    good_paths: List[Path] = []

    for p in batch:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Skipping {p}: {exc}", file=sys.stderr)
            continue

        pil_images.append(img)
        good_paths.append(p)
        w, h = img.size
        orig_sizes.append((h, w))

        t = F.to_tensor(img)  # CxHxW, [0,1]
        t = F.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
        t = F.resize(t, (resolution, resolution))
        tensors.append(t)

    if not tensors:
        return (
            torch.empty(0, 3, resolution, resolution, device=device),
            [], [], [],
        )

    batch_tensor = torch.stack(tensors).to(device)
    return batch_tensor, pil_images, orig_sizes, good_paths


# ===========================================================================
# 3. Visualisation
# ===========================================================================

def _load_font(size: int = 14):
    for name in ("arial.ttf", "DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    boxes: torch.Tensor,       # (K, 4) xyxy absolute
    scores: torch.Tensor,      # (K,)
    labels: torch.Tensor,      # (K,) int
    class_names: Optional[List[str]] = None,
    score_thresh: float = 0.3,
    max_dets: int = 100,
    line_width: int = 2,
) -> Image.Image:
    """Draw bounding boxes, class labels, and confidence scores on a PIL image.

    Returns a new PIL image (the original is not modified).
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    label_font = _load_font(11)

    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Filter & sort
    keep = scores_np >= score_thresh
    boxes_np = boxes_np[keep]
    scores_np = scores_np[keep]
    labels_np = labels_np[keep]

    if len(scores_np) > max_dets:
        order = np.argsort(scores_np)[::-1][:max_dets]
        boxes_np = boxes_np[order]
        scores_np = scores_np[order]
        labels_np = labels_np[order]

    for box, score, label in zip(boxes_np, scores_np, labels_np):
        x1, y1, x2, y2 = map(float, box)
        cls_id = int(label)

        if class_names is not None and 0 <= cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = f"cls_{cls_id}"

        color = _get_class_color(cls_id)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        tag = f"{cls_name} {score:.2f}"
        try:
            bbox = draw.textbbox((0, 0), tag, font=label_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(tag) * 7, 14

        tx = max(0, x1)
        ty = max(0, y1 - th - 2)
        draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=color)
        draw.text((tx + 2, ty + 2), tag, fill="white", font=label_font)

    return img


def draw_uncertainty_heatmap(
    image: Image.Image,
    boxes: torch.Tensor,
    uncertainties: torch.Tensor,
    scores: torch.Tensor,
    score_thresh: float = 0.3,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a per-box LUE uncertainty heatmap onto the image.

    High uncertainty → bright (magenta-hot); low → dark.
    """
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("magma")
    except Exception:
        cmap = None

    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    unc_np = uncertainties.cpu().numpy()

    keep = scores_np >= score_thresh
    boxes_np = boxes_np[keep]
    unc_np = unc_np[keep]

    if len(unc_np) == 0:
        return image.copy()

    h_img, w_img = image.height, image.width
    heat = np.zeros((h_img, w_img), dtype=np.float32)
    count = np.zeros((h_img, w_img), dtype=np.float32)

    u_min, u_max = float(unc_np.min()), float(unc_np.max())
    u_range = u_max - u_min
    if u_range < 1e-6:
        u_range = 1.0
    u_norm = (unc_np - u_min) / u_range

    for uv, box in zip(u_norm, boxes_np):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        heat[y1:y2, x1:x2] += float(uv)
        count[y1:y2, x1:x2] += 1.0

    mask = count > 0
    heat_norm = np.zeros_like(heat)
    if mask.any():
        heat[mask] /= count[mask]
        h_min, h_max = float(heat[mask].min()), float(heat[mask].max())
        if h_max - h_min > 1e-6:
            heat_norm[mask] = (heat[mask] - h_min) / (h_max - h_min)
        else:
            heat_norm[mask] = 0.5

    # Where no detection, set alpha to 0
    alpha_ch = np.where(mask, alpha, 0.0).astype(np.float32)

    if cmap is not None:
        color = cmap(heat_norm)[..., :3]  # (H,W,3)
        color_img = (color * 255).astype(np.uint8)
    else:
        color_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        color_img[..., 0] = (heat_norm * 255).astype(np.uint8)

    heat_pil = Image.fromarray(color_img)
    orig_np = np.array(image.convert("RGB")).astype(np.float32)
    heat_np = np.array(heat_pil).astype(np.float32)
    alpha_3 = np.stack([alpha_ch] * 3, axis=-1)
    blended = heat_np * alpha_3 + orig_np * (1 - alpha_3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ===========================================================================
# 4. Core inference runner
# ===========================================================================

class InferenceRunner:
    """Encapsulates a loaded model and exposes single / batch inference."""

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: str = "cuda",
    ):
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else
            "mps" if (device == "mps" and torch.backends.mps.is_available()) else
            "cpu"
        )
        print(f"[INFO] Device: {self.device}")

        self.model, self.postprocess, self.args, self.class_names = \
            load_model_from_checkpoint(checkpoint, str(self.device))
        self.resolution: int = self.args.resolution
        self.use_lue: bool = getattr(self.args, "use_lue", False)
        self.has_log_vars: bool = (
            self.use_lue and getattr(self.model, "bbox_log_var_embed", None) is not None
        )
        print(f"[INFO] Resolution: {self.resolution}, LUE outputs: {self.has_log_vars}")

    @torch.inference_mode()
    def predict_single(self, image_path: Union[str, Path]) -> dict:
        """Run inference on one image file.

        Returns a dict with:
            boxes, scores, labels, uncertainties (optional),
            orig_size (h,w), path
        """
        path = Path(image_path)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        orig_size = (h, w)

        t = F.to_tensor(img)
        t = F.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
        t = F.resize(t, (self.resolution, self.resolution))
        t = t.unsqueeze(0).to(self.device)

        outputs = self.model(t)

        # Normalise output format
        if isinstance(outputs, tuple):
            out_dict = {"pred_boxes": outputs[0], "pred_logits": outputs[1]}
            if len(outputs) >= 3:
                out_dict["pred_masks"] = outputs[2]
        else:
            out_dict = outputs

        target_sizes = torch.tensor([orig_size], device=self.device)
        results = self.postprocess(out_dict, target_sizes=target_sizes)[0]

        return {
            "boxes": results["boxes"].cpu(),
            "scores": results["scores"].cpu(),
            "labels": results["labels"].cpu(),
            "uncertainties": results.get("uncertainty"),
            "orig_size": orig_size,
            "path": str(path),
        }

    @torch.inference_mode()
    def predict_batch(self, image_paths: List[Path], batch_size: int = 4) -> List[dict]:
        """Run batched inference over a list of image paths."""
        dataset = ImageInferenceDataset(image_paths)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, batch_size),
            collate_fn=lambda batch: collate_images(batch, self.resolution, self.device),
        )

        all_results: List[dict] = []
        for batch_tensor, pil_images, orig_sizes, paths in loader:
            if batch_tensor.numel() == 0:
                continue
            outputs = self.model(batch_tensor)
            if isinstance(outputs, tuple):
                out_dict = {"pred_boxes": outputs[0], "pred_logits": outputs[1]}
            else:
                out_dict = outputs

            target_sizes = torch.tensor(orig_sizes, device=self.device)
            results = self.postprocess(out_dict, target_sizes=target_sizes)

            for res, _img, sz, p in zip(results, pil_images, orig_sizes, paths):
                u = res.get("uncertainty")
                all_results.append({
                    "boxes": res["boxes"].cpu(),
                    "scores": res["scores"].cpu(),
                    "labels": res["labels"].cpu(),
                    "uncertainties": u.cpu() if u is not None else None,
                    "orig_size": sz,
                    "path": str(p),
                })

        return all_results


# ===========================================================================
# 5. Utility: gather image paths
# ===========================================================================

def gather_image_paths(
    image: Optional[str] = None,
    image_dir: Optional[str] = None,
    image_list: Optional[str] = None,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> List[Path]:
    """Collect image paths from --image, --image_dir, or --image_list."""
    paths: List[Path] = []

    if image:
        p = Path(image)
        if p.is_file():
            paths.append(p)
        else:
            print(f"[WARN] --image {image} is not a file, skipping")

    if image_dir:
        root = Path(image_dir)
        if root.is_dir():
            for ext in extensions:
                paths.extend(root.rglob(f"*{ext}"))
                paths.extend(root.rglob(f"*{ext.upper()}"))
        else:
            print(f"[WARN] --image_dir {image_dir} is not a directory, skipping")

    if image_list:
        list_path = Path(image_list)
        if list_path.is_file():
            with open(list_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        paths.append(Path(line))

    # Deduplicate, sort
    seen = set()
    unique = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    unique.sort()
    return unique


# ===========================================================================
# 6. JSON serialisation helpers
# ===========================================================================

def results_to_json(results: List[dict], class_names: Optional[List[str]] = None) -> List[dict]:
    """Convert inference results to JSON-serialisable dicts."""
    out = []
    for r in results:
        entry: dict = {
            "image_path": r["path"],
            "orig_size": list(r["orig_size"]),
            "detections": [],
        }
        boxes = r["boxes"]
        scores = r["scores"]
        labels = r["labels"]
        uncertainties = r.get("uncertainties")

        for i in range(len(scores)):
            det = {
                "bbox_xyxy": boxes[i].tolist(),
                "score": float(scores[i]),
                "class_id": int(labels[i]),
            }
            if class_names is not None and 0 <= int(labels[i]) < len(class_names):
                det["class_name"] = class_names[int(labels[i])]
            if uncertainties is not None and i < len(uncertainties):
                det["uncertainty"] = float(uncertainties[i])
            entry["detections"].append(det)
        out.append(entry)
    return out


# ===========================================================================
# 7. CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Spectral-DETR inference (single image / directory / batch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- required ---------------------------------------------------------
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint (contains model + args)")

    # ---- image sources (at least one required) ----------------------------
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str,
                       help="Single image path")
    group.add_argument("--image_dir", type=str,
                       help="Directory to scan for images (recursive)")
    group.add_argument("--image_list", type=str,
                       help="Text file with one image path per line")

    # ---- inference --------------------------------------------------------
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"])
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for GPU inference (default: 1)")
    parser.add_argument("--score_thresh", type=float, default=0.3,
                        help="Confidence threshold for visualisation / JSON (default: 0.3)")
    parser.add_argument("--max_dets", type=int, default=100,
                        help="Max detections to draw per image (default: 100)")

    # ---- output -----------------------------------------------------------
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for visualisations + JSON "
                             "(default: <checkpoint_dir>/inference_results/)")
    parser.add_argument("--save_json", action="store_true", default=True,
                        help="Save detection results as JSON (default: True)")
    parser.add_argument("--no_json", action="store_false", dest="save_json",
                        help="Skip JSON output")
    parser.add_argument("--save_vis", action="store_true", default=True,
                        help="Save visualised images (default: True)")
    parser.add_argument("--no_vis", action="store_false", dest="save_vis",
                        help="Skip visualisation")
    parser.add_argument("--save_uncertainty", action="store_true", default=False,
                        help="Save LUE uncertainty heatmap overlays "
                             "(only if model was trained with use_lue=True)")
    parser.add_argument("--line_width", type=int, default=2,
                        help="Bounding-box line width (default: 2)")

    args = parser.parse_args()

    # ---- resolve output directory -----------------------------------------
    if args.output_dir is None:
        ckpt_dir = Path(args.checkpoint).parent
        args.output_dir = ckpt_dir / "inference_results"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- gather images ----------------------------------------------------
    image_paths = gather_image_paths(
        image=args.image,
        image_dir=args.image_dir,
        image_list=args.image_list,
    )
    if not image_paths:
        print("[ERROR] No images found.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(image_paths)} image(s)")

    # ---- load model -------------------------------------------------------
    runner = InferenceRunner(checkpoint=args.checkpoint, device=args.device)

    # ---- run inference ----------------------------------------------------
    if len(image_paths) == 1 and args.batch_size == 1:
        print(f"[INFO] Single-image inference: {image_paths[0]}")
        results = [runner.predict_single(image_paths[0])]
    else:
        print(f"[INFO] Batch inference: {len(image_paths)} images, "
              f"batch_size={args.batch_size}")
        results = runner.predict_batch(image_paths, batch_size=args.batch_size)

    # ---- save -------------------------------------------------------------
    vis_dir = out_dir / "visualizations"
    unc_dir = out_dir / "uncertainty_maps"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_uncertainty:
        unc_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Saving results to {out_dir.resolve()}")
    total_dets = 0
    for i, res in enumerate(results):
        path = Path(res["path"])
        stem = path.stem
        ndet = int((res["scores"] >= args.score_thresh).sum().item())
        total_dets += ndet

        # --- visualisation ---
        if args.save_vis:
            try:
                img = Image.open(path).convert("RGB")
                vis = draw_detections(
                    img,
                    res["boxes"], res["scores"], res["labels"],
                    class_names=runner.class_names,
                    score_thresh=args.score_thresh,
                    max_dets=args.max_dets,
                    line_width=args.line_width,
                )
                vis.save(vis_dir / f"{stem}_detected.jpg", quality=95)
            except Exception as exc:
                print(f"[WARN] Visualisation failed for {path.name}: {exc}")

        # --- uncertainty heatmap ---
        if args.save_uncertainty and res.get("uncertainties") is not None:
            try:
                img = Image.open(path).convert("RGB")
                heat = draw_uncertainty_heatmap(
                    img,
                    res["boxes"], res["uncertainties"], res["scores"],
                    score_thresh=args.score_thresh,
                )
                heat.save(unc_dir / f"{stem}_uncertainty.jpg", quality=95)
            except Exception as exc:
                print(f"[WARN] Uncertainty heatmap failed for {path.name}: {exc}")

        # Progress indicator
        if (i + 1) % max(1, len(results) // 10) == 0 or i == len(results) - 1:
            print(f"  [{i+1}/{len(results)}] {path.name}: {ndet} detections ≥ {args.score_thresh:.2f}")

    # --- JSON ---
    if args.save_json:
        json_data = results_to_json(results, runner.class_names)
        json_path = out_dir / "detections.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] JSON results saved to {json_path.resolve()}")

    # --- summary ---
    avg_dets = total_dets / max(1, len(results))
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(results)} images, {total_dets} total detections "
          f"(avg {avg_dets:.1f}/image) @ score ≥ {args.score_thresh:.2f}")
    print(f"Output directory: {out_dir.resolve()}")
    if args.save_vis:
        print(f"  Visualisations:  {vis_dir.resolve()}")
    if args.save_uncertainty:
        print(f"  Uncertainty:     {unc_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
