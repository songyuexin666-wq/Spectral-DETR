#!/usr/bin/env python3
"""
将 COCO 格式数据集转为 YOLO 格式，便于用同一套数据跑 YOLO 对比实验。

COCO 输入: 根目录下 train/, val/, annotations/instances_train.json, instances_val.json
YOLO 输出: 根目录下 images/train, images/val, labels/train, labels/val（归一化中心+宽高）
"""

import argparse
import json
import shutil
from pathlib import Path


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x,y,w,h] 转 YOLO 归一化 (x_center, y_center, w, h)."""
    x, y, w, h = bbox
    xc = x + w / 2
    yc = y + h / 2
    return xc / img_w, yc / img_h, w / img_w, h / img_h


def convert_split(coco_root: Path, out_root: Path, split: str, ann_name: str):
    ann_file = coco_root / "annotations" / ann_name
    img_dir_src = coco_root / split
    if not ann_file.exists() or not img_dir_src.exists():
        return 0, 0
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    id_to_cat = sorted(categories.keys())
    cat_to_idx = {cid: i for i, cid in enumerate(id_to_cat)}
    anns_by_img = {}
    for ann in data.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n_img, n_ann = 0, 0
    for iid, img in images.items():
        fname = img.get("file_name")
        if not fname:
            continue
        src = img_dir_src / fname
        if not src.exists():
            src = img_dir_src / Path(fname).name
        if not src.exists():
            continue
        w, h = img.get("width"), img.get("height")
        if not w or not h:
            continue
        dst_img = out_img / Path(fname).name
        if dst_img.resolve() != src.resolve():
            shutil.copy2(src, dst_img)
        n_img += 1
        lines = []
        for ann in anns_by_img.get(iid, []):
            cid = ann.get("category_id")
            if cid not in cat_to_idx:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            xc, yc, wn, hn = coco_bbox_to_yolo(bbox, w, h)
            cls = cat_to_idx[cid]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            n_ann += 1
        label_path = out_lbl / (Path(fname).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return n_img, n_ann


def main():
    p = argparse.ArgumentParser(description="COCO 转 YOLO 格式，用于 YOLO 对比实验")
    p.add_argument("--coco_root", type=str, default="datasets", help="COCO 数据集根目录")
    p.add_argument("--out", type=str, default="datasets_yolo", help="YOLO 输出根目录")
    args = p.parse_args()
    coco_root = Path(args.coco_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    total_img, total_ann = 0, 0
    for split, ann_name in [("train", "instances_train.json"), ("val", "instances_val.json")]:
        ni, na = convert_split(coco_root, out_root, split, ann_name)
        total_img += ni
        total_ann += na
        print(f"{split}: images={ni}, annotations={na}")
    print(f"输出目录: {out_root}")
    print("dataset.yaml 示例:")
    print(f"  path: {out_root.resolve()}")
    print("  train: images/train")
    print("  val: images/val")


if __name__ == "__main__":
    main()
