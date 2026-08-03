#!/usr/bin/env python3
"""
验证 val 的标注与图片是否一一对应：
  - 标注里每张图在 val/ 下是否存在
  -（可选）val/ 下是否有标注里没出现的图片
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Validate instances_val.json vs val/ images (existence and optional 1:1 check)."
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="/root/autodl-tmp/rf-detr/datasets",
        help="数据集根目录，下含 annotations/、val/",
    )
    parser.add_argument(
        "--check_extra",
        action="store_true",
        help="是否检查 val 目录下存在但 json 中未列出的图片",
    )
    args = parser.parse_args()

    root = Path(args.datasets_root)
    ann_dir = root / "annotations"
    val_img_dir = root / "val"
    if not val_img_dir.exists() and (root / "val2017").exists():
        val_img_dir = root / "val2017"

    val_json = ann_dir / "instances_val.json"
    if not val_json.exists():
        print(f"未找到 {val_json}")
        return
    if not val_img_dir.exists():
        print(f"未找到图片目录 {val_img_dir}")
        return

    with open(val_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_ids_in_ann = {a["image_id"] for a in annotations}

    # 1. 标注里的每张图在 val/ 下是否存在
    missing = []
    for img in images:
        fname = img.get("file_name")
        if not fname:
            missing.append((img["id"], "(无 file_name)"))
            continue
        p = val_img_dir / fname
        if not p.exists():
            p = val_img_dir / Path(fname).name
        if not p.exists():
            missing.append((img["id"], fname))

    if missing:
        print(f"【缺失】标注中有 {len(missing)} 张图在 val 目录下找不到文件：")
        for img_id, fname in missing[:30]:
            print(f"  id={img_id}  file_name={fname!r}")
        if len(missing) > 30:
            print(f"  ... 共 {len(missing)} 条")
    else:
        print("【通过】标注中所有图片在 val 目录下均存在。")

    # 有标注但图缺失的统计
    missing_ids = {m[0] for m in missing}
    ann_without_img = len(image_ids_in_ann & missing_ids)
    if ann_without_img:
        print(f"  其中 {ann_without_img} 张缺失图在 annotations 中有标注。")

    # 2. 可选：val 下是否有 json 未列出的图片
    if args.check_extra:
        listed = {Path(img.get("file_name", "")).name for img in images if img.get("file_name")}
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        extra = []
        for p in val_img_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts and p.name not in listed:
                extra.append(p.name)
        if extra:
            print(f"\n【多出】val 目录下有 {len(extra)} 张图未在标注的 images 中出现：")
            for name in sorted(extra)[:30]:
                print(f"  {name}")
            if len(extra) > 30:
                print(f"  ... 共 {len(extra)} 张")
        else:
            print("\n【通过】val 目录下图片均在标注的 images 中列出。")

    # 简要统计
    print(f"\n统计: images={len(images)}, annotations={len(annotations)}, 缺失={len(missing)}")


if __name__ == "__main__":
    main()
