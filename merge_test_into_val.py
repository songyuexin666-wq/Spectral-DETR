#!/usr/bin/env python3
"""
将 test 合并到 val：
  1. 把 annotations/image_info_test-dev.json 合并进 annotations/instances_val.json
  2. 把 datasets/test 下所有图片复制到 datasets/val

路径示例：/root/autodl-tmp/rf-detr/datasets（下含 annotations/、val/、test/）
运行前会备份 instances_val.json 为 instances_val.json.bak。
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge image_info_test-dev.json into instances_val.json, and test/ images into val/."
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="/root/autodl-tmp/rf-detr/datasets",
        help="数据集根目录，下含 annotations/、val/、test/",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不备份 instances_val.json，直接覆盖",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要执行的操作，不写文件、不复制图片",
    )
    args = parser.parse_args()

    root = Path(args.datasets_root)
    ann_dir = root / "annotations"
    val_img_dir = root / "val"
    test_img_dir = root / "test"

    if not val_img_dir.exists() and (root / "val2017").exists():
        val_img_dir = root / "val2017"
    if not test_img_dir.exists() and (root / "test2017").exists():
        test_img_dir = root / "test2017"

    val_json = ann_dir / "instances_val.json"
    test_json = ann_dir / "image_info_test-dev.json"

    if not val_json.exists():
        raise FileNotFoundError(f"未找到 {val_json}")
    if not test_json.exists():
        raise FileNotFoundError(f"未找到 {test_json}")
    if not test_img_dir.exists():
        raise FileNotFoundError(f"未找到 test 图片目录 {test_img_dir}")

    with open(val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    val_images = {img["id"]: img for img in val_data.get("images", [])}
    val_annotations = list(val_data.get("annotations", []))
    val_categories = val_data.get("categories", [])
    max_val_id = max(val_images.keys(), default=0)

    test_images_raw = test_data.get("images", [])
    if not test_images_raw:
        print("image_info_test-dev.json 中 images 为空，无需合并。")
        return

    # 把 test 里所有图片并入 val 的 json，id 冲突则重映射
    old_to_new = {}
    new_images = []
    for img in test_images_raw:
        old_id = img["id"]
        if old_id in val_images:
            new_id = max_val_id + 1
            max_val_id = new_id
            old_to_new[old_id] = new_id
            new_images.append({**img, "id": new_id})
        else:
            old_to_new[old_id] = old_id
            new_images.append(img)
            max_val_id = max(max_val_id, old_id)

    # test 里若有 annotations 一并合并并重映射 image_id
    test_annotations_raw = test_data.get("annotations", [])
    new_annotations = []
    for ann in test_annotations_raw:
        new_image_id = old_to_new.get(ann["image_id"], ann["image_id"])
        new_annotations.append({**ann, "image_id": new_image_id})
    merged_annotations = val_annotations + new_annotations

    merged = {
        "images": list(val_images.values()) + new_images,
        "annotations": merged_annotations,
        "categories": val_categories,
    }
    if not val_categories and test_data.get("categories"):
        merged["categories"] = test_data["categories"]

    n_test_ann = len(test_annotations_raw)
    if n_test_ann == 0:
        print(
            f"合并后: images={len(merged['images'])} (val {len(val_images)} + test {len(new_images)}), "
            f"annotations={len(merged['annotations'])}（test 无标注，仅合并了图片列表）。"
        )
    else:
        print(
            f"合并后: images={len(merged['images'])}, annotations={len(merged['annotations'])} "
            f"(val + test 有标注 {n_test_ann} 条)。"
        )

    if args.dry_run:
        print("[dry_run] 将写入:", val_json)
        print("[dry_run] 将复制 test 图片到:", val_img_dir)
        return

    if not args.no_backup:
        backup_path = val_json.with_suffix(val_json.suffix + ".bak")
        shutil.copy2(val_json, backup_path)
        print(f"已备份: {backup_path}")

    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"已写入: {val_json}")

    # 把合并进 val 的 test 图片复制到 val 目录（与 json 里新增的 images 对应）
    val_img_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img in new_images:
        fname = img.get("file_name")
        if not fname:
            continue
        src = test_img_dir / fname
        if not src.exists():
            # 可能在 test 子目录里
            src = test_img_dir / Path(fname).name
        if src.exists():
            dst = val_img_dir / Path(fname).name
            if dst.resolve() != src.resolve():
                shutil.copy2(src, dst)
                copied += 1
        else:
            print(f"  警告: 未找到 test 图片 {src}")
    print(f"已复制 {copied} 张 test 图片到 {val_img_dir}")


if __name__ == "__main__":
    main()
