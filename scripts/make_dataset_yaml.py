#!/usr/bin/env python3
"""
从 COCO 的 instances_train.json 读取类别，生成 YOLO 用的 dataset.yaml。
用法: python scripts/make_dataset_yaml.py --coco_root datasets --yolo_root datasets_yolo --out dataset.yaml
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="从 COCO 生成 dataset.yaml（含 nc、names）")
    p.add_argument("--coco_root", type=str, default="datasets", help="COCO 数据集根目录")
    p.add_argument("--yolo_root", type=str, default="datasets_yolo", help="YOLO 数据根目录（path 填此）")
    p.add_argument("--out", type=str, default="dataset.yaml", help="输出 yaml 路径")
    p.add_argument("--absolute", action="store_true", help="path 使用绝对路径")
    args = p.parse_args()

    coco_root = Path(args.coco_root)
    ann_file = coco_root / "annotations" / "instances_train.json"
    if not ann_file.exists():
        ann_file = coco_root / "annotations" / "instances_val.json"
    if not ann_file.exists():
        print(f"未找到 {coco_root / 'annotations' / 'instances_train.json'} 或 instances_val.json")
        return

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = sorted(data.get("categories", []), key=lambda x: x["id"])
    nc = len(categories)
    # Ultralytics 支持 list 或 dict；用 list 最稳妥
    names_list = [c["name"] for c in categories]

    path = Path(args.yolo_root).resolve() if args.absolute else args.yolo_root
    names_str = "[" + ", ".join(repr(n) for n in names_list) + "]"
    content = f"""# 由 make_dataset_yaml.py 从 COCO 生成，用于 YOLO 对比实验
path: {path}
train: images/train
val: images/val
nc: {nc}
names: {names_str}
"""
    out = Path(args.out)
    out.write_text(content, encoding="utf-8")
    print(f"已写入 {out} (nc={nc})")


if __name__ == "__main__":
    main()
