#!/usr/bin/env python3
"""
撤回「把 test 合并进 val」的操作：
  1. 用 instances_val.json.bak 恢复 instances_val.json
  2. 按 image_info_test-dev.json 里的图片列表，从 val/ 里删除当时从 test 复制过去的文件

与 merge_test_into_val.py 使用相同的 --datasets_root 默认值。
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Revert merge: restore instances_val.json from .bak and remove test images from val/."
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="/root/autodl-tmp/rf-detr/datasets",
        help="数据集根目录（与 merge 脚本一致）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要执行的操作，不删除、不覆盖",
    )
    args = parser.parse_args()

    root = Path(args.datasets_root)
    ann_dir = root / "annotations"
    val_img_dir = root / "val"
    if not val_img_dir.exists() and (root / "val2017").exists():
        val_img_dir = root / "val2017"

    val_json = ann_dir / "instances_val.json"
    backup_json = ann_dir / "instances_val.json.bak"
    test_dev_json = ann_dir / "image_info_test-dev.json"

    if not backup_json.exists():
        print(f"未找到备份 {backup_json}，无法恢复 instances_val.json。")
        print("请确认当时合并时没有加 --no_backup。")
        return

    if not test_dev_json.exists():
        raise FileNotFoundError(f"未找到 {test_dev_json}")

    with open(test_dev_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_file_names = {img.get("file_name") for img in test_data.get("images", []) if img.get("file_name")}

    if args.dry_run:
        print("[dry_run] 将恢复:", backup_json, "->", val_json)
        print("[dry_run] 将从 val 删除以下 test 来源的图片:", list(test_file_names)[:5], "..." if len(test_file_names) > 5 else "")
        return

    # 1. 恢复 instances_val.json
    with open(backup_json, "r", encoding="utf-8") as f:
        bak_data = json.load(f)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(bak_data, f, ensure_ascii=False, indent=2)
    print(f"已恢复: {val_json} <- {backup_json}")

    # 2. 从 val 里删除当时从 test 复制过去的文件（只删 json 里列出的 test 图片名）
    removed = 0
    for fname in test_file_names:
        path = val_img_dir / Path(fname).name
        if path.exists():
            path.unlink()
            removed += 1
    print(f"已从 {val_img_dir} 删除 {removed} 张来自 test 的图片。")


if __name__ == "__main__":
    main()
