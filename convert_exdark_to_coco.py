#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExDark数据集（KITTI bbGt格式）转换为COCO格式
支持将ExDark数据集转换为Spectral-DETR项目可用的COCO格式

ExDark数据集结构:
    ExDark_Annno/
    ├── images/  (图片文件夹)
    │   ├── Bicycle/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── Boat/
    │   │   └── ...
    │   └── ...
    └── labels/  (标签文件夹)
        ├── Bicycle/
        │   ├── image1.txt (bbGt格式)
        │   ├── image2.txt
        │   └── ...
        ├── Boat/
        │   └── ...
        └── ...

COCO格式结构（标准格式）:
    exdark_coco/
    ├── train/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── val/
    │   ├── image3.jpg
    │   └── ...
    └── annotations/
        ├── instances_train.json
        └── instances_val.json
"""

import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse


# ExDark的12个类别
EXDARK_CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]


def parse_bbgt_annotation(label_path, img_width, img_height, default_class_name=None):
    """
    解析KITTI bbGt格式标注文件

    ExDark的bbGt格式可能是：
    1. 类别名 x1 y1 x2 y2 [difficult] [其他信息]
    2. x1 y1 x2 y2 [difficult] [class_id]
    3. 完整KITTI格式: class truncated occluded alpha bbox_2d ...

    这里我们处理ExDark格式：类别名（文本）x1 y1 x2 y2 [difficult]

    Args:
        label_path: bbGt标注文件路径
        img_width: 图片宽度
        img_height: 图片高度
        default_class_name: 默认类别名称（从文件夹名推断）

    Returns:
        annotations: 标注列表，每个元素包含bbox、category_id 以及 difficult（不会映射到 iscrowd）
    """
    annotations = []

    if not os.path.exists(label_path):
        return annotations

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"警告: 无法读取标注文件 {label_path}: {e}")
        return annotations

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        try:
            # 尝试解析ExDark格式：类别名 x1 y1 x2 y2 [difficult]
            # 首先检查第一个部分是否是类别名称
            first_part = parts[0]
            category_name = None
            category_id = None
            coord_start_idx = 0

            # 检查是否是类别名称
            if first_part in EXDARK_CLASSES:
                category_name = first_part
                category_id = EXDARK_CLASSES.index(category_name) + 1  # 1-based: 1, 2, 3, ..., 12
                coord_start_idx = 1

            # 如果第一个不是类别名，尝试从后面找
            if category_name is None:
                for i, part in enumerate(parts):
                    if part in EXDARK_CLASSES:
                        category_name = part
                        category_id = EXDARK_CLASSES.index(category_name) + 1  # 1-based: 1, 2, 3, ..., 12
                        # 移除类别名，重新组织parts
                        parts = parts[:i] + parts[i+1:]
                        break

            # 如果还是找不到类别名，使用默认类别（从文件夹名）
            if category_name is None and default_class_name:
                if default_class_name in EXDARK_CLASSES:
                    category_name = default_class_name
                    category_id = EXDARK_CLASSES.index(category_name) + 1  # 1-based: 1, 2, 3, ..., 12

            # 提取坐标（跳过类别名）
            coord_parts = parts[coord_start_idx:]

            # 尝试找到4个连续的数字作为坐标
            coords = []
            for part in coord_parts:
                try:
                    coord = float(part)
                    coords.append(coord)
                    if len(coords) >= 4:
                        break
                except ValueError:
                    # 如果不是数字，跳过
                    continue

            if len(coords) < 4:
                # 如果找不到4个坐标，尝试其他格式
                # 可能是：x1 y1 x2 y2 格式，但前面有类别名
                if len(parts) >= 5:  # 至少5个部分（类别名 + 4个坐标）
                    try:
                        # 尝试从位置1开始读取4个数字
                        coords = [float(parts[i]) for i in range(1, 5)]
                    except (ValueError, IndexError):
                        continue
                else:
                    continue

            # ✅ 修复：ExDark的bbGt格式已经是 (x, y, w, h) COCO格式！
            # 不是 (x1, y1, x2, y2) 格式！
            x, y, w, h = coords[:4]

            # 处理difficult标志（如果有）
            difficult = 0
            # ExDark格式中，difficult通常是坐标后面的第5个或更后面的数字（0或1）
            # 只检查坐标之后的部分，避免将坐标本身误判为difficult
            coord_count = 0
            for part in coord_parts:
                try:
                    float(part)  # 如果是数字
                    coord_count += 1
                    # 如果已经读取了4个坐标，后续的0或1才是difficult标志
                    if coord_count > 4 and part in ['0', '1']:
                        difficult = int(part)
                        break
                except ValueError:
                    continue

            # 边界检查：确保bbox在图片范围内
            # 注意：x, y, w, h 已经是COCO格式，不需要转换
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))

            # 如果bbox超出边界，裁剪宽高
            if x + w > img_width:
                w = img_width - x
            if y + h > img_height:
                h = img_height - y

            # 确保bbox有效（宽高都大于0）
            if w > 0 and h > 0:
                # COCO格式: [x, y, width, height] (左上角坐标 + 宽高)
                bbox = [x, y, w, h]
                area = w * h

                # 如果没有找到类别ID，使用默认类别
                if category_id is None and default_class_name:
                    if default_class_name in EXDARK_CLASSES:
                        category_id = EXDARK_CLASSES.index(default_class_name) + 1  # 1-based: 1, 2, 3, ..., 12

                if category_id is None:
                    # 如果还是找不到，跳过这个标注
                    print(f"警告: 无法确定类别ID，跳过标注: {label_path} 第 {line_idx+1} 行")
                    continue

                annotation = {
                    "bbox": bbox,
                    "area": area,
                    # ExDark 的 difficult != COCO 的 iscrowd；iscrowd=1 会被本项目的数据加载器过滤掉
                    # 因此这里保留 difficult 字段，但不用于 iscrowd
                    "difficult": difficult,
                    "category_id": category_id,
                }

                annotations.append(annotation)

        except (ValueError, IndexError) as e:
            # 只在调试时打印详细错误
            # print(f"警告: 解析标注文件 {label_path} 第 {line_idx+1} 行时出错: {e}，跳过")
            continue

    return annotations


def get_image_size(image_path):
    """获取图片尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"错误: 无法读取图片 {image_path}: {e}")
        return None


def find_image_label_pairs(images_dir, labels_dir, class_name):
    """
    查找某个类别下的所有图片和对应的标注文件

    Args:
        images_dir: 图片根目录（包含各个类别文件夹）
        labels_dir: 标签根目录（包含各个类别文件夹）
        class_name: 类别名称

    Returns:
        list: [(image_path, label_path), ...]
    """
    image_label_pairs = []

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}

    # 类别文件夹路径
    class_images_dir = images_dir / class_name
    class_labels_dir = labels_dir / class_name

    if not class_images_dir.exists():
        print(f"警告: 图片文件夹不存在: {class_images_dir}")
        return image_label_pairs

    if not class_labels_dir.exists():
        print(f"警告: 标签文件夹不存在: {class_labels_dir}")
        return image_label_pairs

    # 遍历类别图片文件夹下的所有图片
    for img_file in class_images_dir.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            # 查找对应的标签文件
            # ExDark格式：标签文件名是 图片名.jpg.txt（而不是图片名.txt）
            # 例如：图片是 2015_06857.jpg，标签是 2015_06857.jpg.txt
            label_file = class_labels_dir / (img_file.name + '.txt')

            if label_file.exists():
                image_label_pairs.append((img_file, label_file))
            else:
                # 如果找不到 .jpg.txt 格式，尝试 .txt 格式（向后兼容）
                label_file_alt = class_labels_dir / (img_file.stem + '.txt')
                if label_file_alt.exists():
                    image_label_pairs.append((img_file, label_file_alt))
                else:
                    print(f"警告: 未找到对应的标签文件: {label_file} 或 {label_file_alt}")

    return image_label_pairs


def convert_exdark_to_coco(exdark_root, output_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    将ExDark数据集转换为COCO格式

    Args:
        exdark_root: ExDark_Annno根目录路径（包含images/和labels/文件夹）
        output_path: 输出COCO格式数据集路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例（可选）
        seed: 随机种子
    """
    exdark_path = Path(exdark_root)
    output_path = Path(output_path)

    # 检查输入路径
    if not exdark_path.exists():
        raise FileNotFoundError(f"ExDark数据集路径不存在: {exdark_root}")

    # 检查images和labels文件夹
    images_dir = exdark_path / "images"
    labels_dir = exdark_path / "labels"

    # 尝试其他可能的文件夹名称
    if not images_dir.exists():
        possible_image_names = ["Images", "image", "Image", "imgs", "Imgs"]
        for name in possible_image_names:
            test_dir = exdark_path / name
            if test_dir.exists():
                images_dir = test_dir
                break

    if not labels_dir.exists():
        possible_label_names = ["Labels", "label", "Label", "annotations", "Annotations", "anns", "Anns"]
        for name in possible_label_names:
            test_dir = exdark_path / name
            if test_dir.exists():
                labels_dir = test_dir
                break

    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图片文件夹。请确保存在 images/ 文件夹: {exdark_root}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"未找到标签文件夹。请确保存在 labels/ 文件夹: {exdark_root}")

    print(f"✓ 找到图片文件夹: {images_dir}")
    print(f"✓ 找到标签文件夹: {labels_dir}")

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建输出子目录（标准COCO格式）
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    annotations_dir = output_path / "annotations"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    if test_ratio > 0:
        test_dir.mkdir(exist_ok=True)

    # 设置随机种子
    random.seed(seed)

    # 分层划分：对每个类别单独按比例划分
    train_data = []
    val_data = []
    test_data = []

    print("\n扫描数据集并进行分层划分...")
    print("=" * 80)

    total_images = 0
    for class_idx, class_name in enumerate(EXDARK_CLASSES):
        # 查找该类别下的所有图片和标注文件
        image_label_pairs = find_image_label_pairs(images_dir, labels_dir, class_name)

        class_total = len(image_label_pairs)
        total_images += class_total

        if class_total == 0:
            print(f"  类别 {class_name:12}: 0 个图片（跳过）")
            continue

        # 随机打乱该类别内的图片顺序
        random.shuffle(image_label_pairs)

        # 计算该类别的划分点
        class_train_end = int(class_total * train_ratio)
        class_val_end = class_train_end + int(class_total * val_ratio)

        # 按比例划分该类别的数据
        class_train = image_label_pairs[:class_train_end]
        class_val = image_label_pairs[class_train_end:class_val_end]
        class_test = image_label_pairs[class_val_end:] if test_ratio > 0 else []

        # 转换为字典格式并添加到对应集合
        class_id = class_idx + 1  # 1-based: 1, 2, 3, ..., 12（标准COCO格式）

        for img_path, label_path in class_train:
            train_data.append({
                'image_path': img_path,
                'label_path': label_path,
                'class_name': class_name,
                'class_id': class_id
            })

        for img_path, label_path in class_val:
            val_data.append({
                'image_path': img_path,
                'label_path': label_path,
                'class_name': class_name,
                'class_id': class_id
            })

        if test_ratio > 0:
            for img_path, label_path in class_test:
                test_data.append({
                    'image_path': img_path,
                    'label_path': label_path,
                    'class_name': class_name,
                    'class_id': class_id
                })

        # 打印该类别的划分统计
        print(f"  类别 {class_name:12}: 总计 {class_total:4} | "
              f"Train {len(class_train):4} ({len(class_train)/class_total*100:5.1f}%) | "
              f"Val {len(class_val):4} ({len(class_val)/class_total*100:5.1f}%)" +
              (f" | Test {len(class_test):4} ({len(class_test)/class_total*100:5.1f}%)" if test_ratio > 0 else ""))

    # 最后再次打乱各个集合（保证训练时batch内类别混合）
    random.shuffle(train_data)
    random.shuffle(val_data)
    if test_ratio > 0:
        random.shuffle(test_data)

    print("=" * 80)

    print(f"\n数据集划分汇总:")
    print(f"  总图片数: {total_images}")
    print(f"  训练集: {len(train_data):4} ({len(train_data)/total_images*100:.1f}%)")
    print(f"  验证集: {len(val_data):4} ({len(val_data)/total_images*100:.1f}%)")
    if test_data:
        print(f"  测试集: {len(test_data):4} ({len(test_data)/total_images*100:.1f}%)")

    # 处理每个数据集分割
    splits = [
        ('train', train_data, train_dir),
        ('val', val_data, val_dir),  # 注意：标准COCO格式使用val而不是valid
    ]
    if test_data:
        splits.append(('test', test_data, test_dir))

    for split_name, split_data, split_dir in splits:
        print(f"\n处理 {split_name} 集...")

        # COCO JSON 数据结构
        coco_data = {
            "info": {
                "description": f"ExDark Dataset - {split_name} set",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [],
            "categories": [
                {
                    "id": idx + 1,  # 1-based: 1, 2, 3, ..., 12（标准COCO格式）
                    "name": class_name,
                    "supercategory": "none"
                }
                for idx, class_name in enumerate(EXDARK_CLASSES)
            ],
            "images": [],
            "annotations": []
        }

        # 处理每张图片
        # COCO格式要求ID从1开始且连续
        image_id = 0
        annotation_id = 0

        for item in tqdm(split_data, desc=f"转换 {split_name} 图片"):
            img_path = item['image_path']
            label_path = item['label_path']
            class_name = item['class_name']
            class_id = item['class_id']  # 1-based class_id

            # 获取图片尺寸
            img_size = get_image_size(img_path)
            if img_size is None:
                continue

            img_width, img_height = img_size

            # 先增加image_id（COCO格式要求从1开始）
            image_id += 1

            # 复制图片到输出目录（使用唯一文件名避免冲突）
            # 使用原始文件名+类别名+ID确保唯一性
            original_name = img_path.stem
            # 使用image_id生成文件名（此时image_id已经是1, 2, 3...）
            img_filename = f"{class_name}_{original_name}_{image_id:06d}{img_path.suffix}"
            output_img_path = split_dir / img_filename
            shutil.copy2(img_path, output_img_path)

            # 添加图片信息
            image_info = {
                "id": image_id,  # COCO格式：从1开始，连续递增
                "file_name": img_filename,
                "width": img_width,
                "height": img_height
            }
            coco_data["images"].append(image_info)

            # 解析标注文件（传入默认类别名）
            annotations = parse_bbgt_annotation(label_path, img_width, img_height, default_class_name=class_name)

            for ann in annotations:
                # 先增加annotation_id（COCO格式要求从1开始）
                annotation_id += 1

                # 获取category_id（1-based，范围1-12）
                category_id = ann.get('category_id')

                # 如果标注文件中没有category_id，使用文件夹名对应的类别
                if category_id is None:
                    category_id = class_id  # 已经是1-based

                # 确保category_id在有效范围内（1到12）
                if category_id < 1 or category_id > len(EXDARK_CLASSES):
                    category_id = class_id
                    print(f"警告: 类别ID {ann.get('category_id')} 超出范围，使用默认类别 {category_id} (对应 {class_name})")

                annotation = {
                    "id": annotation_id,  # 从1开始，每个split独立计数
                    "image_id": image_id,  # 关联到对应的图片ID
                    "category_id": category_id,  # 1-based: 1-12（标准COCO格式，ConvertCoco会自动转为0-based）
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    # 关键：本项目会过滤 iscrowd=1 的标注；ExDark difficult 不应映射为 iscrowd
                    "iscrowd": 0,
                    # 保留原始 difficult，便于后续分析/过滤（训练时不会用到）
                    "difficult": int(ann.get("difficult", 0))
                }

                coco_data["annotations"].append(annotation)

        # 保存COCO JSON文件（标准COCO格式，放在annotations目录）
        if split_name == "test":
            json_filename = "image_info_test-dev.json"
        else:
            json_filename = f"instances_{split_name}.json"
        json_path = annotations_dir / json_filename

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"✓ {split_name} 集转换完成:")
        print(f"  - 图片数量: {len(coco_data['images'])}")
        print(f"  - 标注数量: {len(coco_data['annotations'])}")
        print(f"  - 类别数量: {len(coco_data['categories'])}")
        print(f"  - JSON 文件: {json_path}")

    print(f"\n{'='*60}")
    print("转换完成！")
    print(f"输出目录: {output_path}")
    print(f"\n数据集结构（标准COCO格式）:")
    print(f"  {output_path}/")
    print(f"    ├── train/")
    print(f"    │   ├── image1.jpg")
    print(f"    │   ├── image2.jpg")
    print(f"    │   └── ...")
    print(f"    ├── val/")
    print(f"    │   ├── image3.jpg")
    print(f"    │   └── ...")
    if test_ratio > 0:
        print(f"    ├── test/")
        print(f"    │   ├── image4.jpg")
        print(f"    │   └── ...")
    print(f"    └── annotations/")
    print(f"        ├── instances_train.json")
    print(f"        ├── instances_val.json")
    if test_ratio > 0:
        print(f"        └── image_info_test-dev.json")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="将ExDark数据集（KITTI bbGt格式）转换为COCO格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python convert_exdark_to_coco.py \\
      --input /root/autodl-tmp/1/ExDark_Annno \\
      --output /root/autodl-tmp/1/exdark_coco

  # 自定义数据集划分比例
  python convert_exdark_to_coco.py \\
      --input /root/autodl-tmp/1/ExDark_Annno \\
      --output /root/autodl-tmp/1/exdark_coco \\
      --train-ratio 0.8 \\
      --val-ratio 0.2 \\
      --test-ratio 0.0

ExDark数据集结构:
  ExDark_Annno/
  ├── images/
  │   ├── Bicycle/
  │   │   ├── image1.jpg
  │   │   ├── image2.jpg
  │   │   └── ...
  │   ├── Boat/
  │   │   └── ...
  │   └── ...
  └── labels/
      ├── Bicycle/
      │   ├── image1.txt (bbGt格式)
      │   ├── image2.txt
      │   └── ...
      ├── Boat/
      │   └── ...
      └── ...

输出COCO数据集结构（标准格式）:
  exdark_coco/
  ├── train/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── val/
  │   ├── image3.jpg
  │   └── ...
  └── annotations/
      ├── instances_train.json
      └── instances_val.json
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='ExDark数据集根目录路径（ExDark_Annno）'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出COCO格式数据集路径'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='训练集比例（默认: 0.7）'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='验证集比例（默认: 0.2）'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='测试集比例（默认: 0.1，设为0则不创建测试集）'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）'
    )

    args = parser.parse_args()

    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 数据集划分比例之和为 {total_ratio:.2f}，不等于1.0，将自动归一化")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    try:
        convert_exdark_to_coco(
            exdark_root=args.input,
            output_path=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
