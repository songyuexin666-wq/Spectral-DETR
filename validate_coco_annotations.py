#!/usr/bin/env python3
"""
验证COCO格式标注的正确性
检查bbox是否在图片范围内，是否有异常值
"""
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def validate_coco_annotations(coco_json_path, images_dir, visualize_samples=5):
    """
    验证COCO格式标注文件

    Args:
        coco_json_path: COCO JSON文件路径
        images_dir: 图片目录路径
        visualize_samples: 可视化样本数量
    """
    print(f"=" * 80)
    print(f"验证COCO标注文件: {coco_json_path}")
    print(f"=" * 80)

    # 加载COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    print(f"\n📊 基本统计:")
    print(f"  图片数量: {len(images)}")
    print(f"  标注数量: {len(annotations)}")
    print(f"  类别数量: {len(categories)}")
    print(f"  类别列表: {list(categories.values())}")

    # 建立image_id到image的映射
    image_dict = {img['id']: img for img in images}

    # 统计信息
    bbox_stats = {
        'total': len(annotations),
        'out_of_bounds': 0,
        'negative_size': 0,
        'zero_area': 0,
        'too_small': 0,  # area < 10
        'too_large': 0,  # area > img_area * 0.9
    }

    problems = []

    print(f"\n🔍 检查标注质量...")
    for ann in annotations:
        img = image_dict[ann['image_id']]
        bbox = ann['bbox']  # [x, y, w, h]
        x, y, w, h = bbox
        img_w, img_h = img['width'], img['height']

        # 检查1: 负数尺寸
        if w < 0 or h < 0:
            bbox_stats['negative_size'] += 1
            problems.append({
                'type': 'negative_size',
                'image': img['file_name'],
                'bbox': bbox,
                'image_size': (img_w, img_h)
            })

        # 检查2: 零面积
        if w * h == 0:
            bbox_stats['zero_area'] += 1
            problems.append({
                'type': 'zero_area',
                'image': img['file_name'],
                'bbox': bbox
            })

        # 检查3: 超出边界
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            bbox_stats['out_of_bounds'] += 1
            problems.append({
                'type': 'out_of_bounds',
                'image': img['file_name'],
                'bbox': bbox,
                'image_size': (img_w, img_h),
                'overflow': {
                    'left': -x if x < 0 else 0,
                    'top': -y if y < 0 else 0,
                    'right': (x + w - img_w) if x + w > img_w else 0,
                    'bottom': (y + h - img_h) if y + h > img_h else 0
                }
            })

        # 检查4: 过小
        area = w * h
        if area < 10:
            bbox_stats['too_small'] += 1

        # 检查5: 过大
        if area > img_w * img_h * 0.9:
            bbox_stats['too_large'] += 1

    # 打印统计结果
    print(f"\n📈 标注质量统计:")
    print(f"  ✅ 正常标注: {bbox_stats['total'] - bbox_stats['out_of_bounds'] - bbox_stats['negative_size'] - bbox_stats['zero_area']}/{bbox_stats['total']}")
    print(f"  ❌ 超出边界: {bbox_stats['out_of_bounds']}")
    print(f"  ❌ 负数尺寸: {bbox_stats['negative_size']}")
    print(f"  ❌ 零面积: {bbox_stats['zero_area']}")
    print(f"  ⚠️  过小(<10px²): {bbox_stats['too_small']}")
    print(f"  ⚠️  过大(>90%): {bbox_stats['too_large']}")

    # 打印前5个问题
    if problems:
        print(f"\n❌ 发现 {len(problems)} 个问题，显示前5个:")
        for i, prob in enumerate(problems[:5]):
            print(f"\n  问题 {i+1}: {prob['type']}")
            print(f"    图片: {prob['image']}")
            print(f"    bbox: {prob['bbox']}")
            if 'image_size' in prob:
                print(f"    图片尺寸: {prob['image_size']}")
            if 'overflow' in prob:
                print(f"    溢出: {prob['overflow']}")

    # 可视化样本
    if visualize_samples > 0 and images_dir:
        print(f"\n🎨 可视化 {visualize_samples} 个样本...")
        visualize_annotations(coco_data, images_dir, visualize_samples)

    return bbox_stats, problems


def visualize_annotations(coco_data, images_dir, num_samples=5):
    """
    可视化标注样本
    """
    images_dir = Path(images_dir)
    output_dir = Path("validation_output")
    output_dir.mkdir(exist_ok=True)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # 建立image_id到annotations的映射
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # 随机选择有标注的图片
    images_with_anns = [img for img in images if img['id'] in img_to_anns and len(img_to_anns[img['id']]) > 0]
    sample_images = random.sample(images_with_anns, min(num_samples, len(images_with_anns)))

    for idx, img_info in enumerate(sample_images):
        img_path = images_dir / img_info['file_name']
        if not img_path.exists():
            print(f"  ⚠️  图片不存在: {img_path}")
            continue

        # 加载图片
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # 绘制标注
        anns = img_to_anns[img_info['id']]
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            x, y, w, h = bbox
            cat_name = categories[ann['category_id']]

            # 绘制矩形
            draw.rectangle([x, y, x+w, y+h], outline='red', width=2)

            # 绘制标签
            text = f"{cat_name}"
            draw.text((x, y-10), text, fill='red')

        # 保存
        output_path = output_dir / f"sample_{idx+1}_{img_info['file_name']}"
        img.save(output_path)
        print(f"  ✅ 保存可视化: {output_path} ({len(anns)} 个标注)")


def main():
    parser = argparse.ArgumentParser(description="验证COCO格式标注文件")
    parser.add_argument('--json', type=str, required=True, help='COCO JSON文件路径')
    parser.add_argument('--images', type=str, required=True, help='图片目录路径')
    parser.add_argument('--samples', type=int, default=5, help='可视化样本数量')

    args = parser.parse_args()

    validate_coco_annotations(args.json, args.images, args.samples)


if __name__ == "__main__":
    main()
