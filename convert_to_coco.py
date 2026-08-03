#!/usr/bin/env python3
"""
将 JSON 标注文件转换为 RF-DETR 项目所需的 COCO 格式
"""
import json
import os
import sys
from pathlib import Path
from PIL import Image

def convert_bbox_format(bbox, from_format='auto', to_format='coco'):
    """
    转换 bbox 格式
    COCO 格式: [x, y, width, height] (左上角坐标 + 宽高)
    其他格式: [x1, y1, x2, y2] (左上角和右下角坐标)
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox 必须包含4个值，当前: {bbox}")

    if from_format == 'auto':
        # 自动检测：如果 width 或 height 为负数，可能是 x1,y1,x2,y2 格式
        if bbox[2] < 0 or bbox[3] < 0:
            from_format = 'xyxy'
        else:
            from_format = 'coco'

    if from_format == to_format:
        return bbox

    if from_format == 'xyxy' and to_format == 'coco':
        # [x1, y1, x2, y2] -> [x, y, width, height]
        x1, y1, x2, y2 = bbox
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return [x, y, width, height]
    elif from_format == 'coco' and to_format == 'xyxy':
        # [x, y, width, height] -> [x1, y1, x2, y2]
        x, y, width, height = bbox
        return [x, y, x + width, y + height]
    else:
        return bbox

def validate_and_fix_coco_format(input_file, output_file=None, image_dir=None):
    """
    验证并修复 COCO 格式标注文件
    """
    print("=" * 80)
    print("转换 COCO 格式标注文件...")
    print("=" * 80)

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return False

    # 读取原始 JSON
    print(f"\n读取文件: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n原始数据统计:")
    print(f"  - images: {len(data.get('images', []))}")
    print(f"  - annotations: {len(data.get('annotations', []))}")
    print(f"  - categories: {len(data.get('categories', []))}")

    # 1. 处理 categories
    print(f"\n1. 处理 categories...")
    categories = data.get('categories', [])
    if len(categories) == 0:
        print("  ⚠️  categories 为空，无法继续")
        return False

    # 确保 category id 从 1 开始且连续
    category_id_map = {}
    fixed_categories = []
    for idx, cat in enumerate(categories, 1):
        old_id = cat.get('id', idx)
        new_id = idx
        category_id_map[old_id] = new_id

        fixed_cat = {
            'id': new_id,
            'name': cat.get('name', f'class_{new_id}'),
            'supercategory': cat.get('supercategory', 'none')
        }
        fixed_categories.append(fixed_cat)

    print(f"  ✓ 处理了 {len(fixed_categories)} 个类别")
    print(f"    类别映射: {category_id_map}")

    # 2. 处理 images
    print(f"\n2. 处理 images...")
    images = data.get('images', [])

    # 如果 images 为空，从图片目录生成
    if len(images) == 0:
        if image_dir is None:
            # 尝试从输入文件路径推断
            image_dir = input_path.parent
        else:
            image_dir = Path(image_dir)

        if image_dir.exists():
            print(f"  images 为空，从目录生成: {image_dir}")
            img_files = sorted([f for f in os.listdir(image_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            images = []
            for idx, img_file in enumerate(img_files, 1):
                try:
                    img_path = image_dir / img_file
                    with Image.open(img_path) as img:
                        width, height = img.size
                except:
                    width, height = 640, 640

                images.append({
                    'id': idx,
                    'license': 1,
                    'file_name': img_file,
                    'height': height,
                    'width': width,
                    'date_captured': ''
                })
            print(f"  ✓ 生成了 {len(images)} 个图片条目")
        else:
            print(f"  ❌ 无法生成 images：目录不存在 {image_dir}")
            return False
    else:
        # 验证和修复 images
        print(f"  验证 {len(images)} 个图片条目...")
        image_id_map = {}
        fixed_images = []

        for idx, img in enumerate(images, 1):
            old_id = img.get('id', idx)
            new_id = idx
            image_id_map[old_id] = new_id

            # 验证必需字段
            file_name = img.get('file_name', '')
            if not file_name:
                print(f"  ⚠️  图片 ID {old_id} 缺少 file_name，跳过")
                continue

            fixed_img = {
                'id': new_id,
                'license': img.get('license', 1),
                'file_name': file_name,
                'height': img.get('height', 640),
                'width': img.get('width', 640),
                'date_captured': img.get('date_captured', '')
            }
            fixed_images.append(fixed_img)

        images = fixed_images
        print(f"  ✓ 处理了 {len(images)} 个图片条目")

    # 3. 处理 annotations
    print(f"\n3. 处理 annotations...")
    annotations = data.get('annotations', [])

    if len(annotations) == 0:
        print("  ⚠️  annotations 为空")
    else:
        print(f"  验证和修复 {len(annotations)} 个标注...")

        # 创建 image_id 映射（如果 images 被重新编号）
        image_id_to_new = {}
        if 'image_id_map' in locals():
            image_id_to_new = image_id_map
        else:
            # 从 images 创建映射
            image_ids = {img['id'] for img in images}
            for old_id in image_ids:
                image_id_to_new[old_id] = old_id

        fixed_annotations = []
        invalid_count = 0

        for idx, ann in enumerate(annotations, 1):
            # 映射 image_id
            old_image_id = ann.get('image_id')
            if old_image_id not in image_id_to_new:
                invalid_count += 1
                continue

            new_image_id = image_id_to_new[old_image_id]

            # 映射 category_id
            old_category_id = ann.get('category_id')
            if old_category_id not in category_id_map:
                invalid_count += 1
                continue

            new_category_id = category_id_map[old_category_id]

            # 处理 bbox
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                invalid_count += 1
                continue

            # 转换为 COCO 格式
            try:
                bbox_coco = convert_bbox_format(bbox, from_format='auto', to_format='coco')
            except:
                invalid_count += 1
                continue

            # 计算 area（如果不存在或无效）
            area = ann.get('area', 0)
            if area <= 0:
                area = bbox_coco[2] * bbox_coco[3]  # width * height

            fixed_ann = {
                'id': idx,
                'image_id': new_image_id,
                'category_id': new_category_id,
                'bbox': bbox_coco,
                'area': float(area),
                'iscrowd': int(ann.get('iscrowd', 0))
            }

            # 保留 segmentation（如果存在）
            if 'segmentation' in ann:
                fixed_ann['segmentation'] = ann['segmentation']

            fixed_annotations.append(fixed_ann)

        annotations = fixed_annotations
        print(f"  ✓ 处理了 {len(annotations)} 个标注")
        if invalid_count > 0:
            print(f"  ⚠️  跳过了 {invalid_count} 个无效标注")

    # 4. 构建最终的 COCO 格式数据（标准格式）
    print(f"\n4. 构建最终 COCO 格式...")

    # 确保输出格式完全符合 COCO 标准
    coco_data = {
        'info': data.get('info', {
            'description': 'Converted dataset',
            'version': '1.0',
            'year': 2024
        }),
        'licenses': data.get('licenses', []),
        'categories': fixed_categories,
        'images': images,
        'annotations': annotations
    }

    # 验证输出格式
    print(f"  验证输出格式...")
    assert 'images' in coco_data and isinstance(coco_data['images'], list)
    assert 'annotations' in coco_data and isinstance(coco_data['annotations'], list)
    assert 'categories' in coco_data and isinstance(coco_data['categories'], list)

    # 验证 images 格式
    if len(coco_data['images']) > 0:
        sample_img = coco_data['images'][0]
        required_img_fields = ['id', 'file_name', 'width', 'height']
        for field in required_img_fields:
            assert field in sample_img, f"images 缺少必需字段: {field}"

    # 验证 annotations 格式
    if len(coco_data['annotations']) > 0:
        sample_ann = coco_data['annotations'][0]
        required_ann_fields = ['id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd']
        for field in required_ann_fields:
            assert field in sample_ann, f"annotations 缺少必需字段: {field}"
        assert len(sample_ann['bbox']) == 4, "bbox 必须包含4个值"

    # 验证 categories 格式
    if len(coco_data['categories']) > 0:
        sample_cat = coco_data['categories'][0]
        required_cat_fields = ['id', 'name', 'supercategory']
        for field in required_cat_fields:
            assert field in sample_cat, f"categories 缺少必需字段: {field}"

    print(f"  ✓ 格式验证通过")

    # 5. 保存文件
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"

    output_path = Path(output_file)
    print(f"\n5. 保存到: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"\n最终统计:")
    print(f"  - images: {len(coco_data['images'])}")
    print(f"  - annotations: {len(coco_data['annotations'])}")
    print(f"  - categories: {len(coco_data['categories'])}")
    print(f"\n输出文件: {output_path}")
    print(f"\n使用说明:")
    print(f"  1. 将输出文件重命名为: _annotations.coco.json")
    print(f"  2. 放到对应的数据集目录（train/valid/test）")
    print(f"  3. 确保图片文件与标注文件中的 file_name 匹配")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_to_coco.py <input_json> [output_json] [image_dir]")
        print("\n示例:")
        print("  python convert_to_coco.py datasets/train/annotations.json")
        print("  python convert_to_coco.py datasets/train/annotations.json datasets/train/_annotations.coco.json datasets/train")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    image_dir = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        success = validate_and_fix_coco_format(input_file, output_file, image_dir)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

