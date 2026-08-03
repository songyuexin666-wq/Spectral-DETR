#!/usr/bin/env python3
"""
调试ExDark数据集的bbox格式
检查原始标注文件的实际坐标含义
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import random


def debug_single_annotation(image_path, label_path, output_path="debug_output.jpg"):
    """
    调试单个标注文件，测试不同的坐标解释方式

    Args:
        image_path: 图片路径
        label_path: 标注文件路径
        output_path: 输出路径
    """
    print(f"=" * 80)
    print(f"调试标注文件")
    print(f"图片: {image_path}")
    print(f"标注: {label_path}")
    print(f"=" * 80)

    # 读取图片
    img = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size
    print(f"\n图片尺寸: {img_w} x {img_h}")

    # 读取标注
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 解析第一个有效标注
    bbox_data = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('%'):
            continue
        parts = line.split()
        if len(parts) >= 5:
            bbox_data = parts
            break

    if bbox_data is None:
        print("❌ 没有找到有效的标注")
        return

    print(f"\n原始标注行:")
    print(f"  {' '.join(bbox_data)}")

    # 提取类别和坐标
    class_name = bbox_data[0]
    coords = [float(x) for x in bbox_data[1:5]]
    print(f"\n类别: {class_name}")
    print(f"坐标: {coords}")

    # 测试不同的坐标解释方式
    interpretations = []

    # 方式1: (x1, y1, x2, y2) - 左上角+右下角（当前转换代码的假设）
    x1, y1, x2, y2 = coords
    interpretations.append({
        'name': '方式1: (x1,y1,x2,y2) 左上+右下',
        'bbox': (x1, y1, x2, y2),
        'coco': [x1, y1, x2-x1, y2-y1]
    })

    # 方式2: (x, y, w, h) - COCO格式
    x, y, w, h = coords
    interpretations.append({
        'name': '方式2: (x,y,w,h) COCO格式',
        'bbox': (x, y, x+w, y+h),
        'coco': [x, y, w, h]
    })

    # 方式3: (cx, cy, w, h) - 中心点+宽高
    cx, cy, w, h = coords
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    interpretations.append({
        'name': '方式3: (cx,cy,w,h) 中心+宽高',
        'bbox': (x1, y1, x2, y2),
        'coco': [x1, y1, w, h]
    })

    # 方式4: 归一化的 (x1, y1, x2, y2)
    x1_norm, y1_norm, x2_norm, y2_norm = coords
    x1 = x1_norm * img_w
    y1 = y1_norm * img_h
    x2 = x2_norm * img_w
    y2 = y2_norm * img_h
    interpretations.append({
        'name': '方式4: 归一化(x1,y1,x2,y2)',
        'bbox': (x1, y1, x2, y2),
        'coco': [x1, y1, x2-x1, y2-y1]
    })

    # 方式5: KITTI格式 (left, top, right, bottom)
    left, top, right, bottom = coords
    interpretations.append({
        'name': '方式5: KITTI(left,top,right,bottom)',
        'bbox': (left, top, right, bottom),
        'coco': [left, top, right-left, bottom-top]
    })

    # 创建可视化
    print(f"\n测试不同坐标解释:")
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)

    for i, interp in enumerate(interpretations):
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)

        bbox = interp['bbox']
        x1, y1, x2, y2 = bbox

        # 检查是否在图片范围内
        in_bounds = (0 <= x1 < img_w and 0 <= y1 < img_h and
                     0 <= x2 <= img_w and 0 <= y2 <= img_h and
                     x2 > x1 and y2 > y1)

        # 绘制框
        try:
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((10, 10), interp['name'], fill='red')

            # 打印信息
            print(f"\n  {i+1}. {interp['name']}")
            print(f"     bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            print(f"     尺寸: {x2-x1:.1f} x {y2-y1:.1f}")
            print(f"     在范围内: {'✅' if in_bounds else '❌'}")

            if in_bounds:
                print(f"     ⭐ 这个解释可能是正确的!")
        except Exception as e:
            print(f"\n  {i+1}. {interp['name']}")
            print(f"     ❌ 绘制失败: {e}")

        # 保存
        output_file = output_dir / f"test_{i+1}_{interp['name'].replace(':', '').replace('/', '_')}.jpg"
        img_copy.save(output_file)

    print(f"\n✅ 可视化结果保存在: {output_dir}/")
    print(f"\n💡 请手动查看哪个可视化结果的框与实际物体对应！")


def main():
    """
    交互式调试工具
    """
    if len(sys.argv) < 3:
        print("用法:")
        print("  python debug_bbox_format.py <图片路径> <标注路径>")
        print("\n示例:")
        print("  python debug_bbox_format.py \\")
        print("    ExDark_Annno/images/Bicycle/2015_00001.jpg \\")
        print("    ExDark_Annno/labels/Bicycle/2015_00001.jpg.txt")
        sys.exit(1)

    image_path = sys.argv[1]
    label_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        sys.exit(1)

    if not os.path.exists(label_path):
        print(f"❌ 标注不存在: {label_path}")
        sys.exit(1)

    debug_single_annotation(image_path, label_path)


if __name__ == "__main__":
    main()
