import os
import json
import shutil
import glob
import xml.etree.ElementTree as ET
import random
import hashlib
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始数据集解压后的根目录 (请根据实际情况修改)
# 假设解压后里面有 'Annotations' 和 'JPEGImages' 文件夹
SOURCE_DIR = "datasets"

# 2. 输出目录 (将生成用于 rf-detr 的数据)
OUTPUT_DIR = "./coal_mine_coco"

# 3. 划分比例 (训练集占比)
TRAIN_RATIO = 0.8

# ===========================================

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    filename = root.find('filename').text

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # COCO 格式: [x_min, y_min, width, height]
        w = xmax - xmin
        h = ymax - ymin
        boxes.append({'name': name, 'bbox': [xmin, ymin, w, h]})

    return filename, width, height, boxes

def create_coco_structure(output_dir):
    for subdir in ["train", "val", "annotations"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def convert_to_coco(image_paths, annotations_map, categories, set_name, output_dir):
    coco_output = {
        "info": {"description": "Coal Mine Dataset Converted for RF-DETR"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for name, i in categories.items()]
    }

    print(f"正在处理 {set_name} 集...")

    ann_id = 1
    # 目标图片目录
    dest_img_dir = os.path.join(output_dir, f"{set_name}")

    for img_path in tqdm(image_paths):
        # 复制图片
        shutil.copy(img_path, dest_img_dir)

        # 获取文件名和 XML 路径
        file_name = os.path.basename(img_path)
        xml_path = annotations_map.get(os.path.splitext(file_name)[0])

        if not xml_path or not os.path.exists(xml_path):
            print(f"Warning: No XML found for {file_name}")
            continue

        # 解析 XML
        _, width, height, boxes = parse_xml(xml_path)

        # 添加图片信息 (使用文件名哈希或简单索引作为 image_id)
        # 这里为了简单，去除后缀作为 ID (如果文件名是纯数字最好，如果不是，可以用计数器)
        # 为了兼容性，我们使用简单的整数自增 ID
        image_id = coco_output["images"][-1]["id"] + 1 if coco_output["images"] else 1

        coco_output["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name
        })

        # 添加标注信息
        for box in boxes:
            cat_id = categories[box['name']]
            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": box['bbox'],
                "area": box['bbox'][2] * box['bbox'][3],
                "iscrowd": 0,
                "segmentation": [] # 目标检测通常不需要分割掩码
            })
            ann_id += 1

    # 保存 JSON
    json_path = os.path.join(output_dir, "annotations", f"instances_{set_name}.json")
    with open(json_path, 'w') as f:
        json.dump(coco_output, f)
    print(f"已保存: {json_path}")

def main():
    # 寻找文件
    # 假设结构是 SOURCE_DIR/JPEGImages/*.jpg 和 SOURCE_DIR/Annotations/*.xml
    # 如果你的结构不同，请在这里调整 glob 模式
    print("正在扫描文件...")
    # 尝试多种常见的图片扩展名
    img_files = []
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
        img_files.extend(glob.glob(os.path.join(SOURCE_DIR, '**', ext), recursive=True))

    # 建立 图片名(无后缀) -> 图片路径 的映射
    img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}

    # 寻找 XML 并建立映射
    xml_files = glob.glob(os.path.join(SOURCE_DIR, '**', '*.xml'), recursive=True)
    xml_map = {os.path.splitext(os.path.basename(f))[0]: f for f in xml_files}

    # 取交集，确保图片和标签对应
    # Sort before shuffling so PYTHONHASHSEED cannot change the partition.
    valid_names = sorted(set(img_map.keys()) & set(xml_map.keys()))
    print(f"找到 {len(img_files)} 张图片, {len(xml_files)} 个XML文件.")
    print(f"有效配对样本数: {len(valid_names)}")

    if len(valid_names) == 0:
        print("错误: 未找到匹配的图片和XML文件。请检查 SOURCE_DIR 路径结构。")
        return

    # 预扫描类别
    print("正在预扫描类别...")
    classes = set()
    for name in tqdm(valid_names):
        try:
            tree = ET.parse(xml_map[name])
            for obj in tree.findall('.//object/name'):
                classes.add(obj.text)
        except Exception as e:
            print(f"Error parsing {xml_map[name]}: {e}")

    # 生成类别映射表 (COCO category_id 从 1 开始)
    sorted_classes = sorted(list(classes))
    category_map = {name: i+1 for i, name in enumerate(sorted_classes)}
    print(f"检测到类别 ({len(category_map)}类): {category_map}")

    # 划分数据集
    random.seed(42)
    random.shuffle(valid_names)
    split_idx = int(len(valid_names) * TRAIN_RATIO)
    train_names = valid_names[:split_idx]
    val_names = valid_names[split_idx:]

    split_manifest = {
        "dataset": "Coal mine underground drilling site object detection dataset",
        "source_doi": "10.57760/sciencedb.j00001.01020",
        "source_version": "V1",
        "seed": 42,
        "train_ratio": TRAIN_RATIO,
        "paired_samples": len(valid_names),
        "train_samples": len(train_names),
        "val_samples": len(val_names),
        "classes": sorted_classes,
        "train_ids": train_names,
        "val_ids": val_names,
    }
    canonical_split = json.dumps(
        {"train_ids": train_names, "val_ids": val_names},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    split_manifest["split_sha256"] = hashlib.sha256(canonical_split).hexdigest()

    # 准备路径列表
    train_imgs = [img_map[n] for n in train_names]
    val_imgs = [img_map[n] for n in val_names]

    # 执行转换
    create_coco_structure(OUTPUT_DIR)
    convert_to_coco(train_imgs, xml_map, category_map, "train", OUTPUT_DIR)
    convert_to_coco(val_imgs, xml_map, category_map, "val", OUTPUT_DIR)
    manifest_path = Path(OUTPUT_DIR) / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n转换完成！")
    print(f"数据已保存至: {OUTPUT_DIR}")
    print(f"划分清单已保存至: {manifest_path}")
    print("请记得在 rf-detr 训练命令中使用该路径。")

if __name__ == '__main__':
    main()
