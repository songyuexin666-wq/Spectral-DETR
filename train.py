import os
import sys
import torch

print("=" * 80)
print("开始初始化模型...")
print("=" * 80)

# 检查 CUDA 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from rfdetr import RFDETRBase

print("正在创建 RFDETRBase 模型（这可能需要几分钟下载预训练权重）...")
model = RFDETRBase()
print("模型创建完成！")

print("\n" + "=" * 80)
print("检查数据集路径...")
print("=" * 80)

dataset_dir = "datasets"
if not os.path.exists(dataset_dir):
    print(f"错误：数据集目录 '{dataset_dir}' 不存在！")
    print("请确保数据集目录结构如下：")
    print("datasets/")
    print("  ├── train/")
    print("  │   ├── _annotations.coco.json")
    print("  │   └── *.jpg")
    print("  └── valid/")
    print("      ├── _annotations.coco.json")
    print("      └── *.jpg")
    sys.exit(1)

train_path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
valid_path = os.path.join(dataset_dir, "valid", "_annotations.coco.json")

if not os.path.exists(train_path):
    print(f"错误：训练集标注文件不存在: {train_path}")
    sys.exit(1)

if not os.path.exists(valid_path):
    print(f"警告：验证集标注文件不存在: {valid_path}")
    print("尝试查找 val 目录...")
    valid_path = os.path.join(dataset_dir, "val", "_annotations.coco.json")
    if not os.path.exists(valid_path):
        print(f"错误：验证集标注文件不存在: {valid_path}")
        sys.exit(1)

print(f"✓ 数据集路径检查通过")
print(f"  - 训练集: {train_path}")
print(f"  - 验证集: {valid_path}")

# 检查类别数
print("\n" + "=" * 80)
print("检查类别信息...")
print("=" * 80)

import json
with open(train_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

categories = train_data.get('categories', [])
num_classes = len(categories)
print(f"类别数量: {num_classes}")

if num_classes == 0:
    print("❌ 错误：类别数量为 0！")
    sys.exit(1)

# 检查 category_id 范围
annotations = train_data.get('annotations', [])
if len(annotations) > 0:
    category_ids_in_anns = {ann.get('category_id') for ann in annotations}
    category_ids_in_cats = {cat['id'] for cat in categories}

    invalid_ids = category_ids_in_anns - category_ids_in_cats
    if invalid_ids:
        print(f"❌ 错误：发现无效的 category_id: {invalid_ids}")
        print(f"  有效的 category_id 应该是: {sorted(category_ids_in_cats)}")
        print(f"  请运行: python check_category_ids.py {train_path}")
        sys.exit(1)

    if 0 in category_ids_in_anns:
        print(f"⚠️  警告：发现 category_id = 0")
        print(f"  COCO 格式中类别 ID 应该从 1 开始")
        print(f"  请运行转换脚本修复: python convert_to_coco.py {train_path}")
        sys.exit(1)

    print(f"✓ 类别 ID 检查通过")
    print(f"  类别 ID 范围: {min(category_ids_in_anns)} - {max(category_ids_in_anns)}")
    print(f"  类别数量: {num_classes}")

print("\n" + "=" * 80)
print("开始训练...")
print("=" * 80)

try:
    model.train(
        dataset_dir=dataset_dir,
        dataset_file="roboflow",
        num_classes=num_classes,  # 自动设置类别数
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        device=device,
        num_workers=4,
        use_ema=True,
        tensorboard=True,
    )
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
except Exception as e:
    print(f"\n错误：训练过程中出现异常: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)