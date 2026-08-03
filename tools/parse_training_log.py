#!/usr/bin/env python
"""
解析训练日志, 提取 Fig 7 诊断分析所需的 4 个子图数据

用法:
  python tools/parse_training_log.py --log log.txt --output figures/data/fig7_diagnostic.json
"""

import argparse, json, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True, help='训练日志 log.txt 路径')
parser.add_argument('--output', default='figures/data/fig7_diagnostic.json')
args = parser.parse_args()

print(f"读取训练日志: {args.log}")
with open(args.log, 'r') as f:
    lines = [json.loads(line) for line in f if line.strip()]
print(f"  共 {len(lines)} 个 epoch 记录")

# ── (a) 退化分层 AP (取最后一个 epoch) ──────────────────
last = lines[-1]
buckets_raw = last.get('test_diagnostics_buckets', {})
degradation_stratified = {}
for deg_type in ['brightness_low', 'brightness_mid', 'brightness_high',
                 'contrast_low', 'contrast_mid', 'contrast_high',
                 'blur_sharp', 'blur_mid', 'blur_blur']:
    if deg_type in buckets_raw:
        entry = buckets_raw[deg_type]
        coco = entry.get('coco_eval_bbox', None)
        degradation_stratified[deg_type] = {
            'size': entry.get('size', 0),
            'ap_50_95': coco[0] if coco else None,
            'ap_50': coco[1] if coco else None,
        }

# ── (b) 逐类 AP ─────────────────────────────────────────
per_class = {}
class_map = last.get('test_results_json', {}).get('class_map', [])
for c in class_map:
    per_class[c['class']] = {
        'map_50_95': c['map@50:95'],
        'map_50': c['map@50'],
    }

# ── (c) DQCD 温度分布 (收集所有 epoch) ──────────────────
dqcd_temps = []
for line in lines:
    epoch = line['epoch']
    train_diag = {}
    test_diag = {}
    for k, v in line.items():
        if 'dqcd_temp' in k:
            train_diag[k] = v
        if 'dqcd_dafd_mod' in k:
            train_diag[k] = v
    dqcd_temps.append({'epoch': epoch, 'train_diag': train_diag})

# ── (d) 收敛曲线 (epoch-wise mAP) ────────────────────────
convergence = []
for line in lines:
    coco = line.get('test_coco_eval_bbox', [None]*12)
    ema_coco = line.get('ema_test_coco_eval_bbox', [None]*12)
    convergence.append({
        'epoch': line['epoch'],
        'ap_50_95': coco[0],
        'ap_50': coco[1],
        'ap_s': coco[3] if len(coco) > 3 else None,
        'ema_ap_50_95': ema_coco[0],
        'ema_ap_50': ema_coco[1],
        'lr': line.get('train_lr', None),
        'loss': line.get('test_loss', None),
        'class_error': line.get('test_class_error', None),
    })

# ── 输出 ────────────────────────────────────────────────
result = {
    'description': 'Fig 7 diagnostic data',
    'degradation_stratified': degradation_stratified,
    'per_class': per_class,
    'dqcd_temps': dqcd_temps,
    'convergence': convergence,
}

os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(result, f, indent=2, default=float)

print(f"✓ Fig 7 数据已保存到: {args.output}")
print(f"  退化分层: {len(degradation_stratified)} 个 bucket")
print(f"  逐类 AP: {len(per_class)} 个类")
print(f"  DQCD 温度: {len(dqcd_temps)} 个 epoch")
print(f"  收敛曲线: {len(convergence)} 个 epoch")
