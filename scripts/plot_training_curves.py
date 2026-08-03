#!/usr/bin/env python3
"""
Legacy training-curve visualization helper.
Training Dynamics and Convergence Analysis

从 diagnostics/metrics.jsonl 或 TensorBoard 日志读取训练数据，生成美化的训练曲线图

This script is retained for exploratory analysis. It is not the source of the
final Journal of Imaging manuscript tables or figures.

输出: figures/training_dynamics.png
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import argparse
import numpy as np
from collections import defaultdict

# 设置matplotlib样式
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8-whitegrid')


def load_metrics_from_jsonl(jsonl_path):
    """从metrics.jsonl文件加载训练指标"""
    metrics = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return metrics


def smooth_curve(data, weight=0.85):
    """指数移动平均平滑曲线"""
    if len(data) == 0:
        return []
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_curves(
    baseline_dir=None,
    full_dir=None,
    output_path='figures/training_dynamics.png',
    smooth_weight=0.85
):
    """
    生成训练曲线对比图

    Args:
        baseline_dir: Baseline实验输出目录（包含diagnostics/metrics.jsonl）
        full_dir: Full模型实验输出目录
        output_path: 输出图片路径
        smooth_weight: 平滑系数 [0-1]，越大越平滑
    """
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建子图（2行2列）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Dynamics and Convergence Analysis',
                 fontsize=14, fontweight='bold', y=0.995)

    experiments = []
    if baseline_dir and Path(baseline_dir).exists():
        experiments.append(('Baseline', baseline_dir, '#E63946'))
    if full_dir and Path(full_dir).exists():
        experiments.append(('Spectral-DETR (Full)', full_dir, '#06A77D'))

    if not experiments:
        print("⚠️  警告: 未找到有效的实验目录")
        print(f"   Baseline目录: {baseline_dir}")
        print(f"   Full目录: {full_dir}")
        return

    # 为每个实验加载数据
    all_data = {}
    for exp_name, exp_dir, color in experiments:
        metrics_path = Path(exp_dir) / 'diagnostics' / 'metrics.jsonl'

        if not metrics_path.exists():
            print(f"⚠️  警告: 未找到 {exp_name} 的metrics.jsonl: {metrics_path}")
            continue

        metrics = load_metrics_from_jsonl(metrics_path)

        if not metrics:
            print(f"⚠️  警告: {exp_name} 的metrics.jsonl为空")
            continue

        # 提取各种Loss
        data = {
            'steps': [],
            'total_loss': [],
            'bbox_loss': [],
            'giou_loss': [],
            'class_loss': []
        }

        for m in metrics:
            if 'step' in m:
                data['steps'].append(m['step'])
                data['total_loss'].append(m.get('loss_unscaled', 0))
                data['bbox_loss'].append(m.get('loss_bbox_unscaled', 0))
                data['giou_loss'].append(m.get('loss_giou_unscaled', 0))
                data['class_loss'].append(m.get('loss_ce_unscaled', 0))

        # 平滑曲线
        for key in ['total_loss', 'bbox_loss', 'giou_loss', 'class_loss']:
            data[key] = smooth_curve(data[key], smooth_weight)

        all_data[exp_name] = (data, color)

    if not all_data:
        print("❌ 错误: 没有有效的训练数据")
        return

    # 绘制各种Loss曲线
    loss_configs = [
        ('total_loss', 'Total Loss', axes[0, 0]),
        ('bbox_loss', 'BBox Regression Loss', axes[0, 1]),
        ('giou_loss', 'GIoU Loss', axes[1, 0]),
        ('class_loss', 'Classification Loss', axes[1, 1])
    ]

    for loss_key, loss_title, ax in loss_configs:
        for exp_name, (data, color) in all_data.items():
            ax.plot(data['steps'], data[loss_key],
                   label=exp_name, color=color, linewidth=1.5, alpha=0.9)

        ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
        ax.set_ylabel(loss_title, fontsize=10, fontweight='bold')
        ax.set_title(loss_title, fontsize=11, fontweight='bold', pad=10)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 训练曲线图已保存: {output_path}")

    # 打印收敛统计
    print("\n" + "="*80)
    print("训练收敛统计")
    print("="*80)
    for exp_name, (data, _) in all_data.items():
        print(f"\n{exp_name}:")
        print(f"  训练步数: {len(data['steps'])}")
        if data['total_loss']:
            initial_loss = data['total_loss'][0]
            final_loss = data['total_loss'][-1]
            print(f"  初始Total Loss: {initial_loss:.4f}")
            print(f"  最终Total Loss: {final_loss:.4f}")
            print(f"  Loss下降: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
    print("="*80 + "\n")


def plot_diagnostics_curves(
    full_dir=None,
    output_path='figures/diagnostics_curves.png'
):
    """
    生成诊断曲线图（DAFD/DQCD/LUE的诊断指标）

    Args:
        full_dir: Full模型实验输出目录
        output_path: 输出图片路径
    """
    if not full_dir or not Path(full_dir).exists():
        print(f"⚠️  未找到实验目录: {full_dir}")
        return

    metrics_path = Path(full_dir) / 'diagnostics' / 'metrics.jsonl'
    if not metrics_path.exists():
        print(f"⚠️  未找到metrics.jsonl: {metrics_path}")
        return

    metrics = load_metrics_from_jsonl(metrics_path)
    if not metrics:
        print("⚠️  metrics.jsonl为空")
        return

    # 提取诊断数据
    diag_data = defaultdict(list)
    steps = []

    for m in metrics:
        if 'step' not in m:
            continue

        steps.append(m['step'])

        # DAFD诊断
        diag_data['fafd_gate_mean'].append(m.get('diag/fafd_gate_mean', None))
        diag_data['fafd_gate_sparsity'].append(m.get('diag/fafd_gate_sparsity', None))

    # 过滤None值
    for key in diag_data:
        diag_data[key] = [v if v is not None else np.nan for v in diag_data[key]]

    # 检查是否有有效数据
    has_data = any(not all(np.isnan(v) for v in vals) for vals in diag_data.values())

    if not has_data:
        print("⚠️  未找到诊断数据（可能训练时未开启diagnostics）")
        return

    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Module Diagnostics', fontsize=14, fontweight='bold')

    # DAFD Gate Mean
    if not all(np.isnan(diag_data['fafd_gate_mean'])):
        axes[0].plot(steps, diag_data['fafd_gate_mean'],
                    color='#2E86AB', linewidth=1.5)
        axes[0].set_title('DAFD Gate Mean', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Training Steps', fontsize=10)
        axes[0].set_ylabel('Gate Value', fontsize=10)
        axes[0].grid(True, alpha=0.3)

    # DAFD Gate Sparsity
    if not all(np.isnan(diag_data['fafd_gate_sparsity'])):
        axes[1].plot(steps, diag_data['fafd_gate_sparsity'],
                    color='#F18F01', linewidth=1.5)
        axes[1].set_title('DAFD Gate Sparsity', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Training Steps', fontsize=10)
        axes[1].set_ylabel('Sparsity', fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 诊断曲线图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='生成训练曲线图 (Fig. 5)')
    parser.add_argument('--baseline-dir', type=str,
                       help='Baseline实验输出目录（包含diagnostics/metrics.jsonl）')
    parser.add_argument('--full-dir', type=str,
                       help='Full模型实验输出目录')
    parser.add_argument('--output', type=str,
                       default='figures/training_dynamics.png',
                       help='输出图片路径')
    parser.add_argument('--smooth', type=float, default=0.85,
                       help='曲线平滑系数 [0-1]')
    parser.add_argument('--diagnostics', action='store_true',
                       help='同时生成诊断曲线图')

    args = parser.parse_args()

    # 生成训练曲线
    plot_training_curves(
        baseline_dir=args.baseline_dir,
        full_dir=args.full_dir,
        output_path=args.output,
        smooth_weight=args.smooth
    )

    # 生成诊断曲线（如果需要）
    if args.diagnostics and args.full_dir:
        plot_diagnostics_curves(
            full_dir=args.full_dir,
            output_path='figures/diagnostics_curves.png'
        )


if __name__ == '__main__':
    main()
