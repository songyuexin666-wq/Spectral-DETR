#!/usr/bin/env python3
"""
Legacy visualization helper for an early FPS/AP trade-off figure.

This script contains historical hard-coded values and is not the authoritative
source for the final Journal of Imaging revision. Use tools/make_tables.py and
the final evaluation JSON files to reproduce the manuscript tables.

输出: figures/fps_ap_tradeoff.pdf  (及 .png)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 1.2

# ---------------------------------------------------------------------------
# 数据（来自论文 Table tab:efficiency）
# ---------------------------------------------------------------------------
models = [
    # name,           FPS,    AP,     Params(M), category,   marker, color
    ("YOLOv8n",       198.5,  0.332,  3.0,  "lightweight",  "^",    "#2196F3"),
    ("Baseline",      45.3,   0.332,  30.0, "ablation",     "s",    "#9E9E9E"),
    ("+DAFD",         42.1,   0.344,  30.0, "ablation",     "s",    "#FF9800"),
    ("+DQCD",          45.3,   0.337,  30.0, "ablation",     "s",    "#8BC34A"),
    ("+LUE",          44.8,   0.341,  30.1, "ablation",     "s",    "#E91E63"),
    ("Spectral-DETR", 40.2,   0.361,  30.1, "ours",         "*",    "#F44336"),
]

# ---------------------------------------------------------------------------
# 创建输出目录
# ---------------------------------------------------------------------------
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

# 背景参考线
ax.axhline(y=0.332, color="#BDBDBD", linestyle="--", linewidth=0.8, alpha=0.7,
           label="Baseline AP")

# 按类别顺序绘制
for name, fps, ap, params, cat, marker, color in models:
    # 点大小与参数量成比例（视觉辅助，适度夸张）
    size = 120 + params * 3.5

    zorder = 5 if cat == "ours" else 3
    edgecolor = "#B71C1C" if cat == "ours" else "#424242"
    lw = 1.8 if cat == "ours" else 0.8

    ax.scatter(fps, ap, s=size, c=color, marker=marker,
               edgecolors=edgecolor, linewidths=lw,
               zorder=zorder, alpha=0.92)

    # 标签偏移（避免重叠）
    dx, dy = 0, 0.0015
    ha = "center"
    va = "bottom"

    if name == "YOLOv8n":
        dx, dy = -8, -0.0048
        va = "top"
    elif name == "Baseline":
        dx, dy = 1.5, 0.0015
        ha = "left"
    elif name == "+DAFD":
        dx, dy = -2, 0.0015
        ha = "right"
    elif name == "+DQCD":
        dx, dy = 1.5, -0.0048
        ha = "left"
        va = "top"
    elif name == "+LUE":
        dx, dy = -2, -0.0048
        ha = "right"
        va = "top"
    elif name == "Spectral-DETR":
        dx, dy = 2, 0.0015
        ha = "left"

    fontweight = "bold" if cat == "ours" else "normal"
    fontsize = 10 if cat == "ours" else 9
    ax.annotate(
        name,
        xy=(fps, ap),
        xytext=(fps + dx, ap + dy),
        fontsize=fontsize,
        fontweight=fontweight,
        color=color if cat != "ours" else "#B71C1C",
        ha=ha, va=va,
        zorder=zorder + 1,
    )

# 绘制从 Baseline 到 Spectral-DETR 的改进箭头
ax.annotate(
    "",
    xy=(40.2, 0.361),          # 箭头终点（Spectral-DETR）
    xytext=(45.3, 0.332),      # 箭头起点（Baseline）
    arrowprops=dict(
        arrowstyle="->",
        color="#F44336",
        lw=1.5,
        connectionstyle="arc3,rad=-0.25",
    ),
    zorder=4,
)
ax.text(
    41.5, 0.3465,
    "+2.9% AP",
    fontsize=8.5,
    color="#F44336",
    ha="center",
    style="italic",
    zorder=5,
)

# ---------------------------------------------------------------------------
# 坐标轴设置
# ---------------------------------------------------------------------------
ax.set_xlabel("Inference Speed (FPS)", fontsize=12, fontweight="bold")
ax.set_ylabel("AP@0.5:0.95", fontsize=12, fontweight="bold")
ax.set_title(
    "Speed–Accuracy Trade-off\n(RTX 3090, batch=1, 576×576)",
    fontsize=12, pad=10
)

ax.set_xlim(28, 220)
ax.set_ylim(0.324, 0.372)

# X 轴断轴提示（YOLOv8n 远离其他点，加文字说明）
ax.annotate(
    "← real-time zone",
    xy=(198.5, 0.334),
    fontsize=8, color="#9E9E9E", style="italic", ha="right",
)

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.tick_params(direction="in", length=4)

# ---------------------------------------------------------------------------
# 图例
# ---------------------------------------------------------------------------
legend_elements = [
    mpatches.Patch(facecolor="#F44336", edgecolor="#B71C1C", label="Spectral-DETR (Ours)"),
    mpatches.Patch(facecolor="#9E9E9E", edgecolor="#424242", label="Ablation variants"),
    mpatches.Patch(facecolor="#2196F3", edgecolor="#424242", label="Lightweight baseline"),
    plt.Line2D([0], [0], color="#BDBDBD", linestyle="--", linewidth=0.8,
               label="Baseline AP (0.332)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
          framealpha=0.85, edgecolor="#BDBDBD")

# 气泡大小说明
ax.text(
    0.02, 0.04,
    "Bubble size ∝ Params (M)",
    transform=ax.transAxes,
    fontsize=8, color="#757575", style="italic",
)

# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
plt.tight_layout()

png_path = output_dir / "fps_ap_tradeoff.png"
pdf_path = output_dir / "fps_ap_tradeoff.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

print(f"✅ 已保存: {png_path}")
print(f"✅ 已保存: {pdf_path}")
plt.close()


if __name__ == "__main__":
    pass
