import os
from pathlib import Path

import matplotlib.pyplot as plt

# Legacy visualization helper. The hard-coded values below are retained for
# exploratory plotting only and are not the authoritative source for the final
# Journal of Imaging manuscript tables.


def main():
    # =========================
    # 1. 数据（来自 Table~\ref{tab:efficiency}）
    # =========================
    models = [
        "YOLOv8n",
        "Baseline",
        "+DAFD",
        "+DQCD (train only)",
        "+LUE",
        "Full (Ours)",
    ]

    fps = [
        198.5,  # YOLOv8n
        45.3,   # Baseline
        42.1,   # +DAFD
        45.3,   # +DQCD
        44.8,   # +LUE
        40.2,   # Full (Ours)
    ]

    ap_5095 = [
        0.332,  # YOLOv8n
        0.332,  # Baseline
        0.344,  # +DAFD
        0.337,  # +DQCD
        0.341,  # +LUE
        0.361,  # Full (Ours)
    ]

    # =========================
    # 2. 画图
    # =========================
    plt.figure(figsize=(6, 4.5))

    # 所有模型先画成蓝色圆点
    plt.scatter(fps, ap_5095, c="tab:blue", s=40)

    # 单独用红色星形标记 Full (Ours)
    full_idx = models.index("Full (Ours)")
    plt.scatter(
        [fps[full_idx]],
        [ap_5095[full_idx]],
        c="tab:red",
        s=60,
        marker="*",
        label="Full (Ours)",
        zorder=5,
    )

    # 对每个点加文字标注，略微偏移，避免覆盖点
    for x, y, name in zip(fps, ap_5095, models):
        plt.text(
            x + 1.0,
            y + 0.0005,
            name,
            fontsize=8,
            va="center",
        )

    plt.xlabel("FPS (higher is better)", fontsize=11)
    plt.ylabel("AP@0.5:0.95 (higher is better)", fontsize=11)
    plt.title("FPS–mAP Trade-off Curve", fontsize=12)

    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # =========================
    # 3. 保存为 figures/fps_map_tradeoff.png
    # =========================
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fps_map_tradeoff.png"

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
