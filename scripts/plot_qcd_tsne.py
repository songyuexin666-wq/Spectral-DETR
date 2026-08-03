import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def main():
    parser = argparse.ArgumentParser(
        description="根据提取好的 decoder queries 生成 DQCD t-SNE 可视化图像"
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="diagnostics/qcd_queries.npz",
        help="extract_qcd_queries 生成的 npz 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/qcd_tsne.png",
        help="输出 t-SNE 图像路径",
    )
    parser.add_argument(
        "--max_bg",
        type=int,
        default=5000,
        help="背景 queries 最多可视化数量（默认 5000），前景全部保留",
    )
    args = parser.parse_args()

    data_path = Path(args.npz)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {data_path}")

    data = np.load(data_path)
    queries = data["queries"]  # [N, D]
    labels = data["labels"]    # [N], 1=前景, 0=背景

    print(f"[INFO] 加载 queries: shape={queries.shape}, labels shape={labels.shape}")

    # ==== 背景下采样：保留全部前景，随机采样固定数量背景 ====
    fg_idx = np.where(labels == 1)[0]
    bg_idx = np.where(labels == 0)[0]
    print(f"[INFO] 前景 queries 数量: {len(fg_idx)}, 背景 queries 数量: {len(bg_idx)}")

    if len(bg_idx) > args.max_bg:
        perm = np.random.RandomState(42).permutation(bg_idx)[: args.max_bg]
        bg_idx_sub = perm
    else:
        bg_idx_sub = bg_idx

    keep_idx = np.concatenate([fg_idx, bg_idx_sub])
    queries = queries[keep_idx]
    labels = labels[keep_idx]

    print(
        f"[INFO] 下采样后总点数: {queries.shape[0]} "
        f"(前景 {len(fg_idx)}, 背景 {len(bg_idx_sub)})"
    )

    # 运行 t-SNE 降维到 2D
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=42,
        perplexity=30,
        learning_rate="auto",
    )
    emb2d = tsne.fit_transform(queries)

    fg = labels == 1
    bg = labels == 0

    plt.figure(figsize=(5, 5))
    # 背景：浅灰
    plt.scatter(
        emb2d[bg, 0],
        emb2d[bg, 1],
        s=6,
        c="lightgray",
        label="Background queries",
        alpha=0.6,
    )
    # 前景：红色
    plt.scatter(
        emb2d[fg, 0],
        emb2d[fg, 1],
        s=8,
        c="tab:red",
        label="Foreground queries",
        alpha=0.8,
    )

    plt.legend(markerscale=2, fontsize=8, loc="best", frameon=True)
    plt.xticks([])
    plt.yticks([])
    plt.title("Decoder Query Manifold (DQCD)", fontsize=12)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] 已保存 DQCD t-SNE 图像到: {out_path.resolve()}")


if __name__ == "__main__":
    main()

