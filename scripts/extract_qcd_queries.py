import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset
from rfdetr.main import populate_args
from rfdetr.models import build_model, build_criterion_and_postprocessors


def build_components(
    checkpoint_path: str,
    dataset_file: str,
    coco_path: str | None,
    dataset_dir: str | None,
    device: str,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", None)

    base_kwargs = {}
    if ckpt_args is not None:
        base_kwargs.update(vars(ckpt_args))

    # 数据集覆盖
    if dataset_file is not None:
        base_kwargs["dataset_file"] = dataset_file
    if coco_path is not None:
        base_kwargs["coco_path"] = coco_path
    if dataset_dir is not None:
        base_kwargs["dataset_dir"] = dataset_dir

    # 为了在 outputs 中拿到 hs，用于 DQCD，可强制开启 use_qcd=True（不影响权重）
    base_kwargs["use_qcd"] = True

    base_kwargs["device"] = device
    base_kwargs["eval"] = True

    args = populate_args(**base_kwargs)

    device_torch = torch.device(device)
    model = build_model(args)
    model.to(device_torch)
    model.eval()

    criterion, _ = build_criterion_and_postprocessors(args)

    # 构建 val 集数据集
    dataset = build_dataset("val", args, resolution=args.resolution)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    return model, criterion, data_loader, args


def extract_queries(
    checkpoint: str,
    dataset_file: str,
    coco_path: str | None,
    dataset_dir: str | None,
    device: str,
    max_images: int,
    output: str,
):
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else device
    print(f"[INFO] 使用设备: {device}")

    model, criterion, data_loader, args = build_components(
        checkpoint_path=checkpoint,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        device=device,
    )

    all_queries = []
    all_labels = []

    num_processed = 0
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.inference_mode():
            outputs = model(samples)

        if "hs" not in outputs:
            print("[WARN] 模型输出中没有 'hs'（decoder queries），请确认 use_qcd=True 是否生效。")
            break

        # Hungarian 匹配：得到每个样本中的前景 query 索引
        indices = criterion.matcher(outputs, targets)  # list of (src_idx, tgt_idx)

        hs = outputs["hs"]  # [num_layers, B, Q, dim]
        last_layer = hs[-1]  # [B, Q, dim]
        B, Q, D = last_layer.shape

        for b in range(B):
            src_idx, _ = indices[b]
            fg_mask = torch.zeros(Q, dtype=torch.bool, device=device)
            fg_mask[src_idx] = True

            q_b = last_layer[b]           # [Q, D]
            y_b = fg_mask.long()          # [Q], 1=fg, 0=bg

            # 随机采样部分 query，防止数量过多
            max_keep = min(Q, 128)
            perm = torch.randperm(Q, device=device)[:max_keep]
            all_queries.append(q_b[perm].detach().cpu())
            all_labels.append(y_b[perm].detach().cpu())

        num_processed += B
        if num_processed >= max_images:
            break

    if not all_queries:
        print("[ERROR] 未收集到任何 query，无法生成 t-SNE 输入。")
        return

    queries = torch.cat(all_queries, dim=0).numpy()  # [N, D]
    labels = torch.cat(all_labels, dim=0).numpy()    # [N]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(out_path, queries=queries, labels=labels)
    print(f"[INFO] 已保存 DQCD query 嵌入到: {out_path.resolve()} (shape={queries.shape})")


def main():
    parser = argparse.ArgumentParser(
        description="从模型中提取 decoder queries 用于 DQCD t-SNE/UMAP 可视化"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练好的模型路径，例如 outputs/.../checkpoint_best_total.pth",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="roboflow",
        choices=["coco", "roboflow"],
        help="数据集类型（与训练时一致），默认 roboflow",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default=None,
        help="COCO 数据集根目录（当 dataset_file=coco 时使用）",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Roboflow/自建数据集根目录（当 dataset_file=roboflow 时使用）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda / cpu / mps",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=64,
        help="最多使用多少张验证图像来采样 queries（默认 64）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagnostics/qcd_queries.npz",
        help="输出 npz 文件路径（包含 queries 和 labels）",
    )
    args = parser.parse_args()

    extract_queries(
        checkpoint=args.checkpoint,
        dataset_file=args.dataset_file,
        coco_path=args.coco_path,
        dataset_dir=args.dataset_dir,
        device=args.device,
        max_images=args.max_images,
        output=args.output,
    )


if __name__ == "__main__":
    main()
