import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate
from rfdetr.main import populate_args
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.checkpoint import filter_state_dict_by_shape


def build_args_from_checkpoint(
    ckpt_args,
    checkpoint_path: str,
    dataset_file: Optional[str],
    coco_path: Optional[str],
    dataset_dir: Optional[str],
    device: str,
    output_dir: Optional[str],
):
    """
    Merge checkpoint training args with CLI overrides and return a fresh args Namespace.
    """
    base_kwargs = {}
    if ckpt_args is not None:
        # convert Namespace to dict
        base_kwargs.update(vars(ckpt_args))

    # 数据集类型：优先使用 CLI；否则使用 checkpoint 中的配置
    if dataset_file is not None:
        base_kwargs["dataset_file"] = dataset_file
    elif "dataset_file" not in base_kwargs:
        base_kwargs["dataset_file"] = "coco"
    if coco_path is not None:
        base_kwargs["coco_path"] = coco_path
    if dataset_dir is not None:
        base_kwargs["dataset_dir"] = dataset_dir

    base_kwargs["device"] = device

    # default output dir: same folder as checkpoint
    if output_dir is not None:
        base_kwargs["output_dir"] = output_dir
    else:
        base_kwargs.setdefault(
            "output_dir", str(Path(checkpoint_path).resolve().parent)
        )

    # we are in pure eval mode here
    base_kwargs["eval"] = True

    args = populate_args(**base_kwargs)
    return args


def run_test(
    checkpoint_path: str,
    dataset_file: Optional[str],
    coco_path: Optional[str],
    dataset_dir: Optional[str],
    device: str,
    output_dir: Optional[str],
):
    print("=" * 80)
    print(f"加载模型权重: {checkpoint_path}")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", None)

    args = build_args_from_checkpoint(
        ckpt_args=ckpt_args,
        checkpoint_path=checkpoint_path,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        device=device,
        output_dir=output_dir,
    )

    # 关键：必须与 checkpoint 中保存的类别数一致，否则分类头会被丢弃导致结果极低
    ckpt_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    if getattr(args, "num_classes", None) is None or (args.num_classes + 1) != ckpt_num_classes:
        args.num_classes = ckpt_num_classes - 1
        print(f"  - 已根据 checkpoint 设置 num_classes = {args.num_classes} (模型输出维度 {ckpt_num_classes} = num_classes+1)")

    print("使用的数据集配置:")
    print(f"  - dataset_file: {args.dataset_file}")
    print(f"  - num_classes:  {args.num_classes} (与 checkpoint 一致)")
    if getattr(args, "coco_path", None) is not None:
        print(f"  - coco_path:    {args.coco_path}")
    if getattr(args, "dataset_dir", None) is not None:
        print(f"  - dataset_dir:  {args.dataset_dir}")
    print(f"  - device:       {args.device}")
    print(f"  - output_dir:   {args.output_dir}")

    device_torch = torch.device(args.device)

    print("\n构建模型与损失函数...")
    model = build_model(args)
    model.to(device_torch)
    criterion, postprocess = build_criterion_and_postprocessors(args)

    print("从 checkpoint 加载权重到模型...")
    # 模型已按 checkpoint 的 num_classes 构建，此处应能完整加载（含分类头）
    model_state = model.state_dict()
    ckpt_state, dropped = filter_state_dict_by_shape(model_state, checkpoint["model"])
    if dropped:
        print(f"警告: 仍有 {len(dropped)} 个参数 shape 不一致被跳过（会拉低结果）；只打印前 20 个：")
        for name, ckpt_shape, model_shape in dropped[:20]:
            print(f"  - {name}: ckpt{ckpt_shape} != model{model_shape}")
    else:
        print("  已完整加载 checkpoint（含分类头），与训练时模型一致。")

    model.load_state_dict(ckpt_state, strict=False)
    model.eval()

    print("\n构建 test 数据集与 DataLoader...")
    # 这里强制使用 'test' split
    dataset_test = build_dataset("test", args, resolution=args.resolution)
    sampler_test = SequentialSampler(dataset_test)
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    base_ds_test = get_coco_api_from_dataset(dataset_test)

    print(f"test 样本数: {len(dataset_test)}")

    print("\n开始在 test 上评估...")
    test_stats, coco_evaluator = evaluate(
        model,
        criterion,
        postprocess,
        data_loader_test,
        base_ds_test,
        device_torch,
        args=args,
    )

    print("\nTest 评估结果 (COCO 指标字典):")
    print(test_stats)

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 如果有 coco_evaluator，则保存原始 COCO 评估结果
    if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
        eval_path = output_dir_path / "test_eval.pth"
        torch.save(coco_evaluator.coco_eval["bbox"].eval, eval_path)
        print(f"\n已将 COCO eval 结果保存到: {eval_path}")

    # 保存一个简化版的 JSON 总结
    summary_path = output_dir_path / "test_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(test_stats, f, ensure_ascii=False, indent=2)
    print(f"已将 test 指标保存到: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用已训练的 RF-DETR checkpoint 在 test 集上进行评估（COCO 风格指标）"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练好的模型路径，例如 outputs/.../checkpoint.pth",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="coco",
        choices=["coco", "roboflow"],
        help="数据集类型；默认使用 COCO（会自动加载 datasets/annotations/image_info_test-dev.json）",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default="datasets",
        help="COCO 数据集根目录（包含 annotations/ 和 train/val/test 等目录），默认 'datasets'",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Roboflow/自建 COCO 数据集根目录（包含 train/valid/test 子目录）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="评估设备，如 'cuda'、'cpu' 或 'mps'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估结果输出目录；默认与 checkpoint 同目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(
        checkpoint_path=args.checkpoint,
        dataset_file=args.dataset_file,
        coco_path=args.coco_path,
        dataset_dir=args.dataset_dir,
        device=args.device,
        output_dir=args.output_dir,
    )
