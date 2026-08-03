#!/usr/bin/env python3
# Pure model-forward benchmark for a Spectral-DETR checkpoint.

import argparse
import time
from pathlib import Path

import torch

from rfdetr.models import build_model
from rfdetr.util.checkpoint import filter_state_dict_by_shape


def load_model_for_benchmark(checkpoint_path: str, device: str = "cuda"):
    """
    从训练好的 checkpoint 中还原模型结构，并加载权重，用于 FPS 测试。

    - 使用 checkpoint['args'] 还原训练时的配置
    - 通过 class_embed.bias 反推 num_classes，保证分类头维度一致
    - 严格对齐你在 infer.py 里的建模逻辑
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "args" not in checkpoint:
        raise RuntimeError("checkpoint 中没有保存训练时的 args，无法安全重建模型。")

    args = checkpoint["args"]
    ckpt_model_state = checkpoint["model"]

    # 从分类头维度反推 num_classes（与 infer.py 保持一致）
    if "class_embed.bias" in ckpt_model_state:
        ckpt_num_classes = ckpt_model_state["class_embed.bias"].shape[0]
        args.num_classes = ckpt_num_classes - 1
        print(
            f"[INFO] 从 checkpoint 推断分类头维度为 {ckpt_num_classes}，"
            f"强制将 args.num_classes 设为 {args.num_classes}。"
        )

    # 指定设备
    args.device = device

    # 分辨率：若 args 中没有，则默认 560（与你论文配置一致）
    resolution = getattr(args, "resolution", 560)

    print(f"[INFO] 构建模型：resolution={resolution}, device={device}")
    model = build_model(args)
    model.to(torch.device(device))

    # 过滤掉 shape 不匹配的参数，避免少量 shape 差异导致崩溃
    model_state = model.state_dict()
    ckpt_state, dropped = filter_state_dict_by_shape(model_state, ckpt_model_state)
    if dropped:
        print(f"[WARN] 发现 {len(dropped)} 个 shape 不一致的参数被跳过（只显示前 20 个）：")
        for name, ckpt_shape, model_shape in dropped[:20]:
            print(f"  - {name}: ckpt{ckpt_shape} != model{model_shape}")
    else:
        print("[INFO] 成功完整加载 checkpoint 权重。")

    model.load_state_dict(ckpt_state, strict=False)
    model.eval()

    return model, resolution


def benchmark_fps(
    model: torch.nn.Module,
    resolution: int,
    device: str = "cuda",
    warmup_iters: int = 20,
    iters: int = 100,
):
    """
    在 batch=1、固定分辨率下测「纯前向」FPS：
    - 先 warmup warmup_iters 次
    - 再前向 iters 次，取平均时间
    """
    dummy = torch.randn(1, 3, resolution, resolution, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 正式计时
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.time() - t0

    mean = total / iters  # 秒/张（只包含 model forward）
    fps = 1.0 / mean
    return mean, fps


def main():
    parser = argparse.ArgumentParser("RF-DETR FPS benchmark（纯模型前向）")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练好的 checkpoint 路径，例如 outputs/.../checkpoint.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备，默认 cuda",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="warmup 次数，默认 20",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="计时迭代次数，默认 100（越大越稳定）",
    )

    args = parser.parse_args()

    model, resolution = load_model_for_benchmark(args.checkpoint, device=args.device)
    mean, fps = benchmark_fps(
        model=model,
        resolution=resolution,
        device=args.device,
        warmup_iters=args.warmup,
        iters=args.iters,
    )

    print("\n=== Pure forward FPS Benchmark Result ===")
    print(f"Resolution: {resolution}x{resolution}, batch=1")
    print("Note: only pure model forward time is considered;")
    print("      data loading, preprocessing, post-processing, and COCOeval are excluded.")
    print(f"Mean latency (forward only): {mean*1000:.2f} ms / image")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
