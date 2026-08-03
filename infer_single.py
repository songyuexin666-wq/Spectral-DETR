import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

from rfdetr.models import build_model, PostProcess
from rfdetr.util.checkpoint import filter_state_dict_by_shape


def load_model(checkpoint_path: str, device: str = "cuda"):
    """
    从训练好的 checkpoint 构建模型。

    关键点：直接使用 checkpoint 中保存的 args（包括你的数据集类别数），
    不再重新指定 num_classes，保证分类头和你的数据集完全一致。
    """
    ckpt_path = Path(checkpoint_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "args" not in checkpoint:
        raise RuntimeError(
            "checkpoint 中没有保存训练时的 args，无法安全重建模型。"
        )
    args = checkpoint["args"]
    ckpt_model_state = checkpoint["model"]

    # 关键修复：
    # 训练时实际使用的分类头维度以 checkpoint 中的 class_embed.bias 为准，
    # 而不是 args.num_classes。否则会导致推理时重新构建成 15 类、丢弃
    # 91 类分类头权重，从而检测不到目标。
    if "class_embed.bias" in ckpt_model_state:
        ckpt_num_classes = ckpt_model_state["class_embed.bias"].shape[0]
        # 按 LW-DETR 约定：输出维度 = num_classes + 1（多一个 no-object）
        args.num_classes = ckpt_num_classes - 1
        print(
            f"[INFO] 从 checkpoint 推断分类头维度为 {ckpt_num_classes}，"
            f"强制将 args.num_classes 设为 {args.num_classes}，以完全还原训练时模型结构。"
        )

    # 使用当前指定设备
    args.device = device

    # 构建模型（结构与训练时完全一致）
    print(f"[INFO] Building model with num_classes = {getattr(args, 'num_classes', 'UNKNOWN')} (after override from checkpoint)")
    model = build_model(args)
    model.to(torch.device(device))

    # 加载权重前，检查一次分类头维度是否一致
    model_state = model.state_dict()
    if "class_embed.bias" in ckpt_model_state and "class_embed.bias" in model_state:
        ckpt_num_classes = ckpt_model_state["class_embed.bias"].shape[0]
        model_num_classes = model_state["class_embed.bias"].shape[0]
        print(f"[INFO] checkpoint class_embed.bias dim = {ckpt_num_classes}")
        print(f"[INFO] current model class_embed.bias dim = {model_num_classes}")
        if ckpt_num_classes != model_num_classes:
            print(
                "[ERROR] 检测到分类头维度仍然不一致：这说明模型构建与训练时存在根本差异，"
                "推理结果不可靠。"
            )

    # 加载权重（如有其他 shape 不一致则提示）
    ckpt_state, dropped = filter_state_dict_by_shape(model_state, ckpt_model_state)
    if dropped:
        print(f"[WARN] 发现 {len(dropped)} 个 shape 不一致的参数被跳过（只显示前 20 个）：")
        for name, ckpt_shape, model_shape in dropped[:20]:
            print(f"  - {name}: ckpt{ckpt_shape} != model{model_shape}")
    else:
        print("[INFO] 成功完整加载 checkpoint 权重（包括分类头），与训练时模型一致。")

    model.load_state_dict(ckpt_state, strict=False)
    model.eval()

    # 后处理模块
    num_select = getattr(args, "num_select", 100)
    postprocess = PostProcess(num_select=num_select)

    # 类别名称（如果在训练时保存了）
    class_names = getattr(args, "class_names", None)

    return model, postprocess, args, class_names


def preprocess_image(img_path: str, resolution: int, device: str):
    """
    与 RF-DETR 训练/推理一致的预处理：
    - 读取 RGB
    - ToTensor -> Normalize
    - Resize 到 (resolution, resolution)
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    tensor = F.to_tensor(img)  # [0,1], CxHxW
    if (tensor > 1).any():
        raise ValueError("图像张量中存在 >1 的值，请检查是否重复归一化。")
    if tensor.shape[0] != 3:
        raise ValueError(f"期望 3 通道 RGB 图像，但得到 {tensor.shape[0]} 通道。")

    # 与 RF-DETR 一致的归一化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = F.normalize(tensor, mean, std)
    tensor = F.resize(tensor, (resolution, resolution))

    tensor = tensor.to(device)
    return img, tensor.unsqueeze(0), (orig_h, orig_w)


def visualize_detections(
    image: Image.Image,
    boxes,
    scores,
    labels,
    class_names=None,
    score_thresh: float = 0.5,
    map_text: str | None = None,
    max_dets: int = 50,
):
    """在原图上标注目标检测结果（绘制框 + 类别 + 置信度），并可选显示 mAP 文本。

    - 只保留置信度 >= score_thresh 的预测；
    - 最多显示 max_dets 个最高置信度目标；
    - 置信度以 0~1 的浮点数显示（保留两位小数），类别与置信度之间留空格。
    """
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    # 1) 先按置信度阈值过滤
    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # 2) 再取置信度最高的前 max_dets 个
    if len(scores) > max_dets:
        order = scores.argsort()[::-1][:max_dets]
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

    print(f"[INFO] 可视化 {len(boxes)} 个检测结果（score >= {score_thresh:.2f}，最多 {max_dets} 个）")

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # 先绘制 mAP 文本（如果提供）
    if map_text is not None:
        mt = map_text
        try:
            bbox = draw.textbbox((0, 0), mt, font=font)
            mw, mh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                mw, mh = font.getsize(mt)
            except Exception:
                mw, mh = len(mt) * 8, 16
        margin = 5
        draw.rectangle(
            [margin, margin, margin + mw + 2, margin + mh + 2],
            fill="black",
        )
        draw.text((margin + 1, margin + 1), mt, fill="white", font=font)

    # 再绘制每个目标的框 + 类别 + 置信度
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        cls_id = int(label)

        if class_names is not None and 0 <= cls_id < len(class_names):
            cls_name = str(class_names[cls_id])
        else:
            cls_name = f"id:{cls_id}"

        # 类别和置信度分开渲染，使用不同颜色
        label_text = cls_name
        score_text = f"{score:.2f}"

        # 先画检测框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # 文本位置：框中心附近
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # 兼容不同 Pillow 版本的文本尺寸获取（分别测量类别和置信度）
        try:
            bbox_label = draw.textbbox((0, 0), label_text, font=font)
            lw, lh = bbox_label[2] - bbox_label[0], bbox_label[3] - bbox_label[1]
            bbox_score = draw.textbbox((0, 0), score_text, font=font)
            sw, sh = bbox_score[2] - bbox_score[0], bbox_score[3] - bbox_score[1]
        except Exception:
            try:
                lw, lh = font.getsize(label_text)
                sw, sh = font.getsize(score_text)
            except Exception:
                lw, lh = len(label_text) * 8, 16
                sw, sh = len(score_text) * 8, 16

        # 总宽度：类别 + 间隔 + 置信度（适当加大间隙）
        gap = 10
        tw = lw + gap + sw
        th = max(lh, sh)

        # 将文本背景和文字绘制在中心点附近，避免越界
        x_text = max(0, cx - tw / 2)
        y_text = max(0, cy - th / 2)
        # 背景统一用红色
        draw.rectangle([x_text, y_text, x_text + tw, y_text + th], fill="red")
        # 类别文字：白色
        draw.text((x_text, y_text), label_text, fill="white", font=font)
        # 置信度文字：黄色，紧跟在类别后面
        draw.text((x_text + lw + gap, y_text), score_text, fill="yellow", font=font)

    return image


def run_inference(
    checkpoint: str,
    image_path: str,
    device: str = "cuda",
    score_thresh: float = 0.5,
    output_path: str | None = None,
    map_text: str | None = None,
):
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else device
    print(f"[INFO] 使用设备: {device}")

    model, postprocess, args, class_names = load_model(checkpoint, device)
    orig_img, batch_tensor, (orig_h, orig_w) = preprocess_image(
        image_path, args.resolution, device
    )

    with torch.inference_mode():
        outputs = model(batch_tensor)

        # 兼容 (pred_boxes, pred_logits, [pred_masks]) 形式
        if isinstance(outputs, tuple):
            out_dict = {
                "pred_boxes": outputs[0],
                "pred_logits": outputs[1],
            }
            if len(outputs) == 3:
                out_dict["pred_masks"] = outputs[2]
            outputs = out_dict

        target_sizes = torch.tensor([(orig_h, orig_w)], device=device)
        results = postprocess(outputs, target_sizes=target_sizes)[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]
    uncertainties = results.get("uncertainty", None)

    vis_img = visualize_detections(
        orig_img,
        boxes,
        scores,
        labels,
        class_names=class_names,
        score_thresh=score_thresh,
        map_text=map_text,
        max_dets=50,
    )

    # 统一将输出结果保存到 datasets/tuili 目录下
    base_dir = Path("datasets") / "tuili"
    base_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        # 自动从 checkpoint 路径中提取模型标记，避免不同模型覆盖同一张图
        ckpt_path = Path(checkpoint)
        model_dir_name = ckpt_path.parent.name  # e.g. baseline_20260302_...
        # 取下划线前缀作为简短模型名（如 baseline / ours）
        model_tag = model_dir_name.split("_")[0] if "_" in model_dir_name else model_dir_name

        stem = Path(image_path).stem
        suffix = Path(image_path).suffix
        # 文件名形如: <图名>_<模型标记>_detected.<ext>
        filename = f"{stem}_{model_tag}_detected{suffix}"
    else:
        # 如果用户指定了 output，只取文件名部分，仍然放在 datasets/tuili 下
        filename = Path(output_path).name

    out_path = base_dir / filename

    vis_img.save(out_path)
    print(f"[INFO] 检测结果已保存到: {out_path.resolve()}")

    # ===== LUE 不确定性热图（可选）=====
    if uncertainties is not None:
        try:
            import numpy as np
            try:
                import matplotlib.cm as cm
            except Exception:
                cm = None

            # boxes: [K, 4], uncertainties: [K]
            boxes_np = boxes.cpu().numpy()
            unc_np = uncertainties.cpu().numpy()

            h_img, w_img = orig_h, orig_w
            heat = np.zeros((h_img, w_img), dtype=np.float32)
            count = np.zeros((h_img, w_img), dtype=np.float32)

            # 归一化不确定性到 0-1
            u_min, u_max = float(unc_np.min()), float(unc_np.max())
            u_norm = (unc_np - u_min) / (u_max - u_min + 1e-6)

            for u, box in zip(u_norm, boxes_np):
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, min(w_img - 1, x1))
                x2 = max(0, min(w_img, x2))
                y1 = max(0, min(h_img - 1, y1))
                y2 = max(0, min(h_img, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                heat[y1:y2, x1:x2] += u
                count[y1:y2, x1:x2] += 1.0

            mask = count > 0
            if mask.any():
                heat[mask] = heat[mask] / count[mask]
                # 再次归一化到 0-1
                h_min, h_max = float(heat[mask].min()), float(heat[mask].max())
                heat_norm = np.zeros_like(heat)
                heat_norm[mask] = (heat[mask] - h_min) / (h_max - h_min + 1e-6)
            else:
                heat_norm = heat

            # 使用漂亮的 colormap（如果可用）
            if cm is not None:
                cmap = cm.get_cmap("magma")  # 深色背景 + 高不确定性高亮
                color = cmap(heat_norm)  # [H,W,4], 0-1
                color_img = (color[..., :3] * 255).astype(np.uint8)
            else:
                # 退化到简单红色梯度
                color_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
                color_img[..., 0] = (heat_norm * 255).astype(np.uint8)

            from PIL import Image

            heat_img = Image.fromarray(color_img).resize(orig_img.size)
            # 将热力图以半透明方式叠加在原图上
            overlay = Image.blend(orig_img.convert("RGB"), heat_img.convert("RGB"), alpha=0.5)

            unc_filename = f"{stem}_{model_tag}_uncertainty{suffix}"
            unc_path = base_dir / unc_filename
            overlay.save(unc_path)
            print(f"[INFO] LUE 不确定性热图已保存到: {unc_path.resolve()}")
        except Exception as e:
            print(f"[WARN] 生成不确定性热图失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR 单张图片推理脚本（使用训练好的自定义数据集分类头）。"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练好的模型路径，例如 outputs/.../checkpoint.pth",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="待检测图片路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda / cpu / mps",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="置信度阈值，默认 0.5（只显示高置信度预测）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出可视化图像路径；不指定则自动加上模型名标记并保存到 datasets/tuili/",
    )
    parser.add_argument(
        "--map_text",
        type=str,
        default=None,
        help="可选：要显示在图像左上角的 mAP 文本，例如 'mAP@0.5:0.95=0.361, AP@0.5=0.679'",
    )
    args = parser.parse_args()

    run_inference(
        checkpoint=args.checkpoint,
        image_path=args.image,
        device=args.device,
        score_thresh=args.score_thresh,
        output_path=args.output,
        map_text=args.map_text,
    )


if __name__ == "__main__":
    main()
