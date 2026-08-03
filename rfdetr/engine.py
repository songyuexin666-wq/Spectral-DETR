# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Spectral-DETR
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
import os
from typing import Iterable
import random

import torch
import torch.nn.functional as F

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.util.diagnostics import DiagnosticsWriter
from rfdetr.util.degradation import compute_degradation_scores, bucketize
from rfdetr.util import box_ops

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable
from rfdetr.util.misc import NestedTensor
import numpy as np

def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


def _per_image_localization_difficulty(target, result):
    """Measure localization difficulty as one minus mean best same-class IoU."""
    target_boxes = target["boxes"]
    if target_boxes.numel() == 0:
        return 0.0, 0.0

    orig_h, orig_w = target["orig_size"].tolist()
    scale = target_boxes.new_tensor([orig_w, orig_h, orig_w, orig_h])
    target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes) * scale
    pred_boxes = result["boxes"]
    pred_labels = result["labels"]
    pred_scores = result["scores"]
    keep = pred_scores >= 0.05
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]

    best_ious = []
    for box, label in zip(target_xyxy, target["labels"]):
        same_class = pred_boxes[pred_labels == label]
        if same_class.numel() == 0:
            best_ious.append(box.new_tensor(0.0))
            continue
        ious, _ = box_ops.box_iou(box.unsqueeze(0), same_class)
        best_ious.append(ious.max())
    best_ious = torch.stack(best_ious)
    return (
        float(1.0 - best_ious.mean().item()),
        float((best_ious < 0.5).float().mean().item()),
    )


def _float_output_tensors(value):
    """Convert nested floating-point model outputs to float32 for evaluation."""
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _float_output_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_float_output_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_float_output_tensors(item) for item in value)
    return value


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    diag_writer = None
    if args is not None and getattr(args, "diagnostics", False) and utils.is_main_process():
        diag_dir = getattr(args, "diagnostics_dir", None) or os.path.join(args.output_dir, "diagnostics")
        diag_writer = DiagnosticsWriter(diag_dir)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    # 🛠️ 设置当前epoch到criterion/LUE/DAGS warm-up策略
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    # Sync epoch to model for DAGS Top-K warmup
    _m = model.module if hasattr(model, 'module') else model
    if hasattr(_m, 'set_dags_epoch'):
        _m.set_dags_epoch(epoch)

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    print("LENGTH OF DATA LOADER:", len(data_loader))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
            random.seed(it)
            scale = random.choice(scales)
            with torch.inference_mode():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode='bilinear', align_corners=False)
                samples.mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode='nearest').squeeze(1).bool()

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )


            scaler.scale(losses).backward()

        # reduce losses over all GPUs for logging purposes
        # 使用 .detach() 避免在 backward 后访问已释放的 tensor，防止梯度累积时报错
        loss_dict_detached = {k: v.detach() if isinstance(v, torch.Tensor) and v.requires_grad else v for k, v in loss_dict.items()}
        loss_dict_reduced = utils.reduce_dict(loss_dict_detached)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if diag_writer is not None:
            interval = getattr(args, "diagnostics_interval", 200)
            if (it % interval) == 0:
                diag = {"step": it, "epoch": epoch}
                for k, v in loss_dict_reduced.items():
                    if k.startswith("diag/"):
                        diag[k] = v

                try:
                    target_model = model.module if hasattr(model, "module") else model
                    projector = target_model.backbone[0].projector
                    dafd_stats_per_layer = projector.get_dafd_gate_stats()
                    if dafd_stats_per_layer is not None and len(dafd_stats_per_layer) > 0:
                        for band_key in dafd_stats_per_layer[0]:
                            vals = [s[band_key]['mean'] for s in dafd_stats_per_layer if band_key in s]
                            if vals:
                                diag[f"diag/dafd_{band_key}_mean"] = sum(vals) / len(vals)
                                metric_logger.update(**{f"dafd_{band_key}_mean": sum(vals) / len(vals)})
                            vals_std = [s[band_key]['std'] for s in dafd_stats_per_layer if band_key in s]
                            if vals_std:
                                diag[f"diag/dafd_{band_key}_std"] = sum(vals_std) / len(vals_std)
                            vals_sp = [s[band_key]['sparsity'] for s in dafd_stats_per_layer if band_key in s]
                            if vals_sp:
                                diag[f"diag/dafd_{band_key}_sparsity"] = sum(vals_sp) / len(vals_sp)
                                metric_logger.update(**{f"dafd_{band_key}_sparsity": sum(vals_sp) / len(vals_sp)})
                except Exception:
                    pass

                diag_writer.write_scalars(diag)

                max_images = getattr(args, "diagnostics_max_images", 0)
                if max_images > 0 and data_iter_step == 0:
                    try:
                        gate = projector.get_dafd_gate_visual()
                        if gate is not None:
                            gate_img = gate[0, 0].detach().cpu().numpy()
                            diag_writer.save_grayscale(f"dafd_gate_epoch{epoch}", gate_img)
                    except Exception:
                        pass

                # LUE/DQCD plots
                try:
                    payload = criterion.pop_diagnostics_payload()
                    if "lue_err" in payload and "lue_logvar" in payload:
                        diag_writer.save_scatter(
                            f"lue_unc_vs_err_step{it}",
                            payload["lue_err"],
                            payload["lue_logvar"],
                            xlabel="mean_abs_error(cxywh)",
                            ylabel="mean_log_var",
                        )
                    if "dqcd_pos_sim" in payload and "dqcd_neg_sim" in payload:
                        diag_writer.save_hist_2(
                            f"dqcd_sim_hist_step{it}",
                            payload["dqcd_pos_sim"],
                            payload["dqcd_neg_sim"],
                            label_a="pos-pos",
                            label_b="pos-neg",
                            xlabel="cosine similarity",
                        )
                except Exception:
                    pass
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def coco_extended_metrics(coco_eval):
    """
    Safe version: ignores the –1 sentinel entries so precision/F1 never explode.
    """

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    iou50_idx, area_idx, maxdet_idx = (
        int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, 2)

    P = coco_eval.eval["precision"]
    S = coco_eval.eval["scores"]

    prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]

    prec = prec_raw.copy().astype(float)
    prec[prec < 0] = np.nan

    f1_cls   = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
    f1_macro = np.nanmean(f1_cls, axis=1)

    best_j   = int(f1_macro.argmax())

    macro_precision = float(np.nanmean(prec[best_j]))
    macro_recall    = float(rec_thrs[best_j])
    macro_f1        = float(f1_macro[best_j])

    score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
    score_vec[prec_raw[best_j] < 0] = np.nan
    score_thr = float(np.nanmean(score_vec))

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_ids = coco_eval.params.catIds
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        p_slice = P[:, :, k, area_idx, maxdet_idx]
        valid   = p_slice > -1
        ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
        ap_50    = float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean()) if (p_slice[iou50_idx] > -1).any() else float("nan")

        pc = float(prec[best_j, k]) if prec_raw[best_j, k] > -1 else float("nan")
        rc = macro_recall

        #Doing to this to filter out dataset class
        if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
            continue

        per_class.append({
            "class"      : cat_id_to_name[int(cid)],
            "map@50:95"  : ap_50_95,
            "map@50"     : ap_50,
            "precision"  : pc,
            "recall"     : rc,
        })

    per_class.append({
        "class"     : "all",
        "map@50:95" : map_50_95,
        "map@50"    : map_50,
        "precision" : macro_precision,
        "recall"    : macro_recall,
    })

    return {
        "class_map": per_class,
        "map"      : map_50,
        "precision": macro_precision,
        "recall"   : macro_recall
    }

def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    diag_writer = None
    if args is not None and getattr(args, "diagnostics", False) and utils.is_main_process():
        diag_dir = getattr(args, "diagnostics_dir", None) or os.path.join(args.output_dir, "diagnostics")
        diag_writer = DiagnosticsWriter(diag_dir)

    bucket_evaluators = {}
    bucket_sizes = {}
    sample_diagnostics = []
    record_samples = bool(
        args is not None and getattr(args, "diagnostics_sample_records", False)
    )
    if args is not None and getattr(args, "diagnostics_buckets", None) and base_ds is not None:
        bucket_cfg = args.diagnostics_buckets
        brightness_thr = bucket_cfg.get("brightness", [0.3, 0.6])
        contrast_thr = bucket_cfg.get("contrast", [0.08, 0.16])
        blur_thr = bucket_cfg.get("blur", [0.001, 0.004])
        brightness_labels = ["low", "mid", "high"]
        contrast_labels = ["low", "mid", "high"]
        blur_labels = ["sharp", "mid", "blur"]
        for label in brightness_labels:
            bucket_evaluators[f"brightness_{label}"] = CocoEvaluator(base_ds, iou_types)
            bucket_sizes[f"brightness_{label}"] = 0
        for label in contrast_labels:
            bucket_evaluators[f"contrast_{label}"] = CocoEvaluator(base_ds, iou_types)
            bucket_sizes[f"contrast_{label}"] = 0
        for label in blur_labels:
            bucket_evaluators[f"blur_{label}"] = CocoEvaluator(base_ds, iou_types)
            bucket_sizes[f"blur_{label}"] = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            outputs = _float_output_tensors(outputs)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)

        if record_samples:
            degradation_scores = compute_degradation_scores(samples.tensors)
            gate_means = None
            band_gates = outputs.get("dafd_band_gates")
            if band_gates:
                gate_means = torch.stack([
                    gate.detach().float().mean(dim=(1, 2, 3))
                    for gate in band_gates
                ]).mean(dim=0)
            for index, (target, result) in enumerate(zip(targets, results_all)):
                localization_error, miss_rate = _per_image_localization_difficulty(
                    target, result
                )
                record = {
                    "image_id": int(target["image_id"].item()),
                    "brightness": float(degradation_scores["brightness"][index]),
                    "contrast": float(degradation_scores["contrast"][index]),
                    "blur": float(degradation_scores["blur"][index]),
                    "localization_error": localization_error,
                    "miss_rate_iou50": miss_rate,
                    "gate_mean": (
                        float(gate_means[index]) if gate_means is not None else None
                    ),
                    "mean_uncertainty": (
                        float(result["uncertainty"].float().mean())
                        if result.get("uncertainty") is not None
                        and result["uncertainty"].numel() > 0
                        else None
                    ),
                }
                if "controlled_degradation_type" in target:
                    record["controlled_degradation_type"] = int(
                        target["controlled_degradation_type"].item()
                    )
                if "controlled_degradation_severity" in target:
                    record["controlled_degradation_severity"] = float(
                        target["controlled_degradation_severity"].item()
                    )
                sample_diagnostics.append(record)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results_all)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if bucket_evaluators:
            scores = compute_degradation_scores(samples.tensors)
            brightness_bucket = bucketize(scores["brightness"], brightness_thr, brightness_labels)
            contrast_bucket = bucketize(scores["contrast"], contrast_thr, contrast_labels)
            blur_bucket = bucketize(scores["blur"], blur_thr, blur_labels)

            for (target, output), b, c, bl in zip(zip(targets, results_all), brightness_bucket, contrast_bucket, blur_bucket):
                img_id = target["image_id"].item()
                single_res = {img_id: output}
                bucket_evaluators[f"brightness_{b}"].update(single_res)
                bucket_evaluators[f"contrast_{c}"].update(single_res)
                bucket_evaluators[f"blur_{bl}"].update(single_res)
                bucket_sizes[f"brightness_{b}"] += 1
                bucket_sizes[f"contrast_{c}"] += 1
                bucket_sizes[f"blur_{bl}"] += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if record_samples:
        gathered_records = utils.all_gather(sample_diagnostics)
        stats["diagnostics_sample_records"] = [
            record for rank_records in gathered_records for record in rank_records
        ]
    if bucket_evaluators:
        bucket_results = {}
        for name, evaluator in bucket_evaluators.items():
            size = int(bucket_sizes.get(name, 0))
            if size <= 0:
                # No samples for this bucket -> skip COCO accumulator (it would crash on empty eval_imgs)
                bucket_results[name] = {"size": 0, "coco_eval_bbox": None}
                continue

            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()
            bucket_results[name] = {
                "size": size,
                "coco_eval_bbox": evaluator.coco_eval["bbox"].stats.tolist() if "bbox" in iou_types else None,
            }
        stats["diagnostics_buckets"] = bucket_results
        if diag_writer is not None:
            diag_writer.write_json("degradation_buckets", bucket_results)
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            results_json = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    if diag_writer is not None:
        try:
            payload = criterion.pop_diagnostics_payload()
            if "lue_err" in payload and "lue_logvar" in payload:
                diag_writer.save_scatter(
                    "lue_unc_vs_err_eval",
                    payload["lue_err"],
                    payload["lue_logvar"],
                    xlabel="mean_abs_error(cxywh)",
                    ylabel="mean_log_var",
                )
            if "dqcd_pos_sim" in payload and "dqcd_neg_sim" in payload:
                diag_writer.save_hist_2(
                    "dqcd_sim_hist_eval",
                    payload["dqcd_pos_sim"],
                    payload["dqcd_neg_sim"],
                    label_a="pos-pos",
                    label_b="pos-neg",
                    xlabel="cosine similarity",
                )
        except Exception:
            pass
    return stats, coco_evaluator
