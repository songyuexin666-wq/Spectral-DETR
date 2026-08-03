# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Spectral-DETR
# ------------------------------------------------------------------------

from pydantic import BaseModel
from typing import List, Optional, Literal, Type
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class ModelConfig(BaseModel):
    class Config:
        extra = 'allow'

    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"]
    out_feature_indexes: List[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    positional_encoding_size: int
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    # CVPR v4.0 ablations (optional)
    use_lue: bool = False
    lue_uncertainty_weight: float = 0.5
    lue_warmup_epochs: int = 15  # longer warmup → more stable early CIoU training
    lue_giou_weighting: bool = False  # ⚠️ 必须 False! True 会让 LUE precision 偏向易类，难类 (class_12/13) 直接腰斩
    lue_giou_start_epoch: int = 40
    lue_giou_warmup_epochs: int = 5
    use_dafd: bool = False
    dafd_sparsity_weight: float = 0.0
    dafd_alpha: float = 0.15
    dafd_n_bands: int = 3
    # Encoder taps are zero-based indexes into out_feature_indexes. None applies
    # DAFD to every tap, preserving the behavior of existing checkpoints.
    dafd_feature_indices: Optional[List[int]] = None
    dafd_gate_source_index: Optional[int] = None
    use_dqcd: bool = False
    dqcd_temperature: float = 0.15
    dqcd_weight: float = 0.3
    dqcd_hard_negatives_k: int = 128
    dqcd_gate_mode: Literal["adaptive", "fixed", "shuffled", "random"] = "adaptive"
    dqcd_start_epoch: int = 8
    dqcd_warmup_epochs: int = 0
    dqcd_decay_start_epoch: int = -1
    dqcd_decay_epochs: int = 25
    dqcd_final_weight: float = -1.0
    use_scu: bool = False
    use_degradation_estimator: bool = False  # 🚀 v6: 共享退化估计器
    dags_start_epoch: int = 8
    dags_warmup_epochs: int = 5
    dags_bonus_scale: float = 0.0
    use_soft_nms: bool = False
    soft_nms_sigma: float = 0.5
    soft_nms_iou_threshold: float = 0.5
    scu_salience_weight: float = 0.1
    scu_calib_slope: float = -1.5
    scu_calib_center: float = -4.2


class RFDETRBaseConfig(ModelConfig):
    """
    The configuration for an RF-DETR Base model.
    """
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = "rf-detr-base.pth"
    resolution: int = 560
    positional_encoding_size: int = 37

class RFDETRLargeConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Large model.
    """
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_base"
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: Optional[str] = "rf-detr-large.pth"

class RFDETRNanoConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Nano model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"

class RFDETRSmallConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Small model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = "rf-detr-small.pth"

class RFDETRMediumConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Medium model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"

class RFDETRSegPreviewConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-preview.pt"
    num_classes: int = 90

class TrainConfig(BaseModel):
    class Config:
        extra = 'allow'

    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 100
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = 25
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 5.0   # v5 验证值 (achieves 0.478 mAP@50:95); 7.5 实测退化
    lue_uncertainty_coef: float = 0.5
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 5.0    # v5 验证值, 与 giou_loss_coef 同步
    num_select: int = 300
    dataset_file: Literal["coco", "o365", "roboflow"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str
    coco_path: Optional[str] = None  # COCO格式数据集根目录（包含annotations和train/val文件夹）
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    tensorboard: bool = True
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: List[str] = None
    run_test: bool = True
    segmentation_head: bool = False
    diagnostics: bool = False
    diagnostics_dir: Optional[str] = None
    diagnostics_interval: int = 200
    diagnostics_max_images: int = 4
    diagnostics_benchmark: bool = False
    diagnostics_buckets: Optional[dict] = None
    diagnostics_sample_records: bool = False
    # Runtime / reproducibility (passed through from train_mine.py)
    num_classes: int = 90
    resolution: int = 560
    patch_size: int = 14
    num_windows: int = 4
    seed: int = 42
    device: str = "cuda"
    resume: Optional[str] = None


class SegmentationTrainConfig(TrainConfig):
    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 5.0
    segmentation_head: bool = True
