# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoConfig, AutoBackbone
try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

from rfdetr.util.misc import NestedTensor, is_main_process

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.degradation_estimator import DegradationEstimator

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""
    def __init__(self,
                 name: str,
                 pretrained_encoder: str=None,
                 window_block_indexes: list=None,
                 drop_path=0.0,
                 out_channels=256,
                 out_feature_indexes: list=None,
                 projector_scale: list=None,
                 use_cls_token: bool = False,
                 freeze_encoder: bool = False,
                 layer_norm: bool = False,
                 target_shape: tuple[int, int] = (640, 640),
                 rms_norm: bool = False,
                 backbone_lora: bool = False,
                 gradient_checkpointing: bool = False,
                 load_dinov2_weights: bool = True,
                 patch_size: int = 14,
                 num_windows: int = 4,
                 positional_encoding_size: bool = False,
                 use_dafd: bool = False,  # 🚀 DAFD: 是否使用退化感知频域分解
                 dafd_sparsity_weight: float = 0.0,
                 dafd_alpha: float = 0.15,
                 dafd_n_bands: int = 3,
                 use_degradation_estimator: bool = False,  # 🚀 v6: 共享退化估计器
                 **_unused_kwargs,
                 ):
        super().__init__()
        # an example name here would be "dinov2_base" or "dinov2_registers_windowed_base"
        # if "registers" is in the name, then use_registers is set to True, otherwise it is set to False
        # similarly, if "windowed" is in the name, then use_windowed_attn is set to True, otherwise it is set to False
        # the last part of the name should be the size
        # and the start should be dinov2
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        size = name_parts[-1]
        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert len(name_parts) == 2, "name should be dinov2, then either registers, windowed, both, or none, then the size"
        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
        )
        # build encoder + projector as backbone module
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "only support projector scale P3/P4/P5/P6 in ascending order."
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            use_dafd=use_dafd,  # 🚀 DAFD: 传递DAFD配置
            dafd_sparsity_weight=dafd_sparsity_weight,
            dafd_alpha=dafd_alpha,
            dafd_n_bands=dafd_n_bands,
        )

        # 🚀 v6.0: 共享退化估计器 (DAFD/DQCD/LUE 的统一退化表征)
        self.use_degradation_estimator = use_degradation_estimator
        if use_degradation_estimator:
            self.degradation_estimator = DegradationEstimator()
        else:
            self.degradation_estimator = None

        self._last_deg_global = None   # (B, 4) 全局退化嵌入
        self._last_spatial_prior = None  # (B, 1, H_s, W_s) 空间先验图

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            print("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        """ """
        # 🚀 v6.0: 共享退化估计 — 在 encoder 之前从原始图像提取退化表征
        if self.use_degradation_estimator and self.degradation_estimator is not None:
            # DegradationEstimator 期望 [0,1] 范围的原始图像 (不做 normalization)
            # 但 tensor_list.tensors 已经被 ImageNet-normalized。我们反向归一化近似还原。
            # 实际上 stem 的 BatchNorm 会自行适应，直接输入 normalized tensor 也能工作。
            deg_global, spatial_prior, _ = self.degradation_estimator(tensor_list.tensors)
            self._last_deg_global = deg_global
            self._last_spatial_prior = spatial_prior
        else:
            self._last_deg_global = None
            self._last_spatial_prior = None

        # (H, W, B, C)
        feats = self.encoder(tensor_list.tensors)

        # 🚀 v6.0: 将退化全局嵌入传递给 projector (DAFD 的 FiLM 使用)
        if self.use_degradation_estimator and self._last_deg_global is not None:
            if hasattr(self.projector, 'set_deg_global'):
                self.projector.set_deg_global(self._last_deg_global)

        feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[
                0
            ]
            out.append(NestedTensor(feat, mask))
        return out

    def get_degradation_outputs(self):
        """返回最近的退化估计结果，供下游模块 (transformer / loss) 使用。"""
        return self._last_deg_global, self._last_spatial_prior

    def forward_export(self, tensors: torch.Tensor):
        if self.use_degradation_estimator and self.degradation_estimator is not None:
            deg_global, spatial_prior, _ = self.degradation_estimator(tensors)
            self._last_deg_global = deg_global
            self._last_spatial_prior = spatial_prior
            if hasattr(self.projector, 'set_deg_global'):
                self.projector.set_deg_global(deg_global)
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)

def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate
