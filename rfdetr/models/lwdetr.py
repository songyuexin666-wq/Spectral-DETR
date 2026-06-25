# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
LW-DETR model and criterion classes
"""
import copy
import math
from typing import Callable
import torch
import torch.nn.functional as F
from torch import nn

from rfdetr.util import box_ops
from rfdetr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from rfdetr.models.backbone import build_backbone
from rfdetr.models.matcher import build_matcher
from rfdetr.models.transformer import build_transformer
from rfdetr.models.segmentation_head import SegmentationHead, get_uncertain_point_coords_with_randomness, point_sample

class LWDETR(nn.Module):
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self,
                 backbone,
                 transformer,
                 segmentation_head,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False,
                 use_lue=False,
                 use_dqcd=False,
                 use_dafd=False,
                 use_scu=False,
                 dags_start_epoch=8,
                 dags_warmup_epochs=5,
                 dags_bonus_scale=0.0):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        
        # 🚀 LUE: bbox 位置回归仍保持 4 维 (cx,cy,w,h)，避免破坏 two-stage / iterative refine 逻辑。
        # 不确定性(log_var) 使用单独 head 输出 4 维。
        self.use_lue = use_lue
        self.use_dqcd = use_dqcd
        self.use_dafd = use_dafd
        self.use_scu = use_scu
        self.dags_start_epoch = dags_start_epoch
        self.dags_warmup_epochs = dags_warmup_epochs
        self.dags_bonus_scale = dags_bonus_scale
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.bbox_log_var_embed = None
        if self.use_lue:
            self.bbox_log_var_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head
        
        query_dim=4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        # 🛠️ Deep Fix: LUE 方差初始化修复 (解决"不确定性陷阱")
        # 将 log_var head 的 bias 初始化为 -5.0，对应初始方差 σ² = exp(-5.0) ≈ 0.0067
        if self.use_lue and self.bbox_log_var_embed is not None:
            nn.init.constant_(self.bbox_log_var_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_log_var_embed.layers[-1].bias.data, -5.0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._dags_epoch = 0       # updated by engine each epoch for DAGS Top-K warmup
        self._dags_ratio = 0.0
        self._export = False

    def set_dags_epoch(self, epoch: int):
        """Called by engine each epoch; used to warm up DAGS Top-K bonus."""
        self._dags_epoch = epoch
        if self.dags_bonus_scale <= 0 or epoch < self.dags_start_epoch:
            self._dags_ratio = 0.0
        elif self.dags_warmup_epochs > 0:
            self._dags_ratio = min(
                1.0,
                (epoch - self.dags_start_epoch + 1) / self.dags_warmup_epochs,
            )
        else:
            self._dags_ratio = 1.0

    def reinitialize_detection_head(self, num_classes):
        base = self.class_embed.weight.shape[0]
        num_repeats = int(math.ceil(num_classes / base))
        self.class_embed.weight.data = self.class_embed.weight.data.repeat(num_repeats, 1)
        self.class_embed.weight.data = self.class_embed.weight.data[:num_classes]
        self.class_embed.bias.data = self.class_embed.bias.data.repeat(num_repeats)
        self.class_embed.bias.data = self.class_embed.bias.data[:num_classes]
        
        if self.two_stage:
            for enc_out_class_embed in self.transformer.enc_out_class_embed:
                enc_out_class_embed.weight.data = enc_out_class_embed.weight.data.repeat(num_repeats, 1)
                enc_out_class_embed.weight.data = enc_out_class_embed.weight.data[:num_classes]
                enc_out_class_embed.bias.data = enc_out_class_embed.bias.data.repeat(num_repeats)
                enc_out_class_embed.bias.data = enc_out_class_embed.bias.data[:num_classes]

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        # 🚀 v6.0 DAGS: 从 backbone 获取共享退化表征，传给 transformer + 输出
        spatial_prior = None
        deg_global = None
        backbone = self.backbone[0] if hasattr(self.backbone, '__getitem__') else self.backbone
        if hasattr(backbone, 'get_degradation_outputs'):
            deg_global, spatial_prior = backbone.get_degradation_outputs()

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        # DAGS ratio: scales the Top-K bonus during linear warmup
        self.transformer._dags_ratio = getattr(self, '_dags_ratio', 1.0)
        self.transformer._dags_bonus_scale = getattr(self, 'dags_bonus_scale', 0.0)

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight,
            spatial_prior=spatial_prior, deg_global=deg_global)

        if hs is not None:
            # bbox 位置回归 (4维)
            outputs_coord_raw = self.bbox_embed(hs)  # [num_layers, B, N, 4]

            # 🚀 LUE: 不确定性(log_var)单独 head (4维)
            outputs_log_var = None
            if self.use_lue and self.bbox_log_var_embed is not None:
                outputs_log_var = self.bbox_log_var_embed(hs)  # [num_layers, B, N, 4]
                outputs_log_var = torch.clamp(outputs_log_var, min=-7.0, max=7.0)
            
            if self.bbox_reparam:
                outputs_coord_cxcy = outputs_coord_raw[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_raw[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )
            else:
                outputs_coord = (outputs_coord_raw + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(features[0].tensors, hs, samples.tensors.shape[-2:])

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            # 🚀 LUE: 添加不确定性输出
            if self.use_lue and outputs_log_var is not None:
                out['pred_log_vars'] = outputs_log_var[-1]
            if self.use_dqcd:
                out['hs'] = hs
            if self.segmentation_head is not None:
                out['pred_masks'] = outputs_masks[-1]
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(
                    outputs_class, outputs_coord, outputs_masks if self.segmentation_head is not None else None,
                    None  # LUE 只在最后一层计算：早期 decoder 层 bbox 预测不稳定，
                          # 在其上训练 uncertainty head 会引入噪声梯度，延缓收敛。
                )
            if self.use_dafd:
                dafd_loss = getattr(self.backbone[0].projector, "get_dafd_sparsity_loss", lambda: None)()
                if dafd_loss is not None:
                    out["dafd_sparsity_loss"] = dafd_loss
                # 收集 DAFD per-band gate 统计，供 DQCD 使用
                dafd_gates = getattr(self.backbone[0].projector, "get_dafd_gate_visual", lambda: None)()
                if dafd_gates is not None:
                    out["dafd_band_gates"] = dafd_gates

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)

            cls_enc = torch.cat(cls_enc, dim=1)

            if self.segmentation_head is not None:
                masks_enc = self.segmentation_head(features[0].tensors, [hs_enc,], samples.tensors.shape[-2:], skip_blocks=True)
                masks_enc = torch.cat(masks_enc, dim=1)

            if hs is not None:
                out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['enc_outputs']['pred_masks'] = masks_enc
            else:
                out = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['pred_masks'] = masks_enc

        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        # only use one group in inference
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight)

        outputs_masks = None

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs[0], [hs,], tensors.shape[-2:])[0]
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs[0], [hs_enc,], tensors.shape[-2:], skip_blocks=True)[0]

        if outputs_masks is not None:
            return outputs_coord, outputs_class, outputs_masks
        else:
            return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks, outputs_log_var=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # 🚀 LUE: 添加不确定性支持
        if outputs_masks is not None:
            if outputs_log_var is not None:
                return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c, 'pred_log_vars': d}
                        for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1], outputs_log_var[:-1])]
            else:
                return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1])]
        else:
            if outputs_log_var is not None:
                return [{'pred_logits': a, 'pred_boxes': b, 'pred_log_vars': c}
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_log_var[:-1])]
            else:
                return [{'pred_logits': a, 'pred_boxes': b}
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder, 'blocks'): # Not aimv2
                if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]
            else: # aimv2
                if hasattr(self.backbone[0].encoder.trunk.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.trunk.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


class SetCriterion(nn.Module):
    """Spectral-DETR loss criterion — three reliability layers, all independently ablatable.

    Layer ① DAFD (Feature):   freq-domain decomposition → sparsity reg (in projector, not here)
    Layer ② DQCD (Query):     adaptive supervised InfoNCE on decoder queries
    Layer ③ LUE  (Localization & Classification):
              ├── LUE core:  heteroscedastic Laplace NLL + precision-weighted L1
              ├── SCU:       geometric salience calibration (use_scu toggle)
              └── IA-BCE:    IoU-aware classification loss (ia_bce_loss toggle)

    Standard Hungarian matching → per-loss supervision as in Conditional DETR.
    DegradationEstimator auto-enables when DAFD or DQCD is on (shared foundation).
    """
    def __init__(self,
                num_classes,
                matcher,
                weight_dict,
                focal_alpha,
                losses,
                group_detr=1,
                sum_group_losses=False,
                use_varifocal_loss=False,
                use_position_supervised_loss=False,
                ia_bce_loss=False,
                mask_point_sample_ratio: int = 16,
                use_lue: bool = False,
                lue_uncertainty_weight: float = 0.5,
                lue_warmup_epochs: int = 5,
                lue_giou_weighting: bool = False,
                lue_giou_start_epoch: int = 40,
                lue_giou_warmup_epochs: int = 5,
                use_dqcd: bool = False,
                dqcd_temperature: float = 0.15,
                dqcd_weight: float = 0.3,
                dqcd_hard_negatives_k: int = 128,
                dqcd_start_epoch: int = 8,
                dqcd_warmup_epochs: int = 0,
                dqcd_decay_start_epoch: int = -1,
                dqcd_decay_epochs: int = 25,
                dqcd_final_weight: float = -1.0,
                use_dafd: bool = False,
                use_scu: bool = False,
                scu_salience_weight: float = 0.1,
                scu_calib_slope: float = -1.5,
                scu_calib_center: float = -4.2,
                # 🚀 新增：自适应参数管理
                use_adaptive_params: bool = True,
                innovation_strength: float = 1.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            group_detr: Number of groups to speed detr training. Default is 1.
            use_adaptive_params: 是否使用自适应参数管理（强烈推荐）
            innovation_strength: 创新点整体强度 [0.5-1.5]
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss
        self.mask_point_sample_ratio = mask_point_sample_ratio

        # 🚀 CVPR创新点配置
        self.use_lue = use_lue
        self.use_dqcd = use_dqcd
        self.use_dafd = use_dafd
        self.use_scu = use_scu
        self.current_epoch = 0

        # ⚠️ 关键修复：无条件定义所有参数（即使使用自适应参数）
        self.lue_uncertainty_weight = lue_uncertainty_weight
        self.lue_warmup_epochs = lue_warmup_epochs
        self.lue_giou_weighting = lue_giou_weighting
        self.lue_giou_start_epoch = lue_giou_start_epoch
        self.lue_giou_warmup_epochs = lue_giou_warmup_epochs
        self.dqcd_temperature = dqcd_temperature
        self.dqcd_weight = dqcd_weight
        self.dqcd_hard_negatives_k = dqcd_hard_negatives_k
        self.dqcd_start_epoch = dqcd_start_epoch
        self.dqcd_warmup_epochs = dqcd_warmup_epochs
        self.dqcd_decay_start_epoch = dqcd_decay_start_epoch
        self.dqcd_decay_epochs = dqcd_decay_epochs
        self.dqcd_final_weight = dqcd_final_weight
        self.scu_salience_weight = scu_salience_weight
        self.scu_calib_slope = scu_calib_slope
        self.scu_calib_center = scu_calib_center

        # 🚀 自适应参数管理器（三大创新点协同）
        self.use_adaptive_params = use_adaptive_params
        if use_adaptive_params:
            from rfdetr.util.adaptive_params import AdaptiveParamsManager
            self.param_manager = AdaptiveParamsManager(
                use_lue=use_lue,
                use_fafd=use_dafd,  # 内部用 fafd 命名但实际指向 DAFD
                use_qcd=use_dqcd,
                innovation_strength=innovation_strength,
                warmup_epochs=lue_warmup_epochs,
                qcd_base_weight=dqcd_weight,
                qcd_initial_temperature=dqcd_temperature,
                qcd_hard_negatives_k=dqcd_hard_negatives_k,
            )
            print("✅ 使用自适应参数管理器（三大创新点协同）")
        else:
            # 使用手动配置的参数（向后兼容）
            self.param_manager = None
            print("⚠️  使用手动配置参数（不推荐，建议启用自适应参数）")

        # diagnostics buffers (main process will read these)
        self._diag_lue_err = None
        self._diag_lue_logvar = None
        self._diag_dqcd_pos_sim = None
        self._diag_dqcd_neg_sim = None

        # ✅ Class-balanced loss: running EMA of class frequency (lazy-init in loss_labels)
        self.class_freq_ema = None

    def pop_diagnostics_payload(self, max_points: int = 2000):
        payload = {}

        def _cap(arr):
            if arr is None:
                return None
            if arr.numel() <= max_points:
                return arr
            idx = torch.randperm(arr.numel(), device=arr.device)[:max_points]
            return arr.flatten()[idx]

        if self._diag_lue_err is not None and self._diag_lue_logvar is not None:
            payload["lue_err"] = _cap(self._diag_lue_err).detach().cpu().numpy()
            payload["lue_logvar"] = _cap(self._diag_lue_logvar).detach().cpu().numpy()

        if self._diag_dqcd_pos_sim is not None and self._diag_dqcd_neg_sim is not None:
            payload["dqcd_pos_sim"] = _cap(self._diag_dqcd_pos_sim).detach().cpu().numpy()
            payload["dqcd_neg_sim"] = _cap(self._diag_dqcd_neg_sim).detach().cpu().numpy()

        self._diag_lue_err = None
        self._diag_lue_logvar = None
        self._diag_dqcd_pos_sim = None
        self._diag_dqcd_neg_sim = None
        return payload
    
    def set_epoch(self, epoch: int):
        """设置当前epoch，用于warm-up策略和自适应参数"""
        self.current_epoch = epoch
        if self.use_adaptive_params and self.param_manager is not None:
            self.param_manager.set_epoch(epoch)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2 
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            #init positive weights and negative weights
            pos_weights = torch.zeros_like(src_logits)
            neg_weights =  prob ** gamma

            pos_ind = tuple([id for id in idx] + [target_classes_o])

            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)

            # ❌ Removed: class-balanced re-weighting was suppressing dominant classes
            # by 50%, causing -2.1 mAP regression. The 0-AP rare classes (class_12, class_13)
            # are a data problem, not a loss-weighting problem; they should be addressed
            # in dataset preparation (oversampling, class merging, or removal).

            # a reformulation of the standard loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            # with a focus on statistical stability by using fused logsigmoid
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            # pos_ious_func = pos_ious ** 2
            pos_ious_func = pos_ious

            cls_iou_func_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind = tuple([id for id in idx] + [target_classes_o])
            cls_iou_func_targets[pos_ind] = pos_ious_func
            norm_cls_iou_func_targets = cls_iou_func_targets \
                / (cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(src_logits, norm_cls_iou_func_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind = tuple([id for id in idx] + [target_classes_o])
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = sigmoid_varifocal_loss(src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes
        
        v3.1矿井优化: 自适应混合策略 (小目标用NWD, 大目标用IoU)
        v4.0 CVPR创新: LUE不确定性感知的高斯建模
        
        Args:
            outputs: 模型输出，包含 'pred_boxes' 和可选的 'pred_log_vars'
            targets: Ground Truth
            indices: 匹配结果
            num_boxes: 归一化因子
            
        Returns:
            losses: dict包含 'loss_bbox', 'loss_giou', 可选 'loss_uncertainty'
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        # Compute warmup_ratio for LUE (controls when uncertainty loss activates)
        if self.use_adaptive_params and self.param_manager is not None:
            lue_params = self.param_manager.get_lue_params()
            warmup_ratio = lue_params['warmup_ratio'] if lue_params else 0.0
        else:
            if self.lue_warmup_epochs > 0:
                warmup_ratio = min(1.0, self.current_epoch / self.lue_warmup_epochs)
            else:
                warmup_ratio = 1.0

        # GIoU loss: Generalized IoU with higher weight (giou_loss_coef=5.0)
        # Precision weighting via lue_giou_weighting=True gives stronger gradients
        # for high-confidence predictions, boosting mAP@0.75 / mAP@0.95
        loss_giou_per_box = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))  # (N,)

        # ── L1 bbox loss ──────────────────────────────────────────────────────────
        # Base: plain L1 (identical to baseline)
        l1_per_box = F.l1_loss(src_boxes, target_boxes, reduction='none')  # (N, 4)

        if self.use_lue and 'pred_log_vars' in outputs and warmup_ratio > 0:
            src_log_vars = outputs['pred_log_vars'][idx]  # (N, 4)

            box_area = target_boxes[:, 2] * target_boxes[:, 3]  # normalized w*h

            # ① Active precision-weighted L1 for small/medium objects (core improvement)
            #
            # Motivation: LUE was purely passive (src_boxes.detach()) — it estimated
            # uncertainty but couldn't help the bbox head become more precise. This is
            # heteroscedastic regression: high-confidence predictions (low log_var →
            # high precision) receive stronger gradients → converge faster and more
            # precisely. Low-confidence predictions are softened → don't overfit to
            # noisy/occluded annotations.
            #
            # Implementation safety:
            # - log_var is DETACHED when computing precision → no circular gradient
            # - clamp(-2, 2): precision ∈ [0.14, 7.4], prevents gradient explosion
            # - precision_norm: normalized by batch mean → gradient scale ≈ plain L1
            # - only applied to small/medium objects (scale_gate) to protect APl
            # - warmup blend: pure L1 at ratio=0, precision-weighted at ratio=1
            #
            # Adaptive scale gate (dataset-agnostic):
            #   threshold = median(batch_area) * 3.0, clamped to [0.02, 0.15]
            #   小目标数据集 (Mine):   median≈0.003 → threshold≈0.009 → 几乎全部目标覆盖
            #   中等目标数据集 (ExDark): median≈0.03  → threshold≈0.09  → Car/People 覆盖
            #   大目标数据集 (COCO):   median≈0.05  → threshold≈0.15  → 自动偏大，保护 APl
            #   无需手工针对数据集调参，threshold 随 batch 分布自适应调整
            with torch.no_grad():
                median_area = box_area.median()
                dynamic_threshold = (median_area * 3.0).clamp(0.02, 0.15)
            small_mask = (box_area < dynamic_threshold).float().unsqueeze(1)  # (N, 1)
            # ① Active precision-weighted L1 — 中心化 (验证有效的原版)
            # 多次实验验证: 中心化 + clamp(-1.5, 1.1) 是稳定有效的配置。
            # - clamp 上界 1.1 严格于下界 1.5 (非对称 36%): 正向放大比反向抑制风险高，
            #   precision 上限 exp(1.5)=4.48 是早期 epoch 的安全边界。
            # - z-score 已验证失败 (-2.0 mAP@50): 小样本 std 噪声放大、尺度信息丢失。
            # - 残差版本失败 (逻辑矛盾): LUE 校准好时残差→0 → precision全为1 → 失效。
            # - 对称 clamp(-2,2) 失败 (-4.4 mAP@50 at ep3): 极端样本梯度过大破坏稳定性。
            log_var_d = src_log_vars.detach()
            log_var_centered = log_var_d - log_var_d.mean(dim=0, keepdim=True)
            precision = torch.exp(-log_var_centered.clamp(min=-1.5, max=1.1))   # (N, 4)
            precision_norm = precision / (precision.mean().detach() + 1e-6)
            # Active weighted L1: small/medium → precision_norm * L1; large → standard L1
            weighted_l1 = small_mask * precision_norm * l1_per_box + (1 - small_mask) * l1_per_box
            final_l1 = warmup_ratio * weighted_l1 + (1 - warmup_ratio) * l1_per_box
            losses['loss_bbox'] = final_l1.sum() / num_boxes

            # ✅ Fix 3: Precision-weighted GIoU for small/medium objects (high-IoU optimization)
            # LUE precision amplifies GIoU gradients for confident predictions → pulls
            # boxes from IoU 0.7→0.95 more effectively than plain GIoU, without the
            # center-distance/aspect-ratio noise that CIoU introduces in early training.
            if self.lue_giou_weighting and self.current_epoch >= self.lue_giou_start_epoch:
                if self.lue_giou_warmup_epochs > 0:
                    giou_weight_ratio = min(
                        1.0,
                        (self.current_epoch - self.lue_giou_start_epoch + 1)
                        / self.lue_giou_warmup_epochs,
                    )
                else:
                    giou_weight_ratio = 1.0
                precision_per_box = precision_norm.mean(dim=1)
                small_mask_1d = small_mask.squeeze(1)
                weighted_giou = (small_mask_1d * precision_per_box * loss_giou_per_box
                                  + (1 - small_mask_1d) * loss_giou_per_box)
                giou_blend = warmup_ratio * giou_weight_ratio
                final_giou = giou_blend * weighted_giou + (1 - giou_blend) * loss_giou_per_box
                losses['loss_giou'] = final_giou.sum() / num_boxes
                losses['diag/lue_giou_weight_ratio'] = torch.tensor(
                    giou_weight_ratio, device=src_boxes.device)
            else:
                # 旧 baseline 行为: 直接用原始 GIoU
                losses['loss_giou'] = loss_giou_per_box.sum() / num_boxes
                if self.lue_giou_weighting:
                    losses['diag/lue_giou_weight_ratio'] = torch.tensor(
                        0.0, device=src_boxes.device)

            # ② Uncertainty calibration: bounded log-error regression.
            # The target must stay inside the log-var head's reachable range. Very
            # small coordinate errors otherwise produce targets below -7 and an
            # irreducible loss that grows as localization improves.
            with torch.no_grad():
                raw_log_err_target = torch.log(l1_per_box.detach().clamp_min(1e-8))
                log_err_target = raw_log_err_target.clamp(min=-7.0, max=0.0)
                target_floor_ratio = (raw_log_err_target < -7.0).float().mean()
            loss_uncertainty_per_box = F.smooth_l1_loss(
                src_log_vars,
                log_err_target,
                reduction='none',
                beta=0.5,
            ).mean(dim=1)  # average cx, cy, w, h instead of summing them
            active_small_mask = small_mask.squeeze(1)
            loss_uncertainty = (loss_uncertainty_per_box * active_small_mask).sum()
            losses['loss_uncertainty'] = loss_uncertainty * warmup_ratio / num_boxes

            losses['diag/lue_warmup_ratio'] = torch.tensor(warmup_ratio)

            with torch.no_grad():
                err = torch.abs(src_boxes - target_boxes).mean(dim=1)
                unc = src_log_vars.mean(dim=1)
                if err.numel() > 1:
                    err_c = err - err.mean()
                    unc_c = unc - unc.mean()
                    corr = (err_c * unc_c).mean() / (err_c.std() * unc_c.std() + 1e-6)
                else:
                    corr = torch.tensor(0.0, device=err.device)
                losses['diag/lue_logvar_mean'] = src_log_vars.mean()
                losses['diag/lue_logvar_std'] = src_log_vars.std()
                losses['diag/lue_err_mean'] = err.mean()
                losses['diag/lue_err_unc_corr'] = corr
                losses['diag/lue_target_mean'] = log_err_target.mean()
                losses['diag/lue_target_floor_ratio'] = target_floor_ratio
                losses['diag/lue_scale_gate'] = dynamic_threshold  # 监控自适应阈值
                losses['diag/lue_small_ratio'] = small_mask.mean()  # 被 LUE 覆盖的目标比例
                self._diag_lue_err = err.detach()
                self._diag_lue_logvar = unc.detach()

            # ── 🚀 SCU: Salience-Calibrated Uncertainty ─────────────────────────
            # ✅ Fix 4 (reverted): SCU 绝对值校准已经正确。center=-4.0 与 LUE 收敛点
            # ~-4.6 在 s≈0.24 处完美重合，形成合理过渡区而非冲突。LUE weight 0.5 vs
            # SCU 0.1 的 5:1 比例让两个 supervision 信号互补而非对抗——SCU 提供几何先验，
            # LUE 提供数据驱动校准。rel-calib scheme tried and reverted: -3.5 mAP@50.
            if self.use_scu and 'pred_log_vars' in outputs and warmup_ratio > 0:
                gt_cxcy = target_boxes[:, :2]
                pred_cxcy = src_boxes[:, :2]
                d_center = torch.sqrt(((pred_cxcy - gt_cxcy) ** 2).sum(dim=1))
                sigma = torch.max(target_boxes[:, 2], target_boxes[:, 3]) / 2 + 1e-6
                salience = torch.exp(-d_center / sigma)
                calib_target = self.scu_calib_slope * salience + self.scu_calib_center
                src_log_vars_flat = src_log_vars.mean(dim=1, keepdim=True)
                scu_per_box = F.mse_loss(src_log_vars_flat, calib_target.unsqueeze(1).detach(), reduction='none')
                losses['loss_scu'] = (scu_per_box * small_mask).sum() * self.scu_salience_weight * warmup_ratio / num_boxes
                with torch.no_grad():
                    losses['diag/scu_salience_mean'] = salience.mean()
                    losses['diag/scu_calib_target'] = calib_target.mean()

        else:
            # LUE inactive or warmup not started: plain L1 + plain GIoU (identical to baseline)
            losses['loss_bbox'] = l1_per_box.sum() / num_boxes
            losses['loss_giou'] = loss_giou_per_box.sum() / num_boxes

        return losses
    
    def loss_dqcd(self, outputs, targets, indices, num_boxes):
        """Adaptive supervised InfoNCE over final decoder query embeddings."""
        device = next(iter(outputs.values())).device
        if not self.use_dqcd:
            return {'loss_dqcd': torch.tensor(0.0, device=device)}

        if self.current_epoch < self.dqcd_start_epoch:
            return {'loss_dqcd': torch.tensor(0.0, device=device)}

        if self.dqcd_warmup_epochs > 0:
            dqcd_warmup_ratio = min(
                1.0,
                (self.current_epoch - self.dqcd_start_epoch + 1)
                / self.dqcd_warmup_epochs,
            )
        else:
            dqcd_warmup_ratio = 1.0

        hs = outputs.get('hs')
        if hs is None or hs.ndim != 4:
            return {'loss_dqcd': torch.tensor(0.0, device=device)}

        hs = F.normalize(hs.float(), dim=-1)
        final_hs = hs[-1]
        batch_size, num_queries, _ = final_hs.shape
        active_groups = (
            self.group_detr
            if self.training and num_queries % self.group_detr == 0
            else 1
        )
        queries_per_group = num_queries // active_groups
        matched_masks = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=device)
        matched_records = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_idx = torch.as_tensor(src_idx, device=device, dtype=torch.long)
            tgt_idx = torch.as_tensor(tgt_idx, device=device, dtype=torch.long)
            # Group DETR repeats every target in each training group. DQCD uses
            # the first group only so duplicated GT matches are not treated as
            # independent same-class positives.
            keep = src_idx < queries_per_group
            src_idx = src_idx[keep]
            tgt_idx = tgt_idx[keep]
            if src_idx.numel() == 0:
                continue
            matched_masks[batch_idx, src_idx] = True
            labels = targets[batch_idx]['labels'][tgt_idx].to(device)
            matched_records.extend(
                (batch_idx, int(query_idx), int(label))
                for query_idx, label in zip(src_idx.tolist(), labels.tolist())
            )

        if not matched_records:
            return {'loss_dqcd': final_hs.sum() * 0.0}

        temperatures = torch.full(
            (batch_size,), self.dqcd_temperature, device=device, dtype=final_hs.dtype)
        band_gates = outputs.get('dafd_band_gates')
        if band_gates:
            per_band = [gate.detach().float().mean(dim=(1, 2, 3)) for gate in band_gates]
            gate_mean = torch.stack(per_band, dim=0).mean(dim=0)
            temperatures = self.dqcd_temperature * (0.5 + gate_mean)

        probabilities = outputs['pred_logits'].detach().float().sigmoid()
        top2 = probabilities.topk(k=min(2, probabilities.shape[-1]), dim=-1).values
        ambiguity = 1.0 - (top2[..., 0] - top2[..., -1]).abs()

        anchor_losses = []
        pos_sims_diag = []
        neg_sims_diag = []
        for batch_idx, query_idx, label in matched_records:
            anchor = final_hs[batch_idx, query_idx]
            positives = []
            if hs.shape[0] > 1:
                positives.append(hs[:-1, batch_idx, query_idx])

            same_class = [
                final_hs[b, q].unsqueeze(0)
                for b, q, other_label in matched_records
                if other_label == label and not (b == batch_idx and q == query_idx)
            ]
            if same_class:
                positives.append(torch.cat(same_class, dim=0))
            if not positives:
                continue
            positives = torch.cat(positives, dim=0)

            negative_idx = (~matched_masks[batch_idx, :queries_per_group]).nonzero(
                as_tuple=False).flatten()
            if negative_idx.numel() == 0:
                continue
            negatives = final_hs[batch_idx, negative_idx]
            neg_sim = negatives @ anchor
            hard_score = neg_sim.detach() + 0.25 * ambiguity[batch_idx, negative_idx]
            k = min(self.dqcd_hard_negatives_k, negatives.shape[0])
            if k <= 0:
                k = negatives.shape[0]
            hard_idx = hard_score.topk(k=k, largest=True).indices
            neg_sim = neg_sim[hard_idx]
            pos_sim = positives @ anchor

            temperature = temperatures[batch_idx].clamp_min(1e-4)
            pos_logits = pos_sim / temperature
            neg_logits = neg_sim / temperature
            log_numerator = torch.logsumexp(pos_logits, dim=0)
            log_denominator = torch.logsumexp(torch.cat([pos_logits, neg_logits]), dim=0)
            anchor_losses.append(log_denominator - log_numerator)
            pos_sims_diag.append(pos_sim.detach())
            neg_sims_diag.append(neg_sim.detach())

        if not anchor_losses:
            return {'loss_dqcd': final_hs.sum() * 0.0}

        self._diag_dqcd_pos_sim = torch.cat(pos_sims_diag)
        self._diag_dqcd_neg_sim = torch.cat(neg_sims_diag)
        pos_sim_mean = self._diag_dqcd_pos_sim.mean()
        neg_sim_mean = self._diag_dqcd_neg_sim.mean()
        effective_dqcd_weight = self.dqcd_weight * dqcd_warmup_ratio
        if (
            self.dqcd_decay_start_epoch >= 0
            and self.dqcd_final_weight >= 0
            and self.current_epoch >= self.dqcd_decay_start_epoch
        ):
            decay_horizon = max(1, self.dqcd_decay_epochs)
            decay_ratio = min(
                1.0,
                (self.current_epoch - self.dqcd_decay_start_epoch + 1)
                / decay_horizon,
            )
            effective_dqcd_weight = (
                self.dqcd_weight
                + (self.dqcd_final_weight - self.dqcd_weight) * decay_ratio
            )
        return {
            'loss_dqcd': (
                torch.stack(anchor_losses).mean()
                * effective_dqcd_weight
            ),
            'diag/dqcd_weight': torch.tensor(
                effective_dqcd_weight, device=device),
            'diag/dqcd_temperature': temperatures.mean().detach(),
            'diag/dqcd_temperature_mod': (
                temperatures / self.dqcd_temperature).mean().detach(),
            'diag/dqcd_anchors': torch.tensor(
                float(len(anchor_losses)), device=device),
            'diag/dqcd_pos_sim': pos_sim_mean.detach(),
            'diag/dqcd_neg_sim': neg_sim_mean.detach(),
            'diag/dqcd_negatives': torch.tensor(
                float(sum(item.numel() for item in neg_sims_diag)), device=device),
        }
    
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute BCE-with-logits and Dice losses for segmentation masks on matched pairs.
        Expects outputs to contain 'pred_masks' of shape [B, Q, H, W] and targets with key 'masks'.
        """
        assert 'pred_masks' in outputs, "pred_masks missing in model outputs"
        pred_masks = outputs['pred_masks']  # [B, Q, H, W]
        # gather matched prediction masks
        idx = self._get_src_permutation_idx(indices)
        src_masks = pred_masks[idx]  # [N, H, W]
        # handle no matches
        if src_masks.numel() == 0:
            return {
                'loss_mask_ce': src_masks.sum(),
                'loss_mask_dice': src_masks.sum(),
            }
        # gather matched target masks
        target_masks = torch.cat([t['masks'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, Ht, Wt]
        
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(src_masks.shape[-2], src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                num_points,
                3,
                0.75,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_ce": sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_boxes),
        }

        del src_masks
        del target_masks
        return losses
    
 
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'dqcd': self.loss_dqcd,  # 🚀 DQCD: 退化感知对比去噪
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'dafd_sparsity_loss' in outputs:
            losses['loss_dafd_sparsity'] = outputs['dafd_sparsity_loss']
        band_gates = outputs.get('dafd_band_gates')
        if band_gates:
            detached_gates = [gate.detach().float() for gate in band_gates]
            gate_means = torch.stack([gate.mean() for gate in detached_gates])
            losses['diag/dafd_gate_mean'] = gate_means.mean()
            losses['diag/dafd_gate_std'] = torch.stack(
                [gate.std() for gate in detached_gates]
            ).mean()
            for band_idx, gate_mean in enumerate(gate_means):
                losses[f'diag/dafd_gate_band_{band_idx}'] = gate_mean
        # 🚀 DQCD: 退化感知对比去噪
        if self.use_dqcd:
            losses.update(self.get_loss('dqcd', outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    focal_weight = targets * (targets > 0.0).float() + \
            (1 - alpha) * (prob - targets).abs().pow(gamma) * \
            (targets <= 0.0).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self,
        num_select=300,
        use_soft_nms=False,
        soft_nms_sigma=0.5,
        soft_nms_iou_threshold=0.5,
        use_lue_quality_score=False,
        lue_quality_gamma=0.25,
        lue_quality_center=-5.0,
        lue_quality_max_delta=4.0,
    ) -> None:
        super().__init__()
        self.num_select = num_select
        self.use_soft_nms = use_soft_nms
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_iou_threshold = soft_nms_iou_threshold
        self.use_lue_quality_score = use_lue_quality_score
        self.lue_quality_gamma = float(lue_quality_gamma)
        self.lue_quality_center = float(lue_quality_center)
        self.lue_quality_max_delta = float(lue_quality_max_delta)

    def _lue_quality_from_log_vars(self, log_vars):
        """Convert coordinate log-variance to a bounded localization quality."""
        mean_log_var = log_vars.mean(dim=-1)
        delta = (mean_log_var - self.lue_quality_center).clamp(
            min=0.0,
            max=max(self.lue_quality_max_delta, 0.0),
        )
        return torch.exp(-self.lue_quality_gamma * delta).clamp(min=1e-4, max=1.0)

    def _apply_soft_nms(self, boxes, scores, labels):
        """Class-wise Gaussian Soft-NMS score decay for duplicate DETR queries."""
        if boxes.numel() == 0:
            return scores

        score_dtype = scores.dtype
        boxes = boxes.float()
        updated_scores = scores.float().clone()
        sigma = max(float(self.soft_nms_sigma), 1e-6)
        iou_threshold = float(self.soft_nms_iou_threshold)

        for label in labels.unique():
            cls_idx = (labels == label).nonzero(as_tuple=False).flatten()
            if cls_idx.numel() <= 1:
                continue

            cls_scores = updated_scores[cls_idx].clone()
            order = torch.argsort(cls_scores, descending=True)
            while order.numel() > 1:
                best_local = order[0]
                rest = order[1:]
                best_box = boxes[cls_idx[best_local]].unsqueeze(0)
                rest_boxes = boxes[cls_idx[rest]]
                ious = box_ops.box_iou(best_box, rest_boxes)[0].squeeze(0)
                decay = torch.ones_like(ious)
                overlap = ious > iou_threshold
                decay[overlap] = torch.exp(-(ious[overlap] * ious[overlap]) / sigma)
                cls_scores[rest] = cls_scores[rest] * decay
                order = order[1:][torch.argsort(cls_scores[order[1:]], descending=True)]

            updated_scores[cls_idx] = cls_scores

        return updated_scores.to(dtype=score_dtype)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_masks = outputs.get('pred_masks', None)
        out_log_vars = outputs.get('pred_log_vars', None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        raw_prob = out_logits.sigmoid()
        prob = raw_prob
        lue_quality = None
        if self.use_lue_quality_score and out_log_vars is not None:
            lue_quality = self._lue_quality_from_log_vars(out_log_vars)
            prob = raw_prob * lue_quality.unsqueeze(-1)
        
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        raw_scores = torch.gather(raw_prob.view(out_logits.shape[0], -1), 1, topk_indexes)
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # LUE: 为被选中的 top-K queries 计算每个检测框的不确定性（标量）
        # 使用 4 个坐标 log_var 的均值作为每个框的整体不确定性度量
        uncertainties = None
        selected_quality = None
        if out_log_vars is not None:
            # out_log_vars: [B, Q, 4]，与 out_bbox 对齐
            gathered_logvars = torch.gather(
                out_log_vars, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_log_vars.shape[-1])
            )
            uncertainties = gathered_logvars.mean(dim=-1)
            if lue_quality is not None:
                selected_quality = torch.gather(lue_quality, 1, topk_boxes)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.use_soft_nms:
            scores = torch.stack([
                self._apply_soft_nms(boxes[i], scores[i], labels[i])
                for i in range(scores.shape[0])
            ], dim=0)

        # Optionally gather masks corresponding to the same top-K queries and resize to original size
        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                res_i = {'scores': scores[i], 'labels': labels[i], 'boxes': boxes[i]}
                if uncertainties is not None:
                    res_i['uncertainty'] = uncertainties[i]
                if selected_quality is not None:
                    res_i['lue_quality'] = selected_quality[i]
                    res_i['scores_raw'] = raw_scores[i]
                k_idx = topk_boxes[i]
                masks_i = torch.gather(
                    out_masks[i],
                    0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(
                        1, out_masks.shape[-2], out_masks.shape[-1]
                    ),
                )  # [K, Hm, Wm]
                h, w = target_sizes[i].tolist()
                masks_i = F.interpolate(
                    masks_i.unsqueeze(1),
                    size=(int(h), int(w)),
                    mode='bilinear',
                    align_corners=False,
                )  # [K,1,H,W]
                res_i['masks'] = masks_i > 0.0
                results.append(res_i)
        else:
            for i, (s, l, b, u) in enumerate(zip(
                scores,
                labels,
                boxes,
                uncertainties if uncertainties is not None else [None] * scores.shape[0],
            )):
                res = {'scores': s, 'labels': l, 'boxes': b}
                if u is not None:
                    res['uncertainty'] = u
                if selected_quality is not None:
                    res['lue_quality'] = selected_quality[i]
                    res['scores_raw'] = raw_scores[i]
                results.append(res)

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes + 1
    device = torch.device(args.device)


    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=args.shape if hasattr(args, 'shape') else (args.resolution, args.resolution) if hasattr(args, 'resolution') else (640, 640),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
        use_dafd=getattr(args, 'use_dafd', False),
        dafd_sparsity_weight=getattr(args, 'dafd_sparsity_weight', 0.0),
        # 🚀 DAFD: 退化感知频域分解，默认3频带
        dafd_alpha=getattr(args, 'dafd_alpha', 0.15),
        dafd_n_bands=getattr(args, 'dafd_n_bands', 3),
        # 🚀 v6.0: 共享退化估计器 — 仅当显式开启时才创建
        # (不再自动启用; DAFD/DQCD 可以在没有 Estimator 的情况下独立工作)
        use_degradation_estimator=getattr(args, 'use_degradation_estimator', False),
    )
    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    segmentation_head = SegmentationHead(args.hidden_dim, args.dec_layers, downsample_ratio=args.mask_downsample_ratio) if args.segmentation_head else None

    # 🚀 CVPR v5.0: DAFD + DQCD + SCU
    use_lue = getattr(args, 'use_lue', False)
    use_dqcd = getattr(args, 'use_dqcd', False)
    use_dafd = getattr(args, 'use_dafd', False)
    use_scu = getattr(args, 'use_scu', False)

    model = LWDETR(
        backbone,
        transformer,
        segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
        use_lue=use_lue,
        use_dqcd=use_dqcd,
        use_dafd=use_dafd,
        use_scu=use_scu,
        dags_start_epoch=getattr(args, 'dags_start_epoch', getattr(args, 'dqcd_start_epoch', 8)),
        dags_warmup_epochs=getattr(args, 'dags_warmup_epochs', 5),
        dags_bonus_scale=getattr(args, 'dags_bonus_scale', 0.0),
    )
    return model

def build_criterion_and_postprocessors(args):
    device = torch.device(args.device)
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.segmentation_head:
        weight_dict['loss_mask_ce'] = args.mask_ce_loss_coef
        weight_dict['loss_mask_dice'] = args.mask_dice_loss_coef

    # Read innovation flags before building aux_weight_dict so aux layers inherit them
    use_lue = getattr(args, 'use_lue', False)
    lue_uncertainty_weight = getattr(args, 'lue_uncertainty_weight', 0.5)
    lue_warmup_epochs = getattr(args, 'lue_warmup_epochs', 15)
    lue_giou_weighting = getattr(args, 'lue_giou_weighting', False)
    lue_giou_start_epoch = getattr(args, 'lue_giou_start_epoch', 40)
    lue_giou_warmup_epochs = getattr(args, 'lue_giou_warmup_epochs', 5)
    use_dqcd = getattr(args, 'use_dqcd', False)
    dqcd_temperature = getattr(args, 'dqcd_temperature', 0.15)
    dqcd_weight = getattr(args, 'dqcd_weight', 0.3)
    dqcd_hard_negatives_k = getattr(args, 'dqcd_hard_negatives_k', 128)
    dqcd_start_epoch = getattr(args, 'dqcd_start_epoch', 8)
    dqcd_warmup_epochs = getattr(args, 'dqcd_warmup_epochs', 0)
    dqcd_decay_start_epoch = getattr(args, 'dqcd_decay_start_epoch', -1)
    dqcd_decay_epochs = getattr(args, 'dqcd_decay_epochs', 25)
    dqcd_final_weight = getattr(args, 'dqcd_final_weight', -1.0)
    use_scu = getattr(args, 'use_scu', False)
    scu_salience_weight = getattr(args, 'scu_salience_weight', 0.1)
    scu_calib_slope = getattr(args, 'scu_calib_slope', -1.5)
    scu_calib_center = getattr(args, 'scu_calib_center', -4.2)
    dafd_sparsity_weight = getattr(args, 'dafd_sparsity_weight', 0.0)

    # LUE: register uncertainty as independent auxiliary loss
    if use_lue:
        lue_uncertainty_coef = getattr(args, 'lue_uncertainty_coef', 0.5)
        weight_dict['loss_uncertainty'] = lue_uncertainty_coef

    if dafd_sparsity_weight and dafd_sparsity_weight > 0:
        weight_dict['loss_dafd_sparsity'] = dafd_sparsity_weight

    # 🚀 DQCD: weight 已内嵌到 loss 中，weight_dict 中设为 1.0
    if use_dqcd:
        weight_dict['loss_dqcd'] = 1.0

    # 🚀 SCU: weight 已内嵌到 loss 中，weight_dict 中设为 1.0
    if use_scu:
        weight_dict['loss_scu'] = 1.0

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.segmentation_head:
        losses.append('masks')

    try:
        sum_group_losses = args.sum_group_losses
    except:
        sum_group_losses = False

    if args.segmentation_head:
        criterion = SetCriterion(args.num_classes + 1, matcher=matcher, weight_dict=weight_dict,
                                focal_alpha=args.focal_alpha, losses=losses,
                                group_detr=args.group_detr, sum_group_losses=sum_group_losses,
                                use_varifocal_loss = args.use_varifocal_loss,
                                use_position_supervised_loss=args.use_position_supervised_loss,
                                ia_bce_loss=args.ia_bce_loss,
                                mask_point_sample_ratio=args.mask_point_sample_ratio,
                                use_lue=use_lue,
                                lue_uncertainty_weight=lue_uncertainty_weight,
                                lue_warmup_epochs=lue_warmup_epochs,
                                lue_giou_weighting=lue_giou_weighting,
                                lue_giou_start_epoch=lue_giou_start_epoch,
                                lue_giou_warmup_epochs=lue_giou_warmup_epochs,
                                use_dqcd=use_dqcd,
                                dqcd_temperature=dqcd_temperature,
                                dqcd_weight=dqcd_weight,
                                dqcd_hard_negatives_k=dqcd_hard_negatives_k,
                                dqcd_start_epoch=dqcd_start_epoch,
                                dqcd_warmup_epochs=dqcd_warmup_epochs,
                                dqcd_decay_start_epoch=dqcd_decay_start_epoch,
                                dqcd_decay_epochs=dqcd_decay_epochs,
                                dqcd_final_weight=dqcd_final_weight,
                                use_scu=use_scu,
                                scu_salience_weight=scu_salience_weight,
                                scu_calib_slope=scu_calib_slope,
                                scu_calib_center=scu_calib_center)
    else:
        criterion = SetCriterion(args.num_classes + 1, matcher=matcher, weight_dict=weight_dict,
                                focal_alpha=args.focal_alpha, losses=losses,
                                group_detr=args.group_detr, sum_group_losses=sum_group_losses,
                                use_varifocal_loss = args.use_varifocal_loss,
                                use_position_supervised_loss=args.use_position_supervised_loss,
                                ia_bce_loss=args.ia_bce_loss,
                                use_lue=use_lue,
                                lue_uncertainty_weight=lue_uncertainty_weight,
                                lue_warmup_epochs=lue_warmup_epochs,
                                lue_giou_weighting=lue_giou_weighting,
                                lue_giou_start_epoch=lue_giou_start_epoch,
                                lue_giou_warmup_epochs=lue_giou_warmup_epochs,
                                use_dqcd=use_dqcd,
                                dqcd_temperature=dqcd_temperature,
                                dqcd_weight=dqcd_weight,
                                dqcd_hard_negatives_k=dqcd_hard_negatives_k,
                                dqcd_start_epoch=dqcd_start_epoch,
                                dqcd_warmup_epochs=dqcd_warmup_epochs,
                                dqcd_decay_start_epoch=dqcd_decay_start_epoch,
                                dqcd_decay_epochs=dqcd_decay_epochs,
                                dqcd_final_weight=dqcd_final_weight,
                                use_scu=use_scu,
                                scu_salience_weight=scu_salience_weight,
                                scu_calib_slope=scu_calib_slope,
                                scu_calib_center=scu_calib_center)
    criterion.to(device)
    postprocess = PostProcess(
        num_select=args.num_select,
        use_soft_nms=getattr(args, 'use_soft_nms', False),
        soft_nms_sigma=getattr(args, 'soft_nms_sigma', 0.5),
        soft_nms_iou_threshold=getattr(args, 'soft_nms_iou_threshold', 0.5),
        use_lue_quality_score=getattr(args, 'use_lue_quality_score', False),
        lue_quality_gamma=getattr(args, 'lue_quality_gamma', 0.25),
        lue_quality_center=getattr(args, 'lue_quality_center', -5.0),
        lue_quality_max_delta=getattr(args, 'lue_quality_max_delta', 4.0),
    )

    return criterion, postprocess
