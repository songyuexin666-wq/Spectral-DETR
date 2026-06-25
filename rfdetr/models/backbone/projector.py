# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
Projector
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.size(3),), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name, inplace=False):
    """ get activation """
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# ============================================================================
# 🚀 DAFD: Degradation-Aware Frequency Decomposition (v5.0)
# ============================================================================
# 相比 FAFD (v4.0) 的核心改进:
#   1. 单频段门控 → 多频带分解 (低频/中频/高频，可学习边界)
#   2. 各频带独立门控 + 跨频带交互 (轻量 1x1 conv 融合)
#   3. 场景自适应 FiLM 提升为 per-band 独立调制
#   4. 暴露 per-band 门控统计，供下游 DQCD 使用
#
# 与 DVT (ECCV 2024) 的区别:
#   DVT: 空域跨视图一致性 → 神经场分解 (两阶段训练，与检测任务无关)
#   DAFD: 频域多频带分解 → 可学习门控 (端到端训练，检测任务驱动)
# ============================================================================

class DAFDBlock(nn.Module):
    """
    退化感知频域分解模块 (Degradation-Aware Frequency Decomposition)

    核心创新:
        1. 多频带分解: 将频域分为低频(光照/结构)、中频(纹理)、高频(边缘/噪声)
           三个频带，使用可学习高斯掩码实现软分割
        2. 跨频带交互: 各频带独立门控后经 1x1 conv 融合，允许频带间互相补偿
        3. 任务驱动: 检测 loss 的梯度经反向传播直接指导各频带门控的学习

    物理可解释性:
        - 矿井低光照 → 低频衰减 → 低频门控增强
        - 粉尘噪声 → 高频污染 → 高频门控抑制
        - 运动模糊 → 中频损失 → 中频门控补偿
    """
    def __init__(
        self,
        channels,
        reduction=4,
        alpha: float = 0.15,
        n_bands: int = 3,
        target_keep: float = 0.8,
        entropy_weight: float = 0.0,
        deg_dim: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.alpha = float(alpha)
        self.n_bands = n_bands
        self.target_keep = float(target_keep)
        self.entropy_weight = float(entropy_weight)
        self._last_gates = None  # 存储 per-band gate 供 DQCD 使用

        # 可学习频带中心 + 宽度 (经过 sigmoid 约束到 [0,1])
        # 初始: 低频=0.1, 中频=0.35, 高频=0.6
        if n_bands == 1:
            center_init = torch.tensor([0.35])
            width_init = torch.tensor([0.25])
        elif n_bands == 3:
            center_init = torch.tensor([0.10, 0.35, 0.60])
            width_init = torch.tensor([0.12, 0.18, 0.25])
        else:
            center_init = torch.linspace(0.10, 0.60, steps=n_bands)
            width_init = torch.linspace(0.12, 0.25, steps=n_bands)
        center_unit = ((center_init - 0.05) / 0.60).clamp(1e-4, 1 - 1e-4)
        width_unit = ((width_init - 0.05) / 0.30).clamp(1e-4, 1 - 1e-4)
        self.band_centers = nn.Parameter(torch.logit(center_unit))
        self.band_widths = nn.Parameter(torch.logit(width_unit))

        # Per-band 频域门控 (独立参数，每个频带学自己的滤波策略)
        self.gate_bodies = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1),
            ) for _ in range(n_bands)
        ])
        for gate in self.gate_bodies:
            nn.init.constant_(gate[2].bias, 4.0)  # 初始近恒等
            nn.init.zeros_(gate[2].weight)

        # Per-band 场景自适应 FiLM (降质场景影响各频带的策略不同)
        # Reinitialize the final gate projections explicitly. The legacy line
        # above contains a zero-init call after a comment marker, so that call
        # is not executed. Starting at target_keep also avoids sigmoid
        # saturation and gives detection gradients room to specialize bands.
        gate_init = min(max(self.target_keep, 1e-4), 1.0 - 1e-4)
        gate_bias = math.log(gate_init / (1.0 - gate_init))
        for gate in self.gate_bodies:
            nn.init.constant_(gate[2].bias, gate_bias)
            nn.init.zeros_(gate[2].weight)

        self.scene_encoders = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, 2 * channels),
            ) for _ in range(n_bands)
        ])
        for enc in self.scene_encoders:
            nn.init.zeros_(enc[4].weight)
            nn.init.constant_(enc[4].bias[:channels], 1.0)  # gamma=1
            nn.init.constant_(enc[4].bias[channels:], 0.0)  # beta=0

        # 🚀 v6.0: 全局退化条件 FiLM — 从 DegradationEstimator 的 deg_global 直接调制
        # 每个频带独立的 deg→gamma/beta MLP。zero-init 确保初始不影响原有行为。
        self.deg_film = nn.ModuleList([
            nn.Sequential(
                nn.Linear(deg_dim, channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, 2 * channels),
            ) for _ in range(n_bands)
        ]) if deg_dim > 0 else None
        if self.deg_film is not None:
            for film in self.deg_film:
                nn.init.zeros_(film[2].weight)
                nn.init.zeros_(film[2].bias)  # zero-init → 初始无调制

        # 跨频带交互: 1x1 conv 融合三个频带的滤波结果
        self.cross_band_fusion = nn.Conv2d(
            channels * n_bands, channels, kernel_size=1
        )
        nn.init.zeros_(self.cross_band_fusion.weight)
        nn.init.zeros_(self.cross_band_fusion.bias)

    def _create_band_masks(self, H, W, device):
        """创建频带高斯软掩码，可学习边界且可微。
        ✅ Fix: 不做归一化, 让 mask 是真正的高斯滤波器 (值 ∈ [0,1] 但允许重叠)。
        归一化(让和=1)会让"关掉某频带"失去物理意义——因为剩下的 mask 仍能 100%
        覆盖该频点, 信号永远完整保留。去除归一化后, gate 真正能控制能量通过/抑制。
        """
        # Frequency coordinates follow the signed layout of torch.fft.rfft2.
        u = torch.fft.fftfreq(H, device=device)
        v = torch.fft.rfftfreq(W, device=device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        radius = torch.sqrt(grid_u**2 + grid_v**2)  # (H, W_fft)

        # 可学习频带参数，sigmoid 约束到合理范围
        centers = torch.sigmoid(self.band_centers) * 0.6 + 0.05   # [0.05, 0.65]
        widths = torch.sigmoid(self.band_widths) * 0.3 + 0.05     # [0.05, 0.35]

        masks = []
        for i in range(self.n_bands):
            # 高斯掩码 (软分割，可微)
            mask = torch.exp(-0.5 * ((radius - centers[i]) / (widths[i] + 1e-6)) ** 2)
            masks.append(mask[None, None, :, :])  # (1, 1, H, W_fft)

        # ✅ Fix: 不再归一化。每个 mask 独立, 值在 [0,1], 重叠区域同时被多个频带处理。
        return masks

    def forward(self, x, deg_global=None):
        """
        Args:
            x: (B, C, H, W) 输入特征图
            deg_global: (B, deg_dim) 可选，来自 DegradationEstimator 的全局退化嵌入
        Returns:
            out: (B, C, H, W) 频域分解重建后的特征图
        """
        B, C, H, W = x.shape
        self._last_gates = None

        # 小特征图跳过 (FFT 引入量化噪声)
        if H * W < 32 * 32:
            return x

        # 1) FFT
        x_fft = torch.fft.rfft2(x.float(), norm='backward')  # (B, C, H, W//2+1)

        # 2) 频带分解
        band_masks = self._create_band_masks(H, W, x_fft.device)

        # 3) Per-band 处理
        band_filtered = []
        band_gates = []
        for i in range(self.n_bands):
            # 提取该频带
            band_fft = x_fft * band_masks[i]  # (B, C, H, W//2+1)
            band_amp = torch.log1p(torch.abs(band_fft))  # 幅度谱

            # 场景自适应 FiLM (per-band, self-referential from band stats)
            scene = self.scene_encoders[i](band_amp)
            gamma = scene[:, :C].view(B, C, 1, 1)
            beta = scene[:, C:].view(B, C, 1, 1)

            # 🚀 v6.0: 全局退化条件调制 — 叠加到 scene FiLM 上
            if self.deg_film is not None and deg_global is not None:
                deg_film_out = self.deg_film[i](deg_global.to(x.dtype))  # (B, 2*C)
                gamma = gamma + deg_film_out[:, :C].view(B, C, 1, 1)
                beta  = beta  + deg_film_out[:, C:].view(B, C, 1, 1)

            # 门控
            raw_gate = self.gate_bodies[i](band_amp)
            gate = torch.sigmoid(raw_gate * gamma + beta)  # (B, C, H, W//2+1)
            band_gates.append(gate)

            band_filtered.append(band_fft * gate.float())

        # 4) 各频带独立 IFFT → 空间域
        band_spatial = []
        for band_f in band_filtered:
            band_s = torch.fft.irfft2(band_f, s=(H, W), norm='backward')
            band_spatial.append(band_s.to(dtype=x.dtype))

        # 5) 空间域跨频带融合 (避免复数卷积)
        band_concat = torch.cat(band_spatial, dim=1)                    # (B, C*n_bands, H, W)
        # AMP兼容: conv权重可能是bfloat16，手动对齐dtype
        target_dtype = self.cross_band_fusion.weight.dtype
        x_out = self.cross_band_fusion(band_concat.to(target_dtype))    # (B, C, H, W)

        # 存储 per-band gate 统计供 DQCD 使用
        gate_means = torch.stack([g.mean(dim=[0, 2, 3]) for g in band_gates], dim=0)  # (n_bands, C)
        gate_stds = torch.stack([g.std(dim=[0, 2, 3]) for g in band_gates], dim=0)    # (n_bands, C)
        self._last_gates = {
            'band_gates': band_gates,
            'gate_means': gate_means.detach(),
            'gate_stds': gate_stds.detach(),
        }

        # 6) Alpha blend (残差连接，稳定训练)
        return x + self.alpha * x_out

    def get_gate_stats(self):
        """获取各频带门控统计 (用于日志记录)"""
        if self._last_gates is None:
            return None
        stats = {}
        for i, gate in enumerate(self._last_gates['band_gates']):
            g = gate.detach()
            stats[f'band_{i}'] = {
                'mean': g.mean().item(),
                'std': g.std().item(),
                'sparsity': (g < 0.1).float().mean().item(),
            }
        return stats

    def get_sparsity_loss(self):
        """稀疏度损失: 鼓励 gate 选择性保留，避免坍缩"""
        if self._last_gates is None:
            return None
        loss = 0.0
        n = 0
        for gate in self._last_gates['band_gates']:
            keep_loss = (gate.mean() - self.target_keep) ** 2
            loss = loss + keep_loss
            n += 1
            if self.entropy_weight > 0:
                eps = 1e-6
                g = gate.clamp(min=eps, max=1.0 - eps)
                entropy = -(g * torch.log(g) + (1.0 - g) * torch.log(1.0 - g)).mean()
                loss = loss + self.entropy_weight * entropy
        return loss / n if n > 0 else None
    


class ConvX(nn.Module):
    """ Conv-bn module"""
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1, act='relu', layer_norm=False, rms_norm=False):
        super(ConvX, self).__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel,
                              stride=stride, padding=padding, groups=groups,
                              dilation=dilation, bias=False)
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = get_norm('LN', out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """ forward """
        out = self.act(self.bn(self.conv(x.contiguous())))
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, shortcut, groups, kernels, expand """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, number, shortcut, groups, expansion """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm) for _ in range(n))

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MultiScaleProjector(nn.Module):
    """
    This module implements MultiScaleProjector in :paper:`lwdetr`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        num_blocks=3,
        layer_norm=False,
        rms_norm=False,
        survival_prob=1.0,
        force_drop_last_n_features=0,
        use_dafd=False,  # 🚀 DAFD: 是否使用退化感知频域分解
        dafd_sparsity_weight=0.0,
        # DAFD 超参说明:
        # - dafd_alpha: 残差融合比例 (0=关闭滤波, 1=完全用滤波后特征)。建议 0.15~0.3
        # - dafd_n_bands: 频带数量 (默认 3: 低/中/高)
        # - dafd_target_keep: 频域 gate 的平均保留比例目标 (0.7~0.9)
        dafd_alpha: float = 0.15,
        dafd_n_bands: int = 3,
        dafd_target_keep: float = 0.8,
        dafd_entropy_weight: float = 0.0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            use_fafd (bool): CVPR v4.0 - 是否在每个输入特征上应用频域门控
        """
        super(MultiScaleProjector, self).__init__()

        self.scale_factors = scale_factors
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features
        self.use_dafd = use_dafd
        self.dafd_sparsity_weight = dafd_sparsity_weight
        self.dafd_alpha = float(dafd_alpha)
        self.dafd_n_bands = int(dafd_n_bands)
        self.dafd_target_keep = float(dafd_target_keep)
        self.dafd_entropy_weight = float(dafd_entropy_weight)

        # 🚀 DAFD: 为每个输入特征层级添加DAFD模块
        if self.use_dafd:
            self.dafd_layers = nn.ModuleList([
                DAFDBlock(
                    in_ch,
                    reduction=4,
                    alpha=self.dafd_alpha,
                    n_bands=self.dafd_n_bands,
                    target_keep=self.dafd_target_keep,
                    entropy_weight=self.dafd_entropy_weight,
                )
                for in_ch in in_channels
            ])

        stages_sampling = []
        stages = []
        # use_bias = norm == ""
        use_bias = False
        self.use_extra_pool = False
        for scale in scale_factors:
            stages_sampling.append([])
            for in_dim in in_channels:
                out_dim = in_dim
                layers = []

                # if in_dim > 512:
                #     layers.append(ConvX(in_dim, in_dim // 2, kernel=1))
                #     in_dim = in_dim // 2

                if scale == 4.0:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        get_norm('LN', in_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    ])
                    out_dim = in_dim // 4
                elif scale == 2.0:
                    # a hack to reduce the FLOPs and Params when the dimention of output feature is too large
                    # if in_dim > 512:
                    #     layers = [
                    #         ConvX(in_dim, in_dim // 2, kernel=1),
                    #         nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    #     ]
                    #     out_dim = in_dim // 4
                    # else:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                    ])
                    out_dim = in_dim // 2
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend([
                        ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                    ])
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])

            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            layers = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                get_norm('LN', out_channels),
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

        # 🚀 v6.0: 缓存全局退化嵌入，由 backbone 在 forward 中通过 set_deg_global 注入
        self._deg_global = None

    def set_deg_global(self, deg_global):
        """接收 DegradationEstimator 的全局退化嵌入 (B, deg_dim)，供 DAFD 使用。"""
        self._deg_global = deg_global

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # 🚀 DAFD: 在处理前对每个特征层级应用多频带分解
        # 🚀 v6.0: 传入全局退化嵌入，使 DAFD 的 FiLM 获得图像级退化上下文
        if self.use_dafd:
            deg = self._deg_global
            x = [self.dafd_layers[j](feat, deg_global=deg) for j, feat in enumerate(x)]
        
        num_features = len(x)
        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1 - self.survival_prob
            drop_p = np.random.uniform()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i][:] = 0
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                # don't do it inplace to ensure the compiler can optimize out the backbone layers
                x[-(i+1)] = torch.zeros_like(x[-(i+1)])
                
        results = []
        # x list of len(out_features_indexes)
        for i, stage in enumerate(self.stages):
            feat_fuse = []
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse.append(stage_sampling(x[j]))
            if len(feat_fuse) > 1:
                feat_fuse = torch.cat(feat_fuse, dim=1)
            else:
                feat_fuse = feat_fuse[0]
            results.append(stage(feat_fuse))
        if self.use_extra_pool:
            results.append(
                F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0)
            )
        return results

    def get_dafd_gate_stats(self):
        if not self.use_dafd:
            return None
        stats = [layer.get_gate_stats() for layer in self.dafd_layers if layer.get_gate_stats() is not None]
        if not stats:
            return None
        return stats  # 返回 per-layer, per-band 统计

    def get_dafd_sparsity_loss(self):
        if not self.use_dafd:
            return None
        losses = [layer.get_sparsity_loss() for layer in self.dafd_layers if layer.get_sparsity_loss() is not None]
        if not losses:
            return None
        return torch.stack(losses).mean()

    def get_dafd_gate_visual(self):
        if not self.use_dafd:
            return None
        for layer in self.dafd_layers:
            gates = getattr(layer, "_last_gates", None)
            if gates is not None and 'band_gates' in gates:
                return gates['band_gates']
        return None


class SimpleProjector(nn.Module):
    def __init__(self, in_dim, out_dim, factor_kernel=False):
        super(SimpleProjector, self).__init__()
        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim*2, layer_norm=True, act='silu')
            self.convx2 = ConvX(in_dim*2, out_dim, layer_norm=True, act='silu')
        else:
            self.convx1 = ConvX(in_dim, out_dim, kernel=(3, 1), layer_norm=True, act='silu')
            self.convx2 = ConvX(out_dim, out_dim, kernel=(1, 3), layer_norm=True, act='silu')
        self.ln = get_norm('LN', out_dim)

    def forward(self, x):
        """ forward """
        out = self.ln(self.convx2(self.convx1(x[0])))
        return [out]
