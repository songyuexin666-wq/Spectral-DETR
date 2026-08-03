# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Spectral-DETR
# ------------------------------------------------------------------------
"""
DegradationEstimator — shared, lightweight module that predicts per-image
degradation characteristics from raw pixel input.  Its outputs feed all
downstream reliability modules (DAFD / DDQCD / LUE), giving them a unified
degradation representation instead of each module estimating degradation
independently.

Design (v6.0):
  - A 3‑stage convolutional stem downsamples the input to 1/8 resolution.
  - A global MLP produces a 4‑dim degradation embedding [dark, blur, dust, clutter].
  - A spatial head produces a per‑pixel salience/prior map (1/8 resolution).

Cost: ~27K parameters, ~0.15 ms at 560×560 (negligible).
"""

import torch
import torch.nn as nn


class DegradationEstimator(nn.Module):
    """Predict degradation type, severity, and spatial prior from raw image.

    Outputs
    -------
    deg_global : (B, 4) ∈ [0,1]
        Four interpretable degradation dimensions:
        [darkness, blur_severity, dust_noise, clutter_level]
    spatial_prior : (B, 1, H/8, W/8) ∈ [0,1]
        Per-pixel "detectability" — low in degraded regions, high in clean regions.
        Used by DDQCD (query scoring) and LUE+ (multi-scale boost).
    feat : (B, C_deg, H/8, W/8)
        Intermediate features, available for downstream FiLM conditioning.
    """

    def __init__(self, stem_channels: int = 64, deg_dim: int = 4):
        super().__init__()
        self.deg_dim = deg_dim

        # -------- lightweight stem: 3×3 convs, stride‑2, 1/8 resolution --------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, stem_channels, 3, stride=2, padding=1),  # /8
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # -------- global degradation embedding --------
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stem_channels, stem_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(stem_channels // 4, deg_dim),
            nn.Sigmoid(),  # each dim ∈ [0,1]
        )

        # -------- spatial prior head --------
        self.spatial_head = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

        self._last_outputs = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) raw image tensor (values in [0, 1] after ToTensor,
               BEFORE ImageNet normalisation).
        Returns:
            deg_global:  (B, deg_dim)
            spatial_prior: (B, 1, H/8, W/8)
            feat:         (B, stem_channels, H/8, W/8)
        """
        # If input is already normalised (mean/std applied), denormalise
        # approximately so the stem sees raw-ish pixel statistics.
        # The stem's BatchNorm layers will handle the residual shift.
        self._last_input = x.detach()  # stored for pseudo-label aux loss
        feat = self.stem(x)
        deg_global = self.global_mlp(self.global_pool(feat))
        spatial_prior = self.spatial_head(feat)
        self._last_outputs = (deg_global, spatial_prior, feat)
        return deg_global, spatial_prior, feat

    def get_last_outputs(self):
        return self._last_outputs
