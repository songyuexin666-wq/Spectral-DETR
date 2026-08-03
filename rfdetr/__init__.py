# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Spectral-DETR
# ------------------------------------------------------------------------

import os
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rfdetr.detr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegPreview,
    RFDETRSmall,
)

__all__ = [
    "RFDETRBase",
    "RFDETRLarge",
    "RFDETRMedium",
    "RFDETRNano",
    "RFDETRSegPreview",
    "RFDETRSmall",
]
