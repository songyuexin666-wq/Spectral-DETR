# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

import os
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    from rfdetr.detr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRSegPreview
except ImportError:
    pass  # supervision not installed; inference-only usage still works
